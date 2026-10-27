"""Held-out evaluation of the hourly now-cast: does each added factor improve the estimate?

Split (by sounding date): train 2006-2020, validation 2021 (sets the green/yellow/red cut-offs),
test 2022 onward (never used to fit anything).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .dataset import load_summary
from .features import TARGETS
from sklearn.isotonic import IsotonicRegression

from .model import CLASSIFICATION, Predictors, block_bootstrap_ci, score
from .nowcast import FEATURE_SETS, LEADS, build_samples, surface_feature_table
from .surface import load_surface_archive

VAL_START, TEST_START = "2021-01-01", config.TEST_START
BAND_ORDER = ("green", "yellow", "red")


def _loss(target, y, p):
    return (p - y) ** 2 if target in CLASSIFICATION else np.abs(p - y)


def split3(samples: pd.DataFrame):
    return (samples[samples.index < VAL_START],
            samples[(samples.index >= VAL_START) & (samples.index < TEST_START)],
            samples[samples.index >= TEST_START])


def fit_presence_calibration(val_preds: dict, val: pd.DataFrame) -> dict:
    """Isotonic recalibration of the duct-present probability, fitted on the validation year only."""
    y = val["duct_present"].values
    return {m: IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(val_preds["duct_present"][m], y)
            for m in ("nn", "gbm")}


def apply_presence_calibration(preds: dict, cal: dict) -> dict:
    out = {t: dict(v) for t, v in preds.items()}
    for m, iso in cal.items():
        out["duct_present"][m] = iso.predict(preds["duct_present"][m])
    return out


class Light:
    """Green / yellow / red lights, one per output, from signals the model has at run time.

    * duct present?   how undecided the probability is, p(1 - p)
    * duct strength   spread of the ensemble members' predictions (M-units)
    * duct thickness  spread of the ensemble members' predictions (m)

    For each signal, green is at or below the validation-year median, red is above its 85th
    percentile, yellow is between. The cut-offs are fixed on the validation year before the test
    period is looked at. The overall light is the worst of the three. Red means: do not rely on
    this estimate, rerun the full simulation.

    Why three lights and not one: a confident "strong duct" has a large strength spread but a very
    certain presence, while a coin-flip presence usually means a weak or absent duct whose strength
    error is small. One shared score ranked the errors in opposite directions, so each output
    gets the signal that actually tracks its own error.
    """

    OUTPUTS = ("duct_present", "trap_strength_m", "trap_thickness_m")

    def __init__(self, val_preds: dict, green_q: float = 0.50, red_q: float = 0.85):
        sig = self._signals(val_preds)
        self.cuts = {k: (float(np.quantile(v, green_q)), float(np.quantile(v, red_q))) for k, v in sig.items()}

    @staticmethod
    def _signals(preds: dict) -> dict:
        p = preds["duct_present"]["nn"]
        return {"duct_present": p * (1 - p),
                "trap_strength_m": preds["trap_strength_m"]["nn_std"],
                "trap_thickness_m": preds["trap_thickness_m"]["nn_std"]}

    def colors(self, preds: dict) -> dict:
        """{output: array of 'green'/'yellow'/'red', ..., 'overall': worst of the three}."""
        out = {}
        for k, v in self._signals(preds).items():
            g, r = self.cuts[k]
            out[k] = np.where(v <= g, "green", np.where(v <= r, "yellow", "red"))
        rank = np.max([np.vectorize(BAND_ORDER.index)(out[k]) for k in self.OUTPUTS], axis=0)
        out["overall"] = np.array(BAND_ORDER)[rank]
        return out


def run(seeds: int = 5, verbose: bool = True) -> dict:
    summary = load_summary(config.DEFAULT_STATION)
    hourly = load_surface_archive()
    samples = build_samples(summary, surface_feature_table(hourly))
    train, val, test = split3(samples)
    if verbose:
        print(f"rows: train {len(train)}, validation {len(val)}, test {len(test)}", flush=True)
    models, preds, results, cal = {}, {}, {}, {}
    for name, feats in FEATURE_SETS.items():
        models[name] = Predictors(train, seeds=seeds, features=feats)
        cal[name] = fit_presence_calibration(models[name].predict(val), val)
        preds[name] = apply_presence_calibration(models[name].predict(test), cal[name])
        results[name] = {"all_leads": score(test, preds[name])}
        for lead in LEADS:
            m = (test["lead"] == lead).values
            sub = {t: {k: v[m] for k, v in preds[name][t].items()} for t in TARGETS}
            results[name][f"lead_{lead}h"] = score(test[m], sub)
        if verbose:
            print("fitted", name, flush=True)

    # does each step of added information help the neural network?
    steps = list(FEATURE_SETS)
    gain = {}
    groups = test.index.to_period("M").astype(str).values
    for a, b in zip(steps[:-1], steps[1:]):
        gain[f"{b}  vs  {a}"] = {}
        for t in TARGETS:
            y = test[t].values.astype(float)
            d = _loss(t, y, preds[b][t]["nn"]) - _loss(t, y, preds[a][t]["nn"])
            ref = (np.mean(_loss(t, y, preds[a][t]["nn"]) ** (2 if t not in CLASSIFICATION else 1)))
            gain[f"{b}  vs  {a}"][t] = {"mean_loss_change": float(d.mean()), "ci95": block_bootstrap_ci(d, groups),
                                         "loss": "squared error of probability" if t in CLASSIFICATION else "absolute error"}
    # green / yellow / red from the final model
    final = steps[-1]
    lamp = Light(apply_presence_calibration(models[final].predict(val), cal[final]))
    lights = lamp.colors(preds[final])
    light = {"definition": " ".join(Light.__doc__.split()), "cutoffs": {k: {"green_up_to": v[0], "red_above": v[1]} for k, v in lamp.cuts.items()},
             "per_output": {}, "overall": {}}
    for key in ("duct_present", "trap_strength_m", "trap_thickness_m"):
        y_all = test[key].values.astype(float)
        light["per_output"][key] = {}
        for b in BAND_ORDER:
            m = lights[key] == b
            p = preds[final][key]["nn"][m]
            err = float(np.mean((p - y_all[m]) ** 2)) if key in CLASSIFICATION else float(np.mean(np.abs(p - y_all[m])))
            light["per_output"][key][b] = {"share": float(m.mean()), "n": int(m.sum()),
                                           "error": err, "error_measure": "Brier" if key in CLASSIFICATION else "mean absolute error"}
    for b in BAND_ORDER:
        m = lights["overall"] == b
        row = {"share": float(m.mean()), "n": int(m.sum())}
        for t in TARGETS:
            y = test[t].values.astype(float)[m]
            p = preds[final][t]["nn"][m]
            row[t] = float(np.mean((p - y) ** 2)) if t in CLASSIFICATION else float(np.mean(np.abs(p - y)))
        light["overall"][b] = row
    bundle = {"predictors": models[final], "calibration": cal[final], "light": lamp, "features": FEATURE_SETS[final],
              "trained_before": VAL_START, "calibrated_on": [VAL_START, TEST_START]}
    out = {"split": {"train_to": VAL_START, "validation": [VAL_START, TEST_START], "test_from": TEST_START,
                     "rows": {"train": len(train), "validation": len(val), "test": len(test)}},
           "feature_sets": {k: len(v) for k, v in FEATURE_SETS.items()},
           "results": results, "gain_from_added_information": gain, "light": light}
    return out, bundle


TARGET_LABEL = {"duct_present": "Duct present?", "trap_strength_m": "Duct strength (M-units)",
                "trap_thickness_m": "Duct thickness (m)"}


def _fmt(t, m):
    return f"Brier {m['brier']:.3f}" if t in CLASSIFICATION else f"MAE {m['mae']:.1f}, RMSE {m['rmse']:.1f}"


def write_outputs(out: dict, bundle: dict, results_dir=config.RESULTS_DIR, models_dir=config.DATA_DIR / "models") -> None:
    import pickle

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(models_dir) / "nowcast.pkl", "wb") as f:
        pickle.dump(bundle, f)
    write_report(out, results_dir)


def write_report(out: dict, results_dir=config.RESULTS_DIR) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "nowcast.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    sp = out["split"]
    L = ["# Hourly now-cast results (auto-generated by `python -m utr nowcast-eval`)", "",
         f"- Train: 2006 to {sp['train_to']} (exclusive), validation year sets calibration and light cut-offs, "
         f"test: {sp['test_from']} onward. Rows: {sp['rows']}.",
         "- Each row asks: L hours before a 00Z/12Z sounding, what is that sounding's duct state? L = 0, 3, 6, 9 h.",
         "- Skill = 1 - error / reference error (squared error; Brier for the duct flag). Positive = better than the reference.",
         "", "## Does each added piece of information help? (all lead times)", "",
         "| Target | Inputs | Neural net | Gradient boosting | Seasonal average | Keep last sounding |", "|---|---|---|---|---|---|"]
    for t in TARGETS:
        for name in FEATURE_SETS:
            r = out["results"][name]["all_leads"][t]
            L.append(f"| {TARGET_LABEL[t]} | {name} | {_fmt(t, r['nn'])} | {_fmt(t, r['gbm'])} | {_fmt(t, r['climatology'])} | {_fmt(t, r['persistence'])} |")
    L += ["", "## Gain from adding information (neural net loss change, negative = better; 95% CI over calendar months)", ""]
    for k, v in out["gain_from_added_information"].items():
        L.append(f"- {k}")
        for t, x in v.items():
            L.append(f"  - {TARGET_LABEL[t]}: {x['mean_loss_change']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}] ({x['loss']})")
    full = list(FEATURE_SETS)[-1]
    L += ["", f"## Full model ('{full}') by lead time", "", "| Lead | Duct present (Brier) | Strength RMSE | Thickness MAE |", "|---|---|---|---|"]
    for lead in LEADS:
        r = out["results"][full][f"lead_{lead}h"]
        L.append(f"| {lead} h before sounding | {r['duct_present']['nn']['brier']:.3f} (seasonal {r['duct_present']['climatology']['brier']:.3f}) | "
                 f"{r['trap_strength_m']['nn']['rmse']:.1f} (seasonal {r['trap_strength_m']['climatology']['rmse']:.1f}) | "
                 f"{r['trap_thickness_m']['nn']['mae']:.1f} (seasonal {r['trap_thickness_m']['climatology']['mae']:.1f}) |")
    li = out["light"]
    L += ["", "## Green / yellow / red", "", __import__("inspect").cleandoc(Light.__doc__), "", "| Output | Light | Share of estimates | Error in that band |", "|---|---|---|---|"]
    for key, bands in li["per_output"].items():
        for b in BAND_ORDER:
            x = bands[b]
            L.append(f"| {TARGET_LABEL[key]} | {b} | {x['share']:.0%} | {x['error']:.3f} ({x['error_measure']}) |")
    red_any = li["overall"]["red"]["share"]
    L += ["", f"Estimates with at least one red light (these would trigger a rerun of the full simulation): {red_any:.0%}."]
    (results_dir / "nowcast.md").write_text("\n".join(L) + "\n")

    # figure: skill vs seasonal average for each input set; error by light
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    names = list(FEATURE_SETS)
    w = 0.27
    for i, name in enumerate(names):
        vals = []
        for t in TARGETS:
            r = out["results"][name]["all_leads"][t]
            ref = r["climatology"]["brier"] if t in CLASSIFICATION else r["climatology"]["rmse"] ** 2
            mine = r["nn"]["brier"] if t in CLASSIFICATION else r["nn"]["rmse"] ** 2
            vals.append(1 - mine / ref)
        ax[0].bar(np.arange(3) + (i - 1) * w, vals, w, label=name)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xticks(range(3), ["Duct present", "Strength", "Thickness"])
    ax[0].set_ylabel("skill vs seasonal average")
    ax[0].set_title("Neural net, by inputs used")
    ax[0].legend(fontsize=7)
    colors = {"green": "#2e8b57", "yellow": "#e0b000", "red": "#c0392b"}
    for j, key in enumerate(("trap_strength_m", "trap_thickness_m")):
        errs = [li["per_output"][key][b]["error"] for b in BAND_ORDER]
        ax[1].bar(np.arange(3) + (j - 0.5) * 0.4, [e / errs[0] for e in errs], 0.4, color=[colors[b] for b in BAND_ORDER],
                  alpha=0.55 if j else 1.0, edgecolor="k", label=TARGET_LABEL[key])
    ax[1].set_xticks(range(3), list(BAND_ORDER))
    ax[1].set_ylabel("error relative to green band")
    ax[1].set_title("Error rises from green to red (test period)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(results_dir / "nowcast.png", dpi=140)
    plt.close(fig)
