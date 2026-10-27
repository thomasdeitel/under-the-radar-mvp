"""How wrong is a coverage map built from the last sounding, and can the lights say so?

For every test sounding (the "truth"), compare the map built from the sounding 12 h earlier (what a
unit holds if it only updates twice a day) with the map built from the sounding itself, both with
the reference solver. Then ask whether signals available *before* the new sounding arrives -- the
hourly now-cast's estimate of how much the duct changed -- predict that map error.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config
from .coverage import CoverageMap, build_map, map_difference_db, usable_profile
from .dataset import load_summary
from .nowcast import build_samples, surface_feature_table
from .nowcast_eval import TEST_START, VAL_START, apply_presence_calibration
from .parse import iter_soundings
from .propagation import Scenario
from .surface import load_surface_archive

BAND = ("green", "yellow", "red")


def _signals(samples: pd.DataFrame, preds: dict) -> pd.DataFrame:
    """Run-time signals: how far the now-cast says things moved away from the last sounding."""
    s = pd.DataFrame(index=samples.index)
    s["lead"] = samples["lead"].values
    s["d_strength"] = np.abs(preds["trap_strength_m"]["nn"] - samples["src_trap_strength_m"].values)
    s["d_presence"] = np.abs(preds["duct_present"]["nn"] - samples["src_duct_present"].values)
    s["d_thickness"] = np.abs(preds["trap_thickness_m"]["nn"] - samples["src_trap_thickness_m"].values)
    s["nowcast_hedge"] = preds["duct_present"]["nn"] * (1 - preds["duct_present"]["nn"])
    return s


def _compact(m: CoverageMap) -> CoverageMap:
    """Keep every 5th range (beyond 5 km) and every 2nd height up to 300 m, as float32, to bound memory."""
    ri = np.flatnonzero(m.ranges_m >= 5000)[::5]
    hi = np.flatnonzero((m.heights_m >= 5) & (m.heights_m <= 300))[::2]
    return CoverageMap(m.ranges_m[ri], m.heights_m[hi], m.pf_db[np.ix_(ri, hi)].astype(np.float32), m.seconds, m.solver)


def run(verbose: bool = True) -> tuple[dict, object]:
    """Learn to predict how much a 12-hour-old map has moved, then turn the prediction into lights."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    from .nowcast import FEATURE_SETS
    sc = Scenario()
    feats = FEATURE_SETS[list(FEATURE_SETS)[-1]]
    summary = load_summary(config.DEFAULT_STATION)
    samples = build_samples(summary, surface_feature_table(load_surface_archive()))
    need = set(samples.index) | set(samples.index - pd.Timedelta(hours=12))
    maps = {}
    for s in iter_soundings(Path(config.RAW_DIR) / f"{config.DEFAULT_STATION}-data.txt.zip", 2005):
        if s.time in need and s.time not in maps:
            prof = usable_profile(s, sc)
            if prof is not None:
                maps[s.time] = _compact(build_map(prof, sc, solver="reference"))
    if verbose:
        print(f"reference maps: {len(maps)}", flush=True)
    keep = [t in maps and (t - pd.Timedelta(hours=12)) in maps for t in samples.index]
    samples = samples[keep].copy()
    some = next(iter(maps.values()))
    samples["map_error_db"] = [map_difference_db(maps[t - pd.Timedelta(hours=12)].pf_db, maps[t].pf_db,
                                                 some.ranges_m, some.heights_m) for t in samples.index]
    train = samples[samples.index < VAL_START]
    val = samples[(samples.index >= VAL_START) & (samples.index < TEST_START)]
    test = samples[samples.index >= TEST_START]
    model = HistGradientBoostingRegressor(random_state=0, max_iter=300, learning_rate=0.05).fit(
        train[feats], np.log1p(train["map_error_db"]))
    pv, pt = model.predict(val[feats]), model.predict(test[feats])
    cuts = (float(np.quantile(pv, 0.5)), float(np.quantile(pv, 0.85)))
    band = np.where(pt <= cuts[0], "green", np.where(pt <= cuts[1], "yellow", "red"))
    err = test["map_error_db"].values
    out = {"rows": {"train": len(train), "validation": len(val), "test": len(test)},
           "stale_map_error_db": {"mean": float(err.mean()), "median": float(np.median(err)), "p90": float(np.quantile(err, 0.9)),
                                  "by_year": {int(y): float(g.mean()) for y, g in test["map_error_db"].groupby(test.index.year)}},
           "predicted_vs_actual_spearman": float(spearmanr(pt, err).statistic),
           "reference_error_if_always_guess_mean_mae_db": float(np.mean(np.abs(err - np.median(train["map_error_db"])))),
           "model_mae_db": float(np.mean(np.abs(np.expm1(pt) - err))),
           "cutoffs_log1p_db": cuts,
           "by_band": {b: {"share": float((band == b).mean()), "n": int((band == b).sum()),
                           "mean_map_error_db": float(err[band == b].mean()), "p90_map_error_db": float(np.quantile(err[band == b], 0.9))}
                       for b in BAND}}
    return out, {"model": model, "cutoffs": cuts, "features": feats}


def write_outputs(out: dict, bundle: dict, results_dir=config.RESULTS_DIR, models_dir=config.DATA_DIR / "models") -> None:
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(models_dir) / "coverage_light.pkl", "wb") as f:
        pickle.dump(bundle, f)
    (Path(results_dir) / "coverage.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    s = out["stale_map_error_db"]
    L = ["# Coverage map staleness (auto-generated by `python -m utr coverage-eval`)", "",
         "A coverage map built from the sounding 12 h earlier is compared with the map built from the sounding itself, "
         "both with the reference solver, over cells beyond 5 km and 5-300 m high where either map is above -40 dB.",
         f"- Rows: {out['rows']} (train to 2020, validation 2021 fixes the cut-offs, test 2022 onward)",
         f"- Error of using the 12-hour-old map, test period: mean {s['mean']:.1f} dB, median {s['median']:.1f} dB, "
         f"90th percentile {s['p90']:.1f} dB. By year: " + ", ".join(f"{y}: {v:.1f}" for y, v in s["by_year"].items()),
         f"- A gradient-boosting model predicts how much the map will have moved, from the same inputs as the hourly now-cast. "
         f"Rank correlation with the real change: {out['predicted_vs_actual_spearman']:.2f}. "
         f"Mean absolute error {out['model_mae_db']:.2f} dB, against {out['reference_error_if_always_guess_mean_mae_db']:.2f} dB for always guessing the typical value.",
         "", "## Green / yellow / red for a coverage map", "",
         "| Light | Share of hours | Mean change in coverage map | 90th percentile |", "|---|---|---|---|"]
    for b in BAND:
        x = out["by_band"][b]
        L.append(f"| {b} | {x['share']:.0%} | {x['mean_map_error_db']:.1f} dB | {x['p90_map_error_db']:.1f} dB |")
    L += ["", "Red means the map has probably moved enough that the full simulation should be rerun."]
    (Path(results_dir) / "coverage.md").write_text("\n".join(L) + "\n")
