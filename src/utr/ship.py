"""Bring-your-own-data workflow for an operator who cannot share data with us.

The operator gives the program a CSV of their own observations (weather, sea temperature,
anything numeric, plus the quantities they want predicted). The program checks the data, learns
from it, and then predicts those quantities for each hour or day, with a green/yellow/red light.
Everything runs locally with numpy, pandas and scikit-learn. This module makes no network calls.

Two ways to use it, chosen by `horizon_hours` in the schema:
  * horizon > 0  forecasting: from everything known at time t, predict the target at t + horizon.
  * horizon = 0  surrogate: learn the mapping from the inputs at time t to the target at time t
                 (for example, atmospheric inputs -> the output of a slow simulation run now and then).

Nothing about the operator's data is assumed beyond: a time column and numeric columns.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BAND = ("green", "yellow", "red")


# ---------------------------------------------------------------- schema

@dataclass
class Schema:
    time_column: str = "time"
    targets: list = field(default_factory=list)
    feature_columns: list | None = None        # None = every other numeric column
    angle_columns: list = field(default_factory=list)   # degrees; turned into sin and cos
    horizon_hours: float = 3.0
    step_hours: float = 1.0                    # data are averaged onto this grid
    lags_hours: list = field(default_factory=lambda: [3, 12, 24])
    valid_ranges: dict = field(default_factory=dict)    # {"air_temp_c": [-40, 60]} -> values outside become missing
    max_gap_hours: float = 6.0                 # warn about longer holes in the inputs
    max_input_age_hours: float = 6.0           # at prediction time, warn if the newest row is older
    test_fraction: float = 0.2
    validation_fraction: float = 0.2
    seeds: int = 5

    @classmethod
    def load(cls, path) -> "Schema":
        d = json.loads(Path(path).read_text())
        d = {k: v for k, v in d.items() if not k.startswith("_")}
        return cls(**d)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


# ---------------------------------------------------------------- intake and quality checks

def read_table(path, schema: Schema) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    if schema.time_column not in df.columns:
        raise ValueError(f"no '{schema.time_column}' column; found: {list(df.columns)[:12]}")
    df[schema.time_column] = pd.to_datetime(df[schema.time_column], utc=True, errors="coerce")
    bad = int(df[schema.time_column].isna().sum())
    df = df.dropna(subset=[schema.time_column])
    df[schema.time_column] = df[schema.time_column].dt.tz_localize(None)
    df = df.set_index(schema.time_column).sort_index()
    df.attrs["unparseable_times"] = bad
    return df


def to_grid(df: pd.DataFrame, schema: Schema) -> tuple[pd.DataFrame, dict]:
    """Average onto a regular grid, blank out impossible values, and report what was found."""
    report: dict = {"problems": [], "warnings": []}
    num = df.apply(pd.to_numeric, errors="coerce")
    non_numeric = [c for c in df.columns if num[c].notna().sum() == 0]
    num = num.drop(columns=non_numeric)
    if non_numeric:
        report["warnings"].append(f"ignored non-numeric columns: {non_numeric}")
    if df.attrs.get("unparseable_times"):
        report["warnings"].append(f"{df.attrs['unparseable_times']} rows had unreadable times and were dropped")
    dup = int(num.index.duplicated().sum())
    if dup:
        report["warnings"].append(f"{dup} duplicate timestamps were averaged")
    for col, (lo, hi) in schema.valid_ranges.items():
        if col in num:
            out = int(((num[col] < lo) | (num[col] > hi)).sum())
            if out:
                report["warnings"].append(f"{col}: {out} values outside [{lo}, {hi}] were treated as missing")
            num.loc[(num[col] < lo) | (num[col] > hi), col] = np.nan
    missing_t = [t for t in schema.targets if t not in num]
    if missing_t:
        report["problems"].append(f"target columns not found (or not numeric): {missing_t}")
    step = f"{int(round(schema.step_hours * 60))}min"
    g = num.resample(step).mean()
    report["rows_in"], report["rows_on_grid"] = int(len(df)), int(len(g))
    report["first"], report["last"] = str(g.index.min()), str(g.index.max())
    report["span_days"] = float((g.index.max() - g.index.min()) / pd.Timedelta(days=1)) if len(g) else 0.0
    report["missing_share"] = {c: round(float(g[c].isna().mean()), 3) for c in g.columns}
    # holes: runs of fully empty grid steps
    empty = g.drop(columns=[t for t in schema.targets if t in g], errors="ignore").isna().all(axis=1)
    if len(g) and empty.any():
        runs = (empty != empty.shift()).cumsum()
        longest = int(empty.groupby(runs).sum().max()) * schema.step_hours
        if longest > schema.max_gap_hours:
            report["warnings"].append(f"longest hole with no inputs at all is {longest:.0f} h")
    for t in schema.targets:
        if t in g and g[t].notna().sum() < 30:
            report["problems"].append(f"target '{t}' has only {int(g[t].notna().sum())} observations; need at least 30 to learn")
    if report["span_days"] < 14:
        report["warnings"].append(f"only {report['span_days']:.0f} days of data; results will be rough until there is more")
    report["status"] = "FAIL" if report["problems"] else ("WARN" if report["warnings"] else "OK")
    return g, report


# ---------------------------------------------------------------- features

def feature_frame(g: pd.DataFrame, schema: Schema, inputs: list[str] | None = None) -> pd.DataFrame:
    """Everything known at time t (and the past), one row per grid time."""
    inputs = inputs or [c for c in (schema.feature_columns or g.columns) if c in g and c not in schema.targets]
    step = schema.step_hours
    cols: dict[str, pd.Series] = {}
    for c in inputs:
        s = g[c].ffill(limit=int(round(3 / step)))          # bridge short holes only
        if c in schema.angle_columns:
            rad = np.deg2rad(s)
            cols[f"{c}__sin"], cols[f"{c}__cos"] = np.sin(rad), np.cos(rad)
            continue
        cols[c] = s
        for lag in schema.lags_hours:
            cols[f"{c}__chg{lag}h"] = s - s.shift(int(round(lag / step)))
        win = int(round(6 / step))
        cols[f"{c}__mean6h"] = s.rolling(win, min_periods=max(1, win // 2)).mean()
    for t in schema.targets:                                 # past values of the thing being predicted
        known = g[t].shift(1) if schema.horizon_hours == 0 else g[t]    # same-time value is the answer, so not an input
        last = known.ffill(limit=int(round(72 / step)))
        age = (~known.notna()).astype(int).groupby(known.notna().cumsum()).cumsum() * step
        cols[f"{t}__last"] = last
        cols[f"{t}__age_h"] = age.where(last.notna())
        cols[f"{t}__last24h_ago"] = last.shift(int(round(24 / step)))
    hr = g.index.hour + g.index.minute / 60
    cols["hour_sin"], cols["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    span = (g.index.max() - g.index.min()) / pd.Timedelta(days=1) if len(g) else 0
    if span >= 300:                                          # seasons only mean something with about a year
        doy = g.index.dayofyear
        cols["doy_sin"], cols["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    X = pd.DataFrame(cols, index=g.index)
    return X.loc[:, X.notna().any()]


def labelled(g: pd.DataFrame, X: pd.DataFrame, schema: Schema, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Features at t paired with the target at t + horizon (only where the target was observed)."""
    h = int(round(schema.horizon_hours / schema.step_hours))
    y = g[target].shift(-h)
    ok = y.notna()
    return X[ok], y[ok]


# ---------------------------------------------------------------- learners

def _is_binary(y: pd.Series) -> bool:
    return set(np.unique(y.dropna())) <= {0.0, 1.0}


class _Member:
    """One fitted model with a uniform predict() returning a number (probability for 0/1 targets)."""

    def __init__(self, kind: str, binary: bool, seed: int = 0):
        self.kind, self.binary = kind, binary
        if kind == "ridge":
            est = LogisticRegression(C=0.3, max_iter=2000) if binary else Ridge(alpha=10.0)
        elif kind == "gbm":
            kw = dict(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=10, random_state=seed)
            est = HistGradientBoostingClassifier(**kw) if binary else HistGradientBoostingRegressor(**kw)
        else:
            kw = dict(hidden_layer_sizes=(16,), alpha=1.0, max_iter=600, early_stopping=False, random_state=seed)
            est = MLPClassifier(**kw) if binary else MLPRegressor(**kw)
        self.pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")          # small-data convergence chatter is not useful to the operator
            self.pipe.fit(X.values, y.values)
        return self

    def predict(self, X) -> np.ndarray:
        if self.binary:
            return self.pipe.predict_proba(X.values)[:, 1]
        return self.pipe.predict(X.values)


def _loss(y: np.ndarray, p: np.ndarray, binary: bool) -> float:
    return float(np.mean((p - y) ** 2)) if binary else float(np.sqrt(np.mean((p - y) ** 2)))


@dataclass
class TargetModel:
    target: str
    binary: bool
    columns: list
    members: dict            # kind -> list of fitted members
    used: list               # kinds that beat the baseline on validation
    baseline: str            # "last value" or "mean"
    train_mean: float
    train_ranges: dict
    cut: tuple               # (green_up_to, red_above) on the disagreement signal
    val_scores: dict
    note: str = ""

    def _ensemble(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Mean prediction and a disagreement signal (members within a model type, and between types)."""
        X = X.reindex(columns=self.columns)
        if not self.used:
            return self.baseline_values(X), np.zeros(len(X))
        per_kind = {k: np.array([m.predict(X) for m in self.members[k]]) for k in self.used}
        means = np.array([v.mean(0) for v in per_kind.values()])
        within = np.mean([v.std(0) for v in per_kind.values()], axis=0)
        between = means.std(0) if len(means) > 1 else np.zeros(len(X))
        mean = means.mean(0)
        spread = np.sqrt(within**2 + between**2)
        if self.binary:
            spread = spread + 0.5 * mean * (1 - mean)            # undecided probabilities count too
        return mean, spread

    def baseline_values(self, X: pd.DataFrame) -> np.ndarray:
        col = f"{self.target}__last"
        if self.baseline == "last value" and col in X:
            return X[col].fillna(self.train_mean).values
        return np.full(len(X), self.train_mean)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        mean, spread = self._ensemble(X)
        if self.binary:
            mean = np.clip(mean, 0, 1)
        g, r = self.cut
        light = np.where(spread <= g, "green", np.where(spread <= r, "yellow", "red")).astype(object)
        reason = np.array([""] * len(X), dtype=object)
        Xc = X.reindex(columns=self.columns)
        outside = np.zeros(len(X))
        for c, (lo, hi, sd) in self.train_ranges.items():
            v = Xc[c].values
            outside += ((v < lo - sd) | (v > hi + sd)).astype(float)
        ood = outside / max(len(self.train_ranges), 1) > 0.10
        light[ood & (light == "green")] = "yellow"
        reason[ood] = "inputs outside the range seen in training"
        if not self.used:
            light[light == "green"] = "yellow"
            reason[:] = np.where(reason == "", "no learned skill over the simple baseline yet", reason)
        return pd.DataFrame({"prediction": mean, "light": light, "note": reason, "spread": spread}, index=X.index)


def fit_target(g: pd.DataFrame, X: pd.DataFrame, schema: Schema, target: str) -> tuple[TargetModel, dict]:
    Xl, y = labelled(g, X, schema, target)
    n = len(y)
    binary = _is_binary(y)
    n_test = max(int(n * schema.test_fraction), 10)
    n_val = max(int(n * schema.validation_fraction), 10)
    if n < 60:
        raise ValueError(f"target '{target}' has only {n} usable examples; need at least 60 to learn")
    tr, va, te = slice(0, n - n_val - n_test), slice(n - n_val - n_test, n - n_test), slice(n - n_test, n)
    Xtr, ytr, Xva, yva, Xte, yte = Xl.iloc[tr], y.iloc[tr], Xl.iloc[va], y.iloc[va], Xl.iloc[te], y.iloc[te]
    cols = list(Xl.columns[Xtr.notna().any()])
    Xtr, Xva, Xte = Xtr[cols], Xva[cols], Xte[cols]
    mean = float(ytr.mean())

    def base(Xp, kind):
        if kind == "last value" and f"{target}__last" in Xp:
            return Xp[f"{target}__last"].fillna(mean).values
        return np.full(len(Xp), mean)

    # which simple baseline is better on validation? the learned model must beat it
    bl = min(("last value", "mean"), key=lambda k: _loss(yva.values, base(Xva, k), binary))
    members: dict = {}
    for kind, n_seeds in (("ridge", 1), ("gbm", 1), ("nn", schema.seeds)):
        members[kind] = [_Member(kind, binary, s).fit(Xtr, ytr) for s in range(n_seeds)]
    val_scores = {"baseline": _loss(yva.values, base(Xva, bl), binary)}
    for kind, ms in members.items():
        val_scores[kind] = _loss(yva.values, np.mean([m.predict(Xva) for m in ms], axis=0), binary)
    used = [k for k in members if val_scores[k] < val_scores["baseline"]]
    # refit the chosen members on train + validation so the newest data is learned from
    Xfit, yfit = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])
    members = {k: [_Member(k, binary, s).fit(Xfit, yfit) for s in range(len(members[k]))] for k in members}
    ranges = {c: (float(Xfit[c].min()), float(Xfit[c].max()), float(Xfit[c].std() or 1.0)) for c in cols}
    tm = TargetModel(target, binary, cols, members, used, bl, float(yfit.mean()), ranges, (0.0, 0.0), val_scores)
    # light cut-offs: validation-period disagreement, from models fitted on train only
    members_tr = {k: [_Member(k, binary, s).fit(Xtr, ytr) for s in range(len(v))] for k, v in members.items()}
    probe = TargetModel(target, binary, cols, members_tr, used, bl, mean, ranges, (0.0, 0.0), val_scores)
    _, spread_va = probe._ensemble(Xva)
    if used and len(spread_va) >= 20:
        tm.cut = (float(np.quantile(spread_va, 0.5)), float(np.quantile(spread_va, 0.85)))
        tm.note = "light cut-offs fixed on the validation block"
    else:
        sd = float(ytr.std() or 1.0)
        tm.cut = (0.25 * sd, 0.6 * sd)
        tm.note = "too little validation data for percentile cut-offs; using fractions of the target's spread"
    # honest test on the final block (never used for fitting, cut-offs or model choice)
    pt, st = probe._ensemble(Xte)
    if used:
        te_pred = pt
        te_light = probe_light(probe_cut=tm.cut, spread=st)
    else:
        te_pred, te_light = base(Xte, bl), np.array(["yellow"] * len(Xte))
    test = {"n_train": int(len(Xtr)), "n_validation": int(len(Xva)), "n_test": int(len(Xte)), "binary": binary,
            "loss_name": "Brier" if binary else "RMSE",
            "test": {"model": _loss(yte.values, te_pred, binary), "baseline_" + bl.replace(" ", "_"): _loss(yte.values, base(Xte, bl), binary),
                     "baseline_mean": _loss(yte.values, np.full(len(Xte), mean), binary)},
            "validation": val_scores, "models_used": used, "baseline_used": bl,
            "by_light": {}}
    err = (te_pred - yte.values) ** 2 if binary else np.abs(te_pred - yte.values)
    for b in BAND:
        m = te_light == b
        test["by_light"][b] = {"n": int(m.sum()), "error": float(err[m].mean()) if m.any() else None}
    bl_loss = test["test"]["baseline_" + bl.replace(" ", "_")]
    if bl_loss <= 0:
        test["skill_vs_baseline"] = 0.0
    else:
        test["skill_vs_baseline"] = 1 - (test["test"]["model"] / bl_loss) ** (1 if binary else 2)
    return tm, test


def probe_light(probe_cut, spread):
    g, r = probe_cut
    return np.where(spread <= g, "green", np.where(spread <= r, "yellow", "red"))


# ---------------------------------------------------------------- the store: data seen so far + model

@dataclass
class ShipModel:
    schema: Schema
    inputs: list
    targets: dict                 # name -> TargetModel
    trained_utc: str
    data_fingerprint: str
    rows_seen: int
    report: dict
    history: list = field(default_factory=list)


def _fingerprint(g: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(g, index=True).values.tobytes()).hexdigest()[:16]


def train(csv_paths, schema: Schema, model_dir) -> ShipModel:
    """Read the data, check it, learn from it, and save everything to `model_dir`."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    store = model_dir / "data_store.csv"
    frames = []
    if store.exists():
        frames.append(pd.read_csv(store, parse_dates=[schema.time_column], index_col=schema.time_column))
    for p in ([csv_paths] if isinstance(csv_paths, (str, Path)) else csv_paths):
        frames.append(read_table(p, schema))
    raw = pd.concat(frames).sort_index()
    raw = raw.groupby(level=0).last() if raw.index.duplicated().any() else raw   # newer files win
    g, report = to_grid(raw, schema)
    if report["problems"]:
        raise ValueError("; ".join(report["problems"]))
    raw.index.name = schema.time_column
    raw.to_csv(store)
    base_inputs = [c for c in (schema.feature_columns or g.columns) if c in g and c not in schema.targets]
    X = feature_frame(g, schema, inputs=base_inputs)
    models, tests = {}, {}
    for t in schema.targets:
        models[t], tests[t] = fit_target(g, X, schema, t)
    prev = []
    mpath = model_dir / "model.pkl"
    if mpath.exists():
        prev = pickle.load(open(mpath, "rb")).history
    sm = ShipModel(schema, base_inputs, models, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   _fingerprint(g), int(len(g)), {"data": report, "tests": tests})
    sm.history = prev + [{"trained_utc": sm.trained_utc, "rows": sm.rows_seen, "first": report["first"], "last": report["last"],
                          "test_skill": {t: tests[t]["skill_vs_baseline"] for t in tests}}]
    with open(mpath, "wb") as f:
        pickle.dump(sm, f)
    schema.save(model_dir / "schema.json")
    (model_dir / "report.json").write_text(json.dumps(sm.report, indent=2, default=float) + "\n")
    (model_dir / "report.md").write_text(render_report(sm))
    return sm


def render_report(sm: ShipModel) -> str:
    d, t = sm.report["data"], sm.report["tests"]
    L = ["# What the model learned from your data", "",
         f"- Trained {sm.trained_utc} on {d['rows_on_grid']} time steps from {d['first']} to {d['last']} ({d['span_days']:.0f} days).",
         f"- Data check: **{d['status']}**" + "".join(f"\n  - {w}" for w in d["warnings"]),
         f"- Mode: {'surrogate (same-time inputs to target)' if sm.schema.horizon_hours == 0 else 'forecast ' + str(sm.schema.horizon_hours) + ' h ahead'}",
         "", "## Held-out test (the newest data, never used for fitting, model choice or light cut-offs)", "",
         "| Target | Examples (train/val/test) | Model | Simple baseline | Skill vs baseline | Learned models used |", "|---|---|---|---|---|---|"]
    for name, r in t.items():
        bl = r["baseline_used"]
        L.append(f"| {name} | {r['n_train']}/{r['n_validation']}/{r['n_test']} | {r['loss_name']} {r['test']['model']:.3g} | "
                 f"{r['loss_name']} {r['test']['baseline_' + bl.replace(' ', '_')]:.3g} ({bl}) | {r['skill_vs_baseline']:+.2f} | "
                 f"{', '.join(r['models_used']) or 'none: no skill over the baseline yet'} |")
    L += ["", "Skill is 1 minus (model error / baseline error); positive means the model is better than the baseline. "
          "With only a few weeks of data the test block is small, so treat these as a first read, not a verdict.",
          "", "## Do the lights mean anything on your data?", "", "| Target | Green error (n) | Yellow error (n) | Red error (n) |", "|---|---|---|---|"]
    for name, r in t.items():
        cells = []
        for b in BAND:
            x = r["by_light"][b]
            cells.append("-" if x["error"] is None else f"{x['error']:.3g} ({x['n']})")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "Error should rise from green to red. If it does not, the lights are not yet trustworthy; more data will fix that.",
          "", "## Training history", ""]
    for h in sm.history:
        L.append(f"- {h['trained_utc']}: {h['rows']} steps ({h['first']} to {h['last']}); test skill " +
                 ", ".join(f"{k} {v:+.2f}" for k, v in h["test_skill"].items()))
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- prediction

def predict(csv_path, model_dir, last_only: bool = True) -> tuple[pd.DataFrame, list]:
    """Predictions for the target at (issue time + horizon), from inputs up to each issue time."""
    sm: ShipModel = pickle.load(open(Path(model_dir) / "model.pkl", "rb"))
    schema = sm.schema
    g, report = to_grid(read_table(csv_path, schema), Schema(**{**asdict(schema), "targets": []}))
    warnings = list(report["warnings"])
    g = g.reindex(columns=sorted(set(g.columns) | set(schema.targets)))
    X = feature_frame(g, schema, inputs=[c for c in sm.inputs if c in g.columns])
    missing = [c for c in sm.inputs if c not in g.columns]
    if missing:
        warnings.append(f"input columns the model was trained with are absent: {missing[:8]}")
    if last_only:
        X = X.iloc[[-1]]
    now = pd.Timestamp.now("UTC").tz_localize(None)
    newest = g.dropna(how="all").index.max()
    age = (now - newest) / pd.Timedelta(hours=1)
    if age > schema.max_input_age_hours and last_only:
        warnings.append(f"STALE: newest input row is {age:.0f} h old (limit {schema.max_input_age_hours:.0f} h)")
    out = []
    for t, tm in sm.targets.items():
        p = tm.predict(X)
        p.insert(0, "target", t)
        p.insert(0, "valid_time", p.index + pd.Timedelta(hours=schema.horizon_hours))
        p.index.name = "issue_time"
        out.append(p.reset_index())
    res = pd.concat(out, ignore_index=True)
    if last_only and age > schema.max_input_age_hours:
        res.loc[res["light"] == "green", "light"] = "yellow"
        res["note"] = res["note"].where(res["note"] != "", "inputs are not current")
    return res.drop(columns="spread"), warnings


def daily_summary(pred: pd.DataFrame) -> pd.DataFrame:
    """Per day and target: mean, lowest and highest prediction, and one light for the day.

    The day is red if a third or more of its hours are red, yellow if any hour is red or a third
    or more are yellow, otherwise green.
    """
    d = pred.copy()
    d["day"] = pd.to_datetime(d["valid_time"]).dt.date
    d["is_red"], d["is_yellow"] = (d["light"] == "red").astype(float), (d["light"] == "yellow").astype(float)
    s = d.groupby(["day", "target"]).agg(mean=("prediction", "mean"), low=("prediction", "min"), high=("prediction", "max"),
                                         share_red=("is_red", "mean"), share_yellow=("is_yellow", "mean"),
                                         hours=("prediction", "size")).reset_index()
    s["light"] = np.where(s["share_red"] >= 1 / 3, "red", np.where((s["share_red"] > 0) | (s["share_yellow"] >= 1 / 3), "yellow", "green"))
    return s


# ---------------------------------------------------------------- templates and a synthetic demo

TEMPLATE_COLUMNS = ["time", "air_temp_c", "dewpoint_c", "pressure_hpa", "wind_speed_ms", "wind_dir_deg",
                    "sea_surface_temp_c", "evaporation_duct_height_m", "radar_detection_range_km"]


def write_template(out_dir) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "input_template.csv").write_text(
        "# One row per observation time, UTC. Leave a cell empty when a value is missing.\n"
        "# Add, remove or rename columns freely; every numeric column is used. Targets can be sparse.\n"
        + ",".join(TEMPLATE_COLUMNS) + "\n"
        "2026-11-16T00:00:00Z,18.2,12.1,1016.2,4.1,250,17.6,14.5,\n"
        "2026-11-16T01:00:00Z,18.0,12.4,1016.0,4.8,255,17.6,,\n")
    Schema(targets=["evaporation_duct_height_m"], angle_columns=["wind_dir_deg"],
           valid_ranges={"air_temp_c": [-40, 60], "pressure_hpa": [850, 1080], "wind_speed_ms": [0, 80]}).save(out / "schema.json")
    return [out / "input_template.csv", out / "schema.json"]


def synthetic_demo(path, days: int = 42, seed: int = 0) -> Schema:
    """SYNTHETIC data with a known cause-and-effect, to show the workflow. Not real observations.

    The made-up 'evaporation duct height' rises with wind and with the sea being warmer than the air.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=days * 24, freq="h")
    h = idx.hour.values
    wind = np.clip(5 + 2.5 * np.sin(np.arange(len(idx)) / 37.0) + rng.normal(0, 1.2, len(idx)), 0.2, None)
    sst = 18 + 0.8 * np.sin(np.arange(len(idx)) / 200.0) + rng.normal(0, 0.05, len(idx)).cumsum() * 0.05
    air = sst - 1.5 + 2.0 * np.sin(2 * np.pi * (h - 9) / 24) + rng.normal(0, 0.4, len(idx))
    dew = air - 4 - 2 * np.sin(2 * np.pi * h / 24) + rng.normal(0, 0.5, len(idx))
    pres = 1015 + 4 * np.sin(np.arange(len(idx)) / 80.0) + rng.normal(0, 0.3, len(idx))
    wdir = (250 + 40 * np.sin(np.arange(len(idx)) / 55.0) + rng.normal(0, 10, len(idx))) % 360
    duct = 8 + 1.2 * wind + 2.0 * (sst - air) + 0.15 * (air - dew) + rng.normal(0, 1.0, len(idx))
    df = pd.DataFrame({"time": idx.strftime("%Y-%m-%dT%H:%M:%SZ"), "air_temp_c": air.round(2), "dewpoint_c": dew.round(2),
                       "pressure_hpa": pres.round(1), "wind_speed_ms": wind.round(1), "wind_dir_deg": wdir.round(0),
                       "sea_surface_temp_c": sst.round(2), "evaporation_duct_height_m": duct.round(1)})
    # the target is only measured every 3 hours, as a real instrument or simulation run might be
    df.loc[df.index % 3 != 0, "evaporation_duct_height_m"] = np.nan
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# SYNTHETIC demonstration data generated by `utr ship demo`. Not real observations.\n")
        df.to_csv(f, index=False)
    return Schema(targets=["evaporation_duct_height_m"], angle_columns=["wind_dir_deg"], horizon_hours=3.0)
