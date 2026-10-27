"""Command line: python -m utr {fetch,fetch-surface,check,evaluate,nowcast-eval,nowcast}"""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import config
from .dataset import load_summary
from .fetch import fetch_station
from .quality import check_series, write_report


def cmd_fetch(a) -> int:
    bad = 0
    for st in a.stations:
        r = fetch_station(st, a.raw_dir, a.shared_dir, force=a.force)
        status = "downloaded" if r["downloaded"] else r.get("note", "")
        print(f"[{st}] {status}")
        for p in r["problems"]:
            print(f"  PROBLEM: {p}")
            bad += 1
        if r.get("shared_path"):
            print(f"  published to {r['shared_path']}")
    return 1 if bad else 0


def cmd_check(a) -> int:
    worst = 0
    reports = {}
    for st in a.stations:
        df = load_summary(st, a.raw_dir, config.PROCESSED_DIR, rebuild=a.rebuild)
        rep = check_series(df, max_age_hours=a.max_age_hours)
        reports[st] = rep
        print(f"[{st}] {rep['status']}  newest={rep.get('newest_sounding')}  valid={rep.get('valid_fraction')}")
        for w in rep.get("warnings", []) + rep.get("errors", []):
            print(f"  - {w}")
        worst = max(worst, {"OK": 0, "WARN": 1, "FAIL": 2}[rep["status"]])
    write_report(reports, config.RESULTS_DIR / "data_quality.json")
    return worst


def cmd_evaluate(a) -> int:
    from .experiments import run, write_outputs
    t0 = time.time()
    res = run(leads=tuple(a.leads), seeds=a.seeds, rebuild=a.rebuild)
    write_outputs(res)
    print(f"wrote results/ in {time.time() - t0:.0f}s")
    return 0


def cmd_fetch_surface(a) -> int:
    from .surface import fetch_surface_archive
    bad = 0
    for r in fetch_surface_archive(a.raw_dir, force=a.force):
        for p in r["problems"]:
            if "404" not in p:          # the current year's archive file does not exist yet
                print(f"  PROBLEM: {p}")
                bad += 1
    print("surface archive up to date" if not bad else "surface archive has problems")
    return 1 if bad else 0


def cmd_nowcast_eval(a) -> int:
    from .nowcast_eval import run, write_outputs
    t0 = time.time()
    out, bundle = run(seeds=a.seeds)
    write_outputs(out, bundle)
    print(f"wrote results/nowcast.* and data/models/nowcast.pkl in {time.time() - t0:.0f}s")
    return 0


def cmd_nowcast(a) -> int:
    from .nowcast import runtime_row, surface_feature_table
    from .nowcast_eval import apply_presence_calibration
    import pickle
    import pandas as pd
    model_path = config.DATA_DIR / "models" / "nowcast.pkl"
    if not model_path.exists():
        print("no trained model: run `python -m utr nowcast-eval` first")
        return 2
    bundle = pickle.load(open(model_path, "rb"))
    if a.time:
        from .surface import load_surface_archive
        hourly = load_surface_archive()
        now = pd.Timestamp(a.time)
    else:
        from .surface import fetch_live_surface
        hourly = fetch_live_surface()
        now = hourly.dropna(subset=["t_c", "sst_c"]).index.max()
    obs_age = (pd.Timestamp.now("UTC").tz_localize(None) - now) / pd.Timedelta(hours=1)
    warnings = []
    if not a.time and obs_age > config.SURFACE_MAX_AGE_HOURS:
        warnings.append(f"STALE: newest complete hourly observation is {obs_age:.0f} h old")
    row, info = runtime_row(load_summary(config.DEFAULT_STATION), surface_feature_table(hourly), now)
    preds = apply_presence_calibration(bundle["predictors"].predict(row), bundle["calibration"])
    lights = {k: str(v[0]) for k, v in bundle["light"].colors(preds).items() if k != "overall"}
    age = info["hours_since_sounding"]
    order = ["green", "yellow", "red"]
    if age > 27:
        warnings.append(f"STALE: last sounding is {age:.0f} h old; the model was tested with soundings up to 12 h old")
        cap = "red"
    elif age > 15:
        warnings.append(f"AGING: last sounding is {age:.0f} h old; capping lights at yellow")
        cap = "yellow"
    else:
        cap = "green"
    lights = {k: order[max(order.index(v), order.index(cap))] for k, v in lights.items()}
    overall = order[max(order.index(v) for v in lights.values())]
    result = {
        "valid_at": str(now), "estimate_for_sounding": str(info["target_sounding"]),
        "last_sounding_used": str(info["last_sounding"]), "hours_since_sounding": round(age, 1),
        "duct_probability": round(float(preds["duct_present"]["nn"][0]), 3),
        "duct_strength_m_units": round(float(preds["trap_strength_m"]["nn"][0]), 1),
        "duct_thickness_m": round(float(preds["trap_thickness_m"]["nn"][0]), 0),
        "lights": lights, "overall_light": overall,
        "action": "rerun the full simulation" if overall == "red" else
                  ("use with care" if overall == "yellow" else "use the update"),
        "warnings": warnings,
    }
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "latest_nowcast.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 1 if warnings else 0


def cmd_coverage_eval(a) -> int:
    from .coverage_eval import run, write_outputs
    t0 = time.time()
    out, bundle = run()
    write_outputs(out, bundle)
    print(f"wrote results/coverage.* and data/models/coverage_light.pkl in {time.time() - t0:.0f}s")
    return 0


def cmd_coverage(a) -> int:
    """Coverage map from the latest sounding, with a light for how stale it has probably become."""
    import pickle

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from .coverage import COVERAGE_FLOOR_DB, FAST, build_map, usable_profile
    from .nowcast import runtime_row, surface_feature_table
    from .parse import iter_soundings
    from .propagation import Scenario
    path = config.DATA_DIR / "models" / "coverage_light.pkl"
    if not path.exists():
        print("no staleness model: run `python -m utr coverage-eval` first")
        return 2
    bundle = pickle.load(open(path, "rb"))
    if a.time:
        from .surface import load_surface_archive
        hourly, now = load_surface_archive(), pd.Timestamp(a.time)
    else:
        from .surface import fetch_live_surface
        hourly = fetch_live_surface()
        now = hourly.dropna(subset=["t_c", "sst_c"]).index.max()
    summary = load_summary(config.DEFAULT_STATION)
    row, info = runtime_row(summary, surface_feature_table(hourly), now)
    sc = Scenario()
    snd = None
    for s in iter_soundings(config.RAW_DIR / f"{config.DEFAULT_STATION}-data.txt.zip", 2020):
        if s.time == info["last_sounding"]:
            snd = s
    prof = usable_profile(snd, sc) if snd is not None else None
    if prof is None:
        print("the latest sounding is not usable for a coverage map")
        return 2
    cmap = build_map(prof, sc, solver="teng", cfg=FAST)
    pred = float(np.expm1(bundle["model"].predict(row[bundle["features"]])[0]))
    c0, c1 = (float(np.expm1(c)) for c in bundle["cutoffs"])
    light = "green" if pred <= c0 else ("yellow" if pred <= c1 else "red")
    age = info["hours_since_sounding"]
    warnings = []
    if age > 27:
        light = "red"
        warnings.append(f"STALE: last sounding is {age:.0f} h old; the staleness model was tested with soundings up to 12 h old")
    elif age > 15 and light == "green":
        light = "yellow"
        warnings.append(f"AGING: last sounding is {age:.0f} h old")
    heights = [10, 25, 50, 100, 200]
    result = {"valid_at": str(now), "map_built_from_sounding": str(info["last_sounding"]), "hours_since_sounding": round(age, 1),
              "solver": "TENG-style network, fast setting", "solve_seconds": round(cmap.seconds, 1),
              "expected_change_in_map_db": round(pred, 1), "light": light,
              "action": {"green": "use the map", "yellow": "use with care", "red": "rerun the full simulation"}[light],
              "farthest_covered_range_km": {f"{h} m": round(cmap.farthest_covered_km(h), 1) for h in heights},
              "covered_share_5_to_60_km": {f"{h} m": round(cmap.covered_fraction(h), 2) for h in heights},
              "covered_means": f"propagation factor above {COVERAGE_FLOOR_DB:.0f} dB (placeholder threshold, not a radar spec)",
              "warnings": warnings}
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "latest_coverage.json").write_text(json.dumps(result, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    im = ax.pcolormesh(cmap.ranges_m / 1000, cmap.heights_m, cmap.pf_db.T, vmin=-50, vmax=10, cmap="viridis", shading="auto")
    ax.contour(cmap.ranges_m / 1000, cmap.heights_m, cmap.pf_db.T, levels=[COVERAGE_FLOOR_DB], colors="w", linewidths=1)
    ax.set_ylim(0, 300)
    ax.set_xlabel("range (km)")
    ax.set_ylabel("height (m)")
    ax.set_title(f"Coverage map from sounding {info['last_sounding']:%Y-%m-%d %HZ}, light: {light.upper()}")
    fig.colorbar(im, label="propagation factor (dB)")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "latest_coverage.png", dpi=140)
    print(json.dumps(result, indent=2))
    return 1 if warnings else 0


def cmd_ship(a) -> int:
    """Offline workflow for an operator's own data: template, check, train, learn, predict."""
    from pathlib import Path

    from . import ship
    if a.ship_cmd == "template":
        for p in ship.write_template(a.out):
            print("wrote", p)
        return 0
    if a.ship_cmd == "demo":
        out = Path(a.out)
        sch = ship.synthetic_demo(out / "synthetic_demo.csv", days=a.days)
        sch.save(out / "schema.json")
        print(f"wrote {out / 'synthetic_demo.csv'} (SYNTHETIC) and {out / 'schema.json'}")
        return 0
    if a.ship_cmd == "check":
        schema = ship.Schema.load(a.schema)
        _, rep = ship.to_grid(ship.read_table(a.data, schema), schema)
        print(f"data check: {rep['status']}  rows={rep['rows_on_grid']}  {rep['first']} to {rep['last']}  ({rep['span_days']:.0f} days)")
        for w in rep["warnings"] + rep["problems"]:
            print("  -", w)
        print("  share missing:", {k: v for k, v in rep["missing_share"].items() if v > 0})
        return {"OK": 0, "WARN": 1, "FAIL": 2}[rep["status"]]
    if a.ship_cmd in ("train", "learn"):
        schema_path = a.schema or Path(a.model_dir) / "schema.json"
        schema = ship.Schema.load(schema_path)
        sm = ship.train(a.data, schema, a.model_dir)
        print((Path(a.model_dir) / "report.md").read_text())
        return 0
    if a.ship_cmd == "predict":
        pred, warnings = ship.predict(a.data, a.model_dir, last_only=not a.all)
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pred.to_csv(out, index=False)
        if a.daily:
            daily = ship.daily_summary(pred)
            daily.to_csv(out.with_name(out.stem + "_daily.csv"), index=False)
            print(daily.to_string(index=False))
        else:
            print(pred.tail(24).to_string(index=False))
        for w in warnings:
            print("WARNING:", w)
        print(f"wrote {out}")
        return 1 if warnings else 0
    return 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="utr", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    stations = [config.DEFAULT_STATION, config.STRESS_STATION]

    f = sub.add_parser("fetch", help="download NOAA radiosonde archives (only if changed) and verify them")
    f.add_argument("--stations", nargs="+", default=stations)
    f.add_argument("--raw-dir", default=config.RAW_DIR)
    f.add_argument("--shared-dir", default=None, help="folder the ship software reads from")
    f.add_argument("--force", action="store_true")
    f.set_defaults(func=cmd_fetch)

    c = sub.add_parser("check", help="warn about stale / missing / incomplete data")
    c.add_argument("--stations", nargs="+", default=stations)
    c.add_argument("--raw-dir", default=config.RAW_DIR)
    c.add_argument("--max-age-hours", type=float, default=config.MAX_AGE_HOURS)
    c.add_argument("--rebuild", action="store_true")
    c.set_defaults(func=cmd_check)

    e = sub.add_parser("evaluate", help="baselines vs neural network on held-out data")
    e.add_argument("--leads", nargs="+", type=int, default=[12, 24])
    e.add_argument("--seeds", type=int, default=5)
    e.add_argument("--rebuild", action="store_true")
    e.set_defaults(func=cmd_evaluate)

    fs = sub.add_parser("fetch-surface", help="download hourly airport + pier archives (NOAA ISD, NDBC)")
    fs.add_argument("--raw-dir", default=config.RAW_DIR)
    fs.add_argument("--force", action="store_true")
    fs.set_defaults(func=cmd_fetch_surface)

    ne = sub.add_parser("nowcast-eval", help="train and test the hourly now-cast (about 5 min)")
    ne.add_argument("--seeds", type=int, default=5)
    ne.set_defaults(func=cmd_nowcast_eval)

    n = sub.add_parser("nowcast", help="estimate the duct state from the latest observations (run hourly)")
    n.add_argument("--time", default=None, help="replay a past UTC hour from the archive instead of live data")
    n.set_defaults(func=cmd_nowcast)

    ce = sub.add_parser("coverage-eval", help="learn how fast a coverage map goes stale (about 6 min)")
    ce.set_defaults(func=cmd_coverage_eval)

    cv = sub.add_parser("coverage", help="coverage map from the latest sounding plus a green/yellow/red staleness light")
    cv.add_argument("--time", default=None, help="replay a past UTC hour from the archive")
    cv.set_defaults(func=cmd_coverage)

    sh = sub.add_parser("ship", help="offline: learn from your own data and predict (no internet needed)")
    ss = sh.add_subparsers(dest="ship_cmd", required=True)
    t = ss.add_parser("template", help="write an input template and a schema file")
    t.add_argument("--out", default="ship_inputs")
    d = ss.add_parser("demo", help="write SYNTHETIC demonstration data to try the workflow")
    d.add_argument("--out", default="ship_demo")
    d.add_argument("--days", type=int, default=21)
    k = ss.add_parser("check", help="check a data file and report problems")
    k.add_argument("data")
    k.add_argument("--schema", required=True)
    for name, hlp in (("train", "learn from one or more data files"), ("learn", "add new data to what was learned before and retrain")):
        q = ss.add_parser(name, help=hlp)
        q.add_argument("data", nargs="+")
        q.add_argument("--schema", default=None, help="schema.json (not needed for `learn`)")
        q.add_argument("--model-dir", default="ship_model")
    pr = ss.add_parser("predict", help="predictions with green/yellow/red lights")
    pr.add_argument("data")
    pr.add_argument("--model-dir", default="ship_model")
    pr.add_argument("--out", default="ship_predictions.csv")
    pr.add_argument("--all", action="store_true", help="predict from every row (hourly table), not just the newest")
    pr.add_argument("--daily", action="store_true", help="also write a per-day summary")
    sh.set_defaults(func=cmd_ship)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
