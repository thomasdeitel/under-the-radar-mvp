"""Command line: python -m utr {fetch,check,evaluate,all}"""
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

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
