"""Automatic completeness checks for downloaded model/sounding data.

This is the 'check for missing or incomplete files' part of the MVP: it turns silent
data problems into explicit warnings (stale data, truncated files, soundings that
failed QC, missing synoptic times).
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from . import config


def check_archive(path) -> list[str]:
    """Return a list of problems with a downloaded archive (empty list means OK)."""
    path = Path(path)
    if not path.exists():
        return [f"{path.name}: missing"]
    problems = []
    if path.stat().st_size < 1024:
        problems.append(f"{path.name}: suspiciously small ({path.stat().st_size} bytes)")
        return problems
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad:
                problems.append(f"{path.name}: corrupt member {bad}")
            if not zf.namelist():
                problems.append(f"{path.name}: empty archive")
    except zipfile.BadZipFile:
        problems.append(f"{path.name}: not a valid zip (truncated download?)")
    return problems


def check_series(df: pd.DataFrame, now: pd.Timestamp | None = None,
                 max_age_hours: float | None = None, window_days: int = 14) -> dict:
    """Assess freshness and completeness of the sounding series built from the archive."""
    max_age_hours = config.MAX_AGE_HOURS if max_age_hours is None else max_age_hours
    now = pd.Timestamp.now("UTC").tz_localize(None) if now is None else pd.Timestamp(now)
    warnings, errors = [], []
    if df.empty:
        return {"status": "FAIL", "errors": ["no soundings parsed"], "warnings": []}

    newest = df.index.max()
    age_h = (now - newest).total_seconds() / 3600.0
    if age_h > max_age_hours:
        warnings.append(f"STALE: newest sounding is {age_h:.0f} h old (limit {max_age_hours:.0f} h)")

    recent = df[df.index >= now - pd.Timedelta(days=window_days)]
    expected = pd.date_range((now - pd.Timedelta(days=window_days)).floor("12h"), now.floor("12h"), freq="12h")
    expected = expected[expected <= newest + pd.Timedelta(hours=1)] if len(expected) else expected
    missing = [t for t in expected if t not in df.index]
    if recent.empty:
        warnings.append(f"no soundings in the last {window_days} days")
    elif missing:
        warnings.append(f"MISSING: {len(missing)} of {len(expected)} expected 00Z/12Z soundings "
                        f"in the last {window_days} days")
    if not recent.empty:
        bad = recent[~recent["valid"].astype(bool)]
        if len(bad):
            reasons = bad["qc_issues"].value_counts().head(3).to_dict()
            warnings.append(f"INCOMPLETE: {len(bad)} of {len(recent)} recent soundings failed QC {reasons}")

    valid_frac = float(df["valid"].astype(bool).mean())
    if valid_frac < 0.8:
        warnings.append(f"only {valid_frac:.0%} of all soundings pass QC")
    status = "FAIL" if errors else ("WARN" if warnings else "OK")
    return {
        "status": status, "newest_sounding": str(newest), "age_hours": round(age_h, 1),
        "soundings": int(len(df)), "valid_fraction": round(valid_frac, 4),
        "errors": errors, "warnings": warnings,
    }


def write_report(report: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, default=str) + "\n")
