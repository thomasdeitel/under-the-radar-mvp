"""Build supervised (stale sounding -> later sounding) pairs from sounding summaries."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .features import FEATURE_COLUMNS, TARGETS, build_summary_table
from .parse import iter_soundings


def load_summary(station: str, raw_dir=config.RAW_DIR, processed_dir=config.PROCESSED_DIR,
                 rebuild: bool = False) -> pd.DataFrame:
    """Parse the station archive into one row per sounding (cached as CSV)."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache = processed_dir / f"{station}_summary.csv.gz"
    archive = Path(raw_dir) / f"{station}-data.txt.zip"
    if cache.exists() and not rebuild and (not archive.exists() or cache.stat().st_mtime >= archive.stat().st_mtime):
        return pd.read_csv(cache, index_col="time", parse_dates=["time"])
    qc = config.QC_RELAXED if station == config.STRESS_STATION else config.QC_STANDARD
    df = build_summary_table(iter_soundings(archive, config.START_YEAR), qc)
    df.to_csv(cache)
    return df


def make_pairs(summary: pd.DataFrame, lead_hours: int) -> pd.DataFrame:
    """Pair each sounding (target) with the one `lead_hours` earlier (the stale input)."""
    v = summary[summary["valid"].astype(bool)]
    src = v[FEATURE_COLUMNS].copy()
    src.index = src.index + pd.Timedelta(hours=lead_hours)
    src.columns = ["src_" + c for c in src.columns]
    pairs = v[TARGETS].join(src, how="inner")
    pairs["src_lowest_trap_base_m"] = pairs["src_lowest_trap_base_m"].fillna(-1.0)
    doy = pairs.index.dayofyear.values
    pairs["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    pairs["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    pairs["is_12z"] = (pairs.index.hour == 12).astype(float)
    pairs["month"] = pairs.index.month
    pairs["hour"] = pairs.index.hour
    return pairs.dropna()


MODEL_FEATURES = ["src_" + c for c in FEATURE_COLUMNS] + ["doy_sin", "doy_cos", "is_12z"]


def split(pairs: pd.DataFrame, test_start: str = config.TEST_START):
    cut = pd.Timestamp(test_start)
    return pairs[pairs.index < cut], pairs[pairs.index >= cut]
