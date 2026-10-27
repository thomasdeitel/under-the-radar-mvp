"""Estimate the duct state for the next 00Z/12Z sounding from everything known "now".

Inputs available at any hour (the model is meant to be re-run hourly):
  * the last sounding's full set of conditions and duct state, and the one before it (past ducts)
  * hourly airport weather: temperature, dew point, pressure, wind
  * hourly pier observations: sea-surface temperature, air temperature, wind, pressure
  * trends of those over the last 3-24 h and the sea-air humidity gap that drives evaporation ducts
  * season and time of day

Labels only exist where there is a sounding (00Z and 12Z), so the model is trained and scored
on "L hours before a sounding, what was the duct state at that sounding?" for L = 0, 3, 6, 9.
L = 0 is the now-cast; larger L is the same question asked earlier. Every number in the report
comes from a time period the model never trained on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .features import FEATURE_COLUMNS, TARGETS
from .refractivity import sat_vapor_pressure_hpa

LEADS = (0, 3, 6, 9)
HISTORY_COLS = ["duct_present", "trap_strength_m", "trap_thickness_m", "lowest_trap_base_m"]

SURFACE_FEATURES = [
    "t_c", "td_c", "rh", "dewpoint_depr_c", "pres_hpa", "wspd_ms", "wind_u", "wind_v",
    "sst_c", "air_minus_sea_c", "sea_air_vp_gap_hpa", "pier_wspd_ms",
    "d_t_3h", "d_t_6h", "d_pres_3h", "d_sst_6h", "d_td_6h", "wspd_mean_6h", "t_range_24h",
]
SOUNDING_FEATURES = ["src_" + c for c in FEATURE_COLUMNS]
HISTORY_FEATURES = ["prev2_" + c for c in HISTORY_COLS]
TIME_FEATURES = ["doy_sin", "doy_cos", "is_12z"]

FEATURE_SETS = {
    "sounding only": SOUNDING_FEATURES + TIME_FEATURES,
    "+ earlier ducts": SOUNDING_FEATURES + HISTORY_FEATURES + TIME_FEATURES,
    "+ hourly weather and sea": SOUNDING_FEATURES + HISTORY_FEATURES + TIME_FEATURES + SURFACE_FEATURES + ["hours_since_sounding"],
}


def surface_feature_table(hourly: pd.DataFrame) -> pd.DataFrame:
    """Per-hour features computed from the hourly observation table (index = UTC hour)."""
    h = hourly.copy()
    full = pd.date_range(h.index.min(), h.index.max(), freq="h")
    h = h.reindex(full).ffill(limit=3)                       # bridge short gaps only
    f = pd.DataFrame(index=h.index)
    f["t_c"] = h["t_c"].fillna(h["pier_air_c"])
    f["td_c"] = h["td_c"]
    f["rh"] = 100 * sat_vapor_pressure_hpa(f["td_c"]) / sat_vapor_pressure_hpa(f["t_c"])
    f["dewpoint_depr_c"] = f["t_c"] - f["td_c"]
    f["pres_hpa"] = h["slp_hpa"].fillna(h["pier_pres_hpa"])
    f["wspd_ms"] = h["wspd_ms"].fillna(h["pier_wspd_ms"])
    ang = np.deg2rad(h["wdir_deg"])
    calm = h["wspd_ms"].fillna(0) < 0.5
    f["wind_u"] = np.where(calm, 0.0, -f["wspd_ms"] * np.sin(ang))
    f["wind_v"] = np.where(calm, 0.0, -f["wspd_ms"] * np.cos(ang))
    f.loc[h["wdir_deg"].isna() & ~calm, ["wind_u", "wind_v"]] = np.nan
    f["sst_c"] = h["sst_c"]
    f["air_minus_sea_c"] = h["pier_air_c"].fillna(f["t_c"]) - h["sst_c"]
    e_air = sat_vapor_pressure_hpa(f["td_c"])
    f["sea_air_vp_gap_hpa"] = 0.98 * sat_vapor_pressure_hpa(h["sst_c"]) - e_air   # saltwater factor 0.98
    f["pier_wspd_ms"] = h["pier_wspd_ms"]
    f["d_t_3h"] = f["t_c"] - f["t_c"].shift(3)
    f["d_t_6h"] = f["t_c"] - f["t_c"].shift(6)
    f["d_pres_3h"] = f["pres_hpa"] - f["pres_hpa"].shift(3)
    f["d_sst_6h"] = f["sst_c"] - f["sst_c"].shift(6)
    f["d_td_6h"] = f["td_c"] - f["td_c"].shift(6)
    f["wspd_mean_6h"] = f["wspd_ms"].rolling(6, min_periods=3).mean()
    f["t_range_24h"] = f["t_c"].rolling(24, min_periods=12).max() - f["t_c"].rolling(24, min_periods=12).min()
    return f[SURFACE_FEATURES]


def _sounding_block(summary: pd.DataFrame, times: pd.DatetimeIndex, hours_back: int, cols: list[str], prefix: str):
    """Columns of the valid sounding `hours_back` hours before each time (NaN if there was none)."""
    v = summary[summary["valid"].astype(bool)][cols].copy()
    v.index = v.index + pd.Timedelta(hours=hours_back)
    v.columns = [prefix + c for c in cols]
    return v.reindex(times)


def build_samples(summary: pd.DataFrame, surface_feats: pd.DataFrame, leads=LEADS) -> pd.DataFrame:
    """One row per (target sounding, lead): everything known `lead` hours before the sounding."""
    valid = summary[summary["valid"].astype(bool)]
    times = valid.index
    base = valid[TARGETS].copy()
    src = _sounding_block(summary, times, 12, FEATURE_COLUMNS, "src_")
    prev2 = _sounding_block(summary, times, 24, HISTORY_COLS, "prev2_")
    frames = []
    for lead in leads:
        df = pd.concat([base, src, prev2], axis=1)
        for c in ("src_lowest_trap_base_m", "prev2_lowest_trap_base_m"):
            df[c] = df[c].where(df[c].notna() | df[c.replace("lowest_trap_base_m", "duct_present")].isna(), -1.0)
        now = times - pd.Timedelta(hours=lead)
        sf = surface_feats.reindex(now)
        sf.index = times
        df = pd.concat([df, sf], axis=1)
        df["hours_since_sounding"] = 12 - lead
        df["lead"] = lead
        doy = times.dayofyear.values
        df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
        df["is_12z"] = (times.hour == 12).astype(float)
        df["month"], df["hour"] = times.month, times.hour
        frames.append(df)
    out = pd.concat(frames).sort_index(kind="stable")
    # keep rows with both earlier soundings and a surface record at "now"
    need = SOUNDING_FEATURES + HISTORY_FEATURES + TARGETS
    out = out.dropna(subset=need)
    out = out[out[["t_c", "td_c", "sst_c", "pres_hpa"]].notna().all(axis=1)]
    return out[out.index >= pd.Timestamp(f"{config.SURFACE_START_YEAR + 1}-01-01")]


def next_slot(now: pd.Timestamp) -> pd.Timestamp:
    """The next 00Z or 12Z at or after `now`."""
    day = now.normalize()
    for h in (0, 12, 24):
        t = day + pd.Timedelta(hours=h)
        if t >= now:
            return t


def runtime_row(summary: pd.DataFrame, surface_feats: pd.DataFrame, now: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """One row of model inputs for "right now": latest sounding at or before `now`, hourly weather at `now`."""
    valid = summary[summary["valid"].astype(bool)]
    past = valid[valid.index <= now]
    if past.empty:
        raise ValueError("no usable sounding at or before the requested time")
    s_time = past.index[-1]
    target = next_slot(now)
    row = {f"src_{c}": past.iloc[-1][c] for c in FEATURE_COLUMNS}
    prev2 = valid.loc[valid.index == s_time - pd.Timedelta(hours=12)]
    for c in HISTORY_COLS:
        row[f"prev2_{c}"] = prev2.iloc[0][c] if len(prev2) else np.nan
    for c in ("src_lowest_trap_base_m", "prev2_lowest_trap_base_m"):
        if pd.isna(row[c]) and not pd.isna(row[c.replace("lowest_trap_base_m", "duct_present")]):
            row[c] = -1.0
    hour = now.floor("h")
    if hour not in surface_feats.index:
        raise ValueError(f"no hourly weather record for {hour}")
    row.update(surface_feats.loc[hour].to_dict())
    age = (now - s_time) / pd.Timedelta(hours=1)
    row["hours_since_sounding"] = age
    doy = target.dayofyear
    row.update(doy_sin=np.sin(2 * np.pi * doy / 365.25), doy_cos=np.cos(2 * np.pi * doy / 365.25),
               is_12z=float(target.hour == 12), month=target.month, hour=target.hour, lead=0)
    info = {"now": now, "target_sounding": target, "last_sounding": s_time, "hours_since_sounding": float(age)}
    return pd.DataFrame([row]), info
