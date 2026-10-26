"""Turn one sounding into QC flags, refractivity-based duct metrics and model features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .parse import Sounding
from .refractivity import (duct_metrics, heights_agl, modified_refractivity,
                           refractivity, sat_vapor_pressure_hpa, vapor_pressure_hpa)


def _interp_at(z, values, target):
    if z[-1] < target:
        return np.nan
    return float(np.interp(target, z, values))


def summarize(s: Sounding, qc: dict | None = None) -> dict:
    """One row per sounding: QC result plus ducting metrics and low-level features."""
    qc = config.QC_STANDARD if qc is None else qc
    row = {"station": s.station, "time": s.time, "n_levels_reported": s.n_levels_reported}
    issues: list[str] = []

    e = vapor_pressure_hpa(s.t_c, s.rh_pct, s.dpdp_c)
    good = np.isfinite(s.p_hpa) & np.isfinite(s.t_c) & np.isfinite(e)
    p, t, e = s.p_hpa[good], s.t_c[good], e[good]
    if len(p) < 3:
        row.update(valid=False, qc_issues="too few usable levels", n_levels=int(len(p)))
        return row

    # order surface -> up and drop non-decreasing pressure (duplicates / bad order)
    order = np.argsort(-p, kind="stable")
    p, t, e = p[order], t[order], e[order]
    keep = np.concatenate([[True], np.diff(p) < 0])
    p, t, e = p[keep], t[keep], e[keep]

    z = heights_agl(p, t, e)
    n = refractivity(p, t, e)
    m = modified_refractivity(n, z)
    # keep levels up to 3 km plus the first level above it, so we can tell whether the
    # sounding actually reached 3 km
    cap = config.MAX_DUCT_HEIGHT_M
    ext = z <= cap
    above = np.flatnonzero(~ext)
    if len(above):
        ext[above[0]] = True
    z, p, t, e, m = z[ext], p[ext], t[ext], e[ext], m[ext]
    rh = 100.0 * e / sat_vapor_pressure_hpa(t)   # consistent RH even where it was not reported

    top = float(z[-1])
    gaps = np.diff(z)
    max_gap = float(gaps.max()) if len(gaps) else np.nan
    n_low = int((z <= cap).sum())
    if n_low < qc["min_levels"]:
        issues.append(f"only {n_low} levels below 3 km")
    if top < cap:
        issues.append(f"profile tops out at {top:.0f} m")
    if max_gap > qc["max_gap_m"]:
        issues.append(f"vertical gap of {max_gap:.0f} m")
    if np.nanmax(np.abs(t)) > 60 or np.any(m < 100):
        issues.append("implausible values")

    row.update(n_levels=n_low, top_height_m=top, max_gap_m=max_gap,
               valid=not issues, qc_issues="; ".join(issues))
    if not row["valid"]:
        return row

    inside = z <= cap
    row.update(duct_metrics(z[inside], m[inside]))
    row["sfc_t_c"] = float(t[0])
    row["sfc_p_hpa"] = float(p[0])
    row["sfc_rh"] = float(rh[0])
    row["rh_1km"] = _interp_at(z, rh, 1000.0)
    t_1km = _interp_at(z, t, 1000.0)
    row["t_lapse_0_1km"] = (t_1km - t[0]) / 1.0 if np.isfinite(t_1km) else np.nan   # degC per km
    row["m_diff_0_1km"] = _interp_at(z, m, 1000.0) - float(m[0])
    z500 = _interp_at(z, m, 500.0)
    row["dmdz_0_500"] = (z500 - float(m[0])) / 500.0 if np.isfinite(z500) else np.nan
    row["max_warming_above_sfc_c"] = float(max(0.0, np.max(t - t[0])))
    return row


FEATURE_COLUMNS = [
    "sfc_t_c", "sfc_p_hpa", "sfc_rh", "rh_1km", "t_lapse_0_1km", "m_diff_0_1km",
    "dmdz_0_500", "max_warming_above_sfc_c", "duct_present", "trap_thickness_m",
    "trap_strength_m", "lowest_trap_base_m", "surface_duct",
]
TARGETS = ["duct_present", "trap_strength_m", "trap_thickness_m"]


def build_summary_table(soundings, qc: dict | None = None) -> pd.DataFrame:
    rows = [summarize(s, qc) for s in soundings]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["time", "n_levels_reported"]).drop_duplicates("time", keep="last")
    return df.set_index("time").sort_index()
