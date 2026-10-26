"""Atmospheric refractivity and ducting from a radiosonde profile.

Radar rays bend toward higher refractivity. When modified refractivity M falls with height
(dM/dz < 0) a layer traps the beam, which can extend detection range far past the horizon.
This is a physical proxy for "radar performance", not a propagation simulation.
"""
from __future__ import annotations

import numpy as np

from . import config

RD = 287.05
G = 9.80665


def sat_vapor_pressure_hpa(t_c):
    """Buck (1981) saturation vapour pressure over water, hPa."""
    t_c = np.asarray(t_c, dtype=float)
    return 6.1121 * np.exp((18.678 - t_c / 234.5) * (t_c / (257.14 + t_c)))


def vapor_pressure_hpa(t_c, rh_pct=None, dpdp_c=None):
    """Vapour pressure from RH, falling back to dew-point depression where RH is missing."""
    t_c = np.asarray(t_c, dtype=float)
    e = np.full_like(t_c, np.nan)
    if rh_pct is not None:
        rh = np.asarray(rh_pct, dtype=float)
        e = rh / 100.0 * sat_vapor_pressure_hpa(t_c)
    if dpdp_c is not None:
        dp = np.asarray(dpdp_c, dtype=float)
        e_dp = sat_vapor_pressure_hpa(t_c - dp)
        e = np.where(np.isnan(e), e_dp, e)
    return e


def refractivity(p_hpa, t_c, e_hpa):
    """Radio refractivity N (ITU-R P.453): N = 77.6 P/T + 3.73e5 e/T^2."""
    tk = np.asarray(t_c, dtype=float) + 273.15
    return 77.6 * np.asarray(p_hpa, dtype=float) / tk + 3.73e5 * np.asarray(e_hpa, dtype=float) / tk**2


def heights_agl(p_hpa, t_c, e_hpa):
    """Height above the lowest level (m), by hypsometric integration with virtual temperature.

    Levels must be ordered from the surface upward (decreasing pressure).
    """
    p = np.asarray(p_hpa, dtype=float)
    tk = np.asarray(t_c, dtype=float) + 273.15
    e = np.asarray(e_hpa, dtype=float)
    tv = tk / (1.0 - 0.378 * e / p)
    z = np.zeros_like(p)
    for k in range(1, len(p)):
        tv_mean = 0.5 * (tv[k] + tv[k - 1])
        z[k] = z[k - 1] + (RD * tv_mean / G) * np.log(p[k - 1] / p[k])
    return z


def modified_refractivity(n, z_m):
    """M = N + 0.157 z (z in metres). Only the gradient matters, so AGL height is fine."""
    return np.asarray(n, dtype=float) + 0.157 * np.asarray(z_m, dtype=float)


def duct_metrics(z_m, m, min_strength=None):
    """Summarise trapping layers (dM/dz < 0) in an M profile.

    Returns a dict with duct_present, trap_thickness_m, trap_strength_m (total M drop),
    lowest_trap_base_m (NaN if none) and surface_duct.
    """
    min_strength = config.DUCT_MIN_STRENGTH_M if min_strength is None else min_strength
    z = np.asarray(z_m, dtype=float)
    m = np.asarray(m, dtype=float)
    dz = np.diff(z)
    dm = np.diff(m)
    ok = dz > 1.0
    trapping = ok & (dm < 0)
    thickness = float(dz[trapping].sum())
    strength = float(-dm[trapping].sum())
    present = int(strength >= min_strength)
    base = np.nan
    if present:
        idx = np.flatnonzero(trapping)
        base = float(z[idx[0]])
    return {
        "duct_present": present,
        "trap_thickness_m": thickness if present else 0.0,
        "trap_strength_m": strength if present else 0.0,
        "lowest_trap_base_m": base,
        "surface_duct": int(present and base <= config.SURFACE_DUCT_BASE_M),
    }
