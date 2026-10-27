"""Radar coverage map from one atmospheric reading, plus how stale a map has become.

A "coverage map" here is the propagation factor (dB relative to free space, along the beam axis
amplitude) over range and height for one radial, from the TENG-style network solver in teng.py.
`FAST` is the setting meant for ordinary computers; the reference split-step solver in
propagation.py is the stand-in for "the full simulation" used to check it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parse import Sounding
from .propagation import (Scenario, m_on_grid, m_profile, propagation_factor_db,
                          solve_split_step)
from .teng import TengConfig, solve_teng

FAST = TengConfig(kz_cut=1.0, dr_m=100.0)       # about 6 s for a 60 km radial on 2 CPU cores
FINE = TengConfig(kz_cut=1.4, dr_m=50.0)        # about 16 s; used as a self-check
COVERAGE_FLOOR_DB = -30.0                        # placeholder threshold for "covered", not a radar spec
MAP_HEIGHT_RANGE_M = (5.0, 300.0)
MAP_MIN_RANGE_M = 5000.0


@dataclass
class CoverageMap:
    ranges_m: np.ndarray
    heights_m: np.ndarray
    pf_db: np.ndarray            # [n_range, n_height]
    seconds: float
    solver: str

    def covered(self, floor_db: float = COVERAGE_FLOOR_DB) -> np.ndarray:
        return self.pf_db >= floor_db

    def farthest_covered_km(self, height_m: float, floor_db: float = COVERAGE_FLOOR_DB) -> float:
        """Farthest range (beyond 5 km) where the given height is covered; 0 if it never is."""
        j = int(np.argmin(np.abs(self.heights_m - height_m)))
        ok = (self.pf_db[:, j] >= floor_db) & (self.ranges_m >= MAP_MIN_RANGE_M)
        return float(self.ranges_m[np.flatnonzero(ok)[-1]] / 1000) if ok.any() else 0.0

    def covered_fraction(self, height_m: float, floor_db: float = COVERAGE_FLOOR_DB) -> float:
        """Share of the 5-60 km range cells at this height that are covered (gaps are blind spots)."""
        j = int(np.argmin(np.abs(self.heights_m - height_m)))
        sel = self.ranges_m >= MAP_MIN_RANGE_M
        return float((self.pf_db[sel, j] >= floor_db).mean())


def usable_profile(s: Sounding, sc: Scenario):
    """(height, M) arrays if the sounding is good enough to drive the solver, else None."""
    zo, mo = m_profile(s, sc.zmax_m)
    if len(zo) < 10 or zo[-1] < 650:
        return None
    return zo, mo


def build_map(profile, sc: Scenario | None = None, solver: str = "teng", cfg: TengConfig = FAST) -> CoverageMap:
    """Coverage map for one (height, M) profile with the TENG-style network solver or the reference."""
    import time
    sc = sc or Scenario()
    zo, mo = profile
    mg = m_on_grid(zo, mo, sc.z)
    t0 = time.perf_counter()
    if solver == "teng":
        out = solve_teng(sc, mg, cfg)
        r, u = out["ranges"], out["fields"]
    else:
        r, u = solve_split_step(sc, mg)
    secs = time.perf_counter() - t0
    pf = propagation_factor_db(sc, r, u)
    return CoverageMap(r, sc.z, pf, secs, solver)


def map_difference_db(a: np.ndarray, b: np.ndarray, ranges_m: np.ndarray, heights_m: np.ndarray) -> float:
    """Mean absolute dB difference between two maps over the cells that matter (either map above -40 dB,
    beyond 5 km, between 5 and 300 m high)."""
    hm = (heights_m >= MAP_HEIGHT_RANGE_M[0]) & (heights_m <= MAP_HEIGHT_RANGE_M[1])
    rm = ranges_m >= MAP_MIN_RANGE_M
    A, B = a[np.ix_(rm, hm)], b[np.ix_(rm, hm)]
    mask = (A > -40) | (B > -40)
    return float(np.abs(A - B)[mask].mean()) if mask.any() else 0.0
