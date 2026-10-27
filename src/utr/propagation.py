"""Reference radar propagation solver: narrow-angle parabolic equation (PE), split-step sine transform.

This is the stand-in for the "full simulation" in the MVP. It is NOT the Navy's high-fidelity
model. It marches a field u(r, z) in range r over a flat-earth modified-refractivity profile M(z):

    du/dr = (i / 2k) d2u/dz2 + i k 1e-6 (M(z) - M0) u - gamma(z) u

with a perfectly conducting, flat ground at z = 0 (u = 0, horizontal polarisation) and an
absorbing layer at the top of the domain. Range plays the role of "time" for the numerical
time-stepping; the TENG-style solver in teng.py marches the same equation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dst, idst

from . import config
from .parse import Sounding
from .refractivity import (heights_agl, modified_refractivity, refractivity,
                           vapor_pressure_hpa)

C = 299_792_458.0


@dataclass(frozen=True)
class Scenario:
    """Geometry and radar settings. Defaults are placeholders, not Navy radar parameters."""
    freq_hz: float = config.RADAR_FREQ_HZ
    beam_deg: float = config.BEAM_DEG          # full width of the Gaussian starter at half amplitude
    src_height_m: float = config.SRC_HEIGHT_M
    zmax_m: float = config.ZMAX_M              # domain top, including the absorbing layer
    absorb_frac: float = config.ABSORB_FRAC    # top fraction of the domain that absorbs
    dz_m: float = config.DZ_M
    rmax_m: float = config.RMAX_M
    dr_m: float = config.DR_M

    @property
    def k(self) -> float:
        return 2 * np.pi * self.freq_hz / C

    @property
    def sigma_p(self) -> float:
        """Spread of the starter in vertical wavenumber, rad/m."""
        sigma_theta = np.deg2rad(self.beam_deg) / (2 * np.sqrt(2 * np.log(2)))
        return self.k * sigma_theta

    @property
    def nz(self) -> int:
        return int(round(self.zmax_m / self.dz_m))

    @property
    def z(self) -> np.ndarray:
        return np.arange(1, self.nz) * self.dz_m          # interior points; u(0)=u(zmax)=0

    @property
    def ranges(self) -> np.ndarray:
        n = int(round(self.rmax_m / self.dr_m))
        return np.arange(1, n + 1) * self.dr_m


# ---------------------------------------------------------------- atmosphere -> M(z)

def m_profile(s: Sounding, zmax_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Modified refractivity (M-units) versus height AGL (m) for one sounding."""
    e = vapor_pressure_hpa(s.t_c, s.rh_pct, s.dpdp_c)
    good = np.isfinite(s.p_hpa) & np.isfinite(s.t_c) & np.isfinite(e)
    p, t, e = s.p_hpa[good], s.t_c[good], e[good]
    order = np.argsort(-p, kind="stable")
    p, t, e = p[order], t[order], e[order]
    keep = np.concatenate([[True], np.diff(p) < 0])
    p, t, e = p[keep], t[keep], e[keep]
    z = heights_agl(p, t, e)
    m = modified_refractivity(refractivity(p, t, e), z)
    return z, m


def m_on_grid(z_obs: np.ndarray, m_obs: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linear interpolation onto the solver grid; constant gradient above the last level."""
    out = np.interp(grid, z_obs, m_obs)
    above = grid > z_obs[-1]
    if above.any() and len(z_obs) > 1:
        slope = (m_obs[-1] - m_obs[-2]) / max(z_obs[-1] - z_obs[-2], 1.0)
        out[above] = m_obs[-1] + 0.157 * (grid[above] - z_obs[-1]) if slope < 0 else \
            m_obs[-1] + slope * (grid[above] - z_obs[-1])
    return out


def standard_atmosphere_m(grid: np.ndarray, m0: float = 330.0) -> np.ndarray:
    """Standard atmosphere (dN/dz = -39.2 N/km, so dM/dz = +0.118 M/m)."""
    return m0 + (157.0 - 39.2) / 1000.0 * grid


# ---------------------------------------------------------------- fields

def absorber(sc: Scenario) -> np.ndarray:
    """Damping gamma(z) in 1/m, ramping up smoothly (quadratically) over the top of the domain."""
    z = sc.z
    z0 = sc.zmax_m * (1 - sc.absorb_frac)
    x = np.clip((z - z0) / (sc.zmax_m - z0), 0.0, 1.0)
    return config.ABSORB_STRENGTH * x**2


def starter(sc: Scenario) -> np.ndarray:
    """Gaussian antenna aimed horizontally at height src_height_m, plus its ground image (odd)."""
    z = sc.z
    a = 0.5 * sc.sigma_p**2
    return np.exp(-a * (z - sc.src_height_m) ** 2) - np.exp(-a * (z + sc.src_height_m) ** 2)


def free_space_axis_amplitude(sc: Scenario, r: np.ndarray) -> np.ndarray:
    """|u| on the beam axis in free space (analytic Gaussian beam); the propagation-factor reference."""
    a = 0.5 * sc.sigma_p**2
    return np.abs((1 + 2j * a * np.asarray(r) / sc.k) ** -0.5)


def solve_split_step(sc: Scenario, m_grid: np.ndarray, store_every: int = 1):
    """March the PE with the split-step sine-transform method.

    m_grid: modified refractivity on sc.z. Returns (ranges stored, field array [n_r, n_z]).
    """
    z = sc.z
    k = sc.k
    kz = np.pi * np.arange(1, sc.nz) / sc.zmax_m              # DST-I wavenumbers
    prop = np.exp(-1j * kz**2 * sc.dr_m / (2 * k))
    v = k * 1e-6 * (m_grid - m_grid[0])
    half = np.exp((1j * v - absorber(sc)) * sc.dr_m / 2)
    u = starter(sc).astype(complex)
    rs, out = [], []
    for i, r in enumerate(sc.ranges):
        u = half * u
        u = idst(dst(u, type=1) * prop, type=1)
        u = half * u
        if i % store_every == 0:
            rs.append(r)
            out.append(u.copy())
    return np.array(rs), np.array(out)


def propagation_factor_db(sc: Scenario, r: np.ndarray, u: np.ndarray) -> np.ndarray:
    """20 log10 of |u| relative to the free-space beam-axis amplitude, per (range, height)."""
    ref = free_space_axis_amplitude(sc, r)[:, None]
    return 20 * np.log10(np.maximum(np.abs(u), 1e-12) / ref)
