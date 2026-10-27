"""TENG-style neural-network solver for the same parabolic equation as propagation.py.

TENG (Time-Evolving Natural Gradient, Chen et al. 2024) represents the solution of a PDE with a
neural network u_theta and advances it in "time" by projecting each time-stepping update back
onto the network's parameters with a natural-gradient (Gauss-Newton) least-squares step. Here
"time" is range r, because the narrow-angle PE is a Schroedinger-type equation in r.

Network: u_theta(z) = sum_j c_j sin(w_j z), with complex amplitudes c_j and (optionally trainable)
frequencies w_j. It is a one-hidden-layer network with sine activations and no bias, so the
perfectly conducting ground (u = 0 at z = 0) holds exactly.

At every range step we integrate d(theta)/dr = argmin || J d(theta) - F(u_theta) ||  (the
tangent-space projection, i.e. natural gradient with J the network Jacobian), with RK4 in
parameter space. This is the TDVP flavour of TENG; the optimisation-based flavour is in
`teng_step_optimised`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .propagation import Scenario, absorber, starter


@dataclass
class TengConfig:
    kz_cut: float = 1.7            # highest vertical wavenumber the network carries, rad/m
    train_w: bool = False          # also evolve the sine frequencies (inner weights)
    rcond: float = 1e-8            # truncation of the natural-gradient least squares
    damping: float = 1e-6          # relative Tikhonov damping when train_w is on
    dr_m: float | None = None      # range step; defaults to the scenario's


def init_params(sc: Scenario, cfg: TengConfig):
    """Frequencies on the DST grid up to kz_cut, amplitudes fitted to the starter field."""
    kz = np.pi * np.arange(1, sc.nz) / sc.zmax_m
    w = kz[kz <= cfg.kz_cut].copy()
    z = sc.z
    phi = np.sin(np.outer(z, w))
    c, *_ = np.linalg.lstsq(phi, starter(sc).astype(complex), rcond=cfg.rcond)
    return w, c


def evaluate(w: np.ndarray, c: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.sin(np.outer(z, w)) @ c


def _rhs(sc: Scenario, w, c, z, v_pot, gam):
    """F(u_theta) = (i/2k) u_zz + (i V - gamma) u on the grid, with u_zz analytic."""
    s = np.sin(np.outer(z, w))
    u = s @ c
    uzz = -(s * (w**2)) @ c
    return u, (1j / (2 * sc.k)) * uzz + (1j * v_pot - gam) * u, s


def _jacobian(w, c, z, s, train_w):
    """Real-stacked Jacobian of [Re u; Im u] w.r.t. [Re c, Im c (, w)]."""
    n = len(z)
    cols = [np.vstack([s, np.zeros_like(s)]), np.vstack([np.zeros_like(s), s])]
    if train_w:
        d = (np.cos(np.outer(z, w)) * z[:, None]) * c[None, :]       # d u / d w_j
        cols.append(np.vstack([d.real, d.imag]))
    return np.hstack(cols)


def _theta_dot(sc, cfg, w, c, z, v_pot, gam, pinv_cache):
    u, f, s = _rhs(sc, w, c, z, v_pot, gam)
    rhs = np.concatenate([f.real, f.imag])
    p = len(w)
    if not cfg.train_w:
        if pinv_cache.get("p") is None:
            J = _jacobian(w, c, z, s, False)
            pinv_cache["p"] = np.linalg.pinv(J, rcond=cfg.rcond)
        d = pinv_cache["p"] @ rhs
        return np.concatenate([d[:p], d[p:2 * p]]), None
    J = _jacobian(w, c, z, s, True)
    # Tikhonov-damped (Levenberg-Marquardt) natural-gradient solve for the nonlinear inner weights
    gram = J.T @ J
    lam = cfg.damping * np.trace(gram) / gram.shape[0]
    d = np.linalg.solve(gram + lam * np.eye(gram.shape[0]), J.T @ rhs)
    resid = float(np.linalg.norm(J @ d - rhs) / max(np.linalg.norm(rhs), 1e-30))
    return d, resid


def solve_teng(sc: Scenario, m_grid: np.ndarray, cfg: TengConfig | None = None,
               store_every: int = 1, params0=None):
    """March the network through range. Returns dict with ranges, fields, params and diagnostics."""
    cfg = cfg or TengConfig()
    z = sc.z
    dr = cfg.dr_m or sc.dr_m
    v_pot = sc.k * 1e-6 * (m_grid - m_grid[0])
    gam = absorber(sc)
    w, c = params0 if params0 is not None else init_params(sc, cfg)
    w, c = w.copy(), c.copy()
    p = len(w)
    cache: dict = {}
    n_steps = int(round(sc.rmax_m / dr))
    rs, fields, proj_resid = [], [], []
    t0 = time.perf_counter()

    def dtheta(w_, c_):
        d, resid = _theta_dot(sc, cfg, w_, c_, z, v_pot, gam, cache)
        if cfg.train_w:
            return d[:p] + 1j * d[p:2 * p], d[2 * p:], resid
        return d[:p] + 1j * d[p:2 * p], np.zeros(p), resid

    for i in range(1, n_steps + 1):
        k1c, k1w, r1 = dtheta(w, c)
        k2c, k2w, _ = dtheta(w + dr / 2 * k1w, c + dr / 2 * k1c)
        k3c, k3w, _ = dtheta(w + dr / 2 * k2w, c + dr / 2 * k2c)
        k4c, k4w, _ = dtheta(w + dr * k3w, c + dr * k3c)
        c = c + dr / 6 * (k1c + 2 * k2c + 2 * k3c + k4c)
        if cfg.train_w:
            w = w + dr / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
        if i % max(1, round(sc.dr_m / dr) * store_every) == 0:
            rs.append(i * dr)
            fields.append(evaluate(w, c, z))
            proj_resid.append(r1 if r1 is not None else np.nan)
    return {"ranges": np.array(rs), "fields": np.array(fields), "w": w, "c": c,
            "proj_resid": np.array(proj_resid), "seconds": time.perf_counter() - t0,
            "n_params": 2 * p + (p if cfg.train_w else 0)}
