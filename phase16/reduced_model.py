"""Reduced-model quantities for Phase 16.

Two-variable model (bump position phi, efficacy-well position psi; axial degrees):
    dphi/dt = -a (phi - psi) + xi(t),     dpsi/dt = b (phi - psi)
Relative coordinate delta = phi - psi relaxes at lambda = a + b; the joint coordinate
c = (b phi + a psi)/lambda feels no deterministic force (neutral mode of the ring symmetry).
Deterministic restoration after an activity-only shift: F(T) = a/(a+b) * (1 - exp(-(a+b) T)).
With noise on phi only (intensity 2D): MSD_phi(t) = 2D[(b/lam)^2 t + a(2b+a)/lam^3 (1 - exp(-lam t))],
so the long-time diffusion is reduced by (1 - F_inf)^2 relative to the short-time value.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def dax(a, b):
    """Shortest axial difference a - b in [-90, 90)."""
    return (np.asarray(a) - np.asarray(b) + 90.0) % 180.0 - 90.0


def axial_center(weights, theta_deg):
    """Doubled-angle centre (deg, [0,180)) and normalised resultant length of a non-negative profile."""
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    z = np.sum(w * np.exp(2j * np.deg2rad(theta_deg)), axis=-1)
    m = np.sum(w, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.rad2deg(np.angle(z)) / 2.0) % 180.0, np.abs(z) / m


def pav_short(times_ms, angles_deg, grid_ms, window_ms=150.0):
    """Trailing-window doubled-angle population vector; 150 ms spans about one theta cycle."""
    t = np.asarray(times_ms); z = np.exp(2j * np.deg2rad(np.asarray(angles_deg)))
    out = np.full(len(grid_ms), np.nan)
    for k, g in enumerate(grid_ms):
        m = (t > g - window_ms) & (t <= g)
        if m.any():
            out[k] = (np.rad2deg(np.angle(z[m].sum())) / 2.0) % 180.0
    return out


def efficacy(u, R, route, U, k=1.0):
    """E->I efficacy multiplier read by the circuit in each model variant."""
    if route in ("dynamic", "u_output_only"):
        return k * u * R
    if route in ("u_release_only", "mean_u_release", "depletion_only"):
        return k * U * R
    if route == "facilitation_only":
        return u
    raise ValueError(route)


def drive_offset(s, p, theta_deg):
    """Offset (deg) of the E->I drive profile s*p from the activity profile s.
    Zero when efficacy is uniform under the bump; its sign gives the direction of the efficacy gradient."""
    c_s, _ = axial_center(s, theta_deg)
    c_d, _ = axial_center(s * p, theta_deg)
    return dax(c_d, c_s), c_s


def restoration_curve(a, b, T):
    lam = a + b
    return a / lam * (1.0 - np.exp(-lam * np.asarray(T)))


def msd_two_variable(t, D, a, b):
    lam = a + b; t = np.asarray(t, dtype=float)
    return 2.0 * D * ((b / lam) ** 2 * t + a * (2.0 * b + a) / lam ** 3 * (1.0 - np.exp(-lam * t)))


def fit_restoration(t_s, F, a0=3.0, b0=8.0):
    """Fit F(T) = a/(a+b)(1-exp(-(a+b)T)); returns a, b (1/s) and RMS residual."""
    t_s, F = np.asarray(t_s), np.asarray(F); ok = np.isfinite(F)
    r = least_squares(lambda p: restoration_curve(p[0], p[1], t_s[ok]) - F[ok], [a0, b0], bounds=([1e-6, 1e-6], [200, 200]))
    return r.x[0], r.x[1], float(np.sqrt(np.mean(r.fun ** 2)))


def fit_clamp_rate(t_s, F):
    """Under a u-clamp the well cannot follow (b = 0): F(T) = 1 - exp(-a T)."""
    t_s, F = np.asarray(t_s), np.asarray(F); ok = np.isfinite(F) & (F < 0.999)
    r = least_squares(lambda p: 1.0 - np.exp(-p[0] * t_s[ok]) - F[ok], [2.0], bounds=([0.0], [200.0]))
    return r.x[0], float(np.sqrt(np.mean(r.fun ** 2)))


def fit_msd(t_s, msd, joint=True):
    """Fit MSD with the two-variable form (joint=True) or pure diffusion 2Dt + c (joint=False)."""
    t_s, msd = np.asarray(t_s), np.asarray(msd); ok = np.isfinite(msd)
    if not joint:
        A = np.vstack([2 * t_s[ok], np.ones(ok.sum())]).T
        (D, c0), *_ = np.linalg.lstsq(A, msd[ok], rcond=None)
        return dict(D=D, offset=c0)
    r = least_squares(lambda p: msd_two_variable(t_s[ok], *p) - msd[ok], [1.0, 3.0, 8.0], bounds=([0, 0, 0], [1e4, 200, 200]))
    D, a, b = r.x
    return dict(D=D, a=a, b=b, F_inf=a / (a + b), long_short_ratio=(b / (a + b)) ** 2)


def bootstrap_mean_ci(x, n=10000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n, x.size), replace=True).mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
