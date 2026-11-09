from __future__ import annotations

import numpy as np

from analysis.p0_msd import (
    _bootstrap_paired_diffusion_difference,
    _phase13_groups_from_manifest,
    fit_effective_diffusion,
    relative_trace_on_grid,
)
from analysis.pav import PAVTrace


def _trace(times, axis):
    times = np.asarray(times, float)
    return PAVTrace(times_ms=times, axis_deg=np.asarray(axis, float), length=np.ones(times.size))


def test_relative_trace_unwraps_axial_boundary_and_zeros_origin():
    times = np.arange(0.0, 30.0, 5.0)
    ref = _trace(times, np.zeros_like(times))
    # On an axial circle, 80 -> 89 -> -82 -> -73 is continuous positive travel.
    noisy = _trace(times, np.array([0.0, 40.0, 80.0, 89.0, -82.0, -73.0]))
    out = relative_trace_on_grid(noisy, ref, start_ms=5.0, end_ms=25.0, step_ms=5.0)
    assert np.allclose(out.relative_from_origin_deg, [0.0, 40.0, 49.0, 58.0, 67.0], atol=1e-9)
    assert np.allclose(out.increment_sq_deg2, out.relative_from_origin_deg ** 2)


def test_fit_effective_diffusion_recovers_known_linear_msd():
    elapsed = np.arange(0.0, 2001.0, 50.0)
    d_true = 3.25
    msd = 7.0 + 2.0 * d_true * (elapsed / 1000.0)
    fit = fit_effective_diffusion(elapsed, msd, (400.0, 1800.0))
    assert abs(fit.d_deg2_per_s - d_true) < 1e-10
    assert abs(fit.r2 - 1.0) < 1e-12


def test_fit_effective_diffusion_preserves_negative_local_slope():
    elapsed = np.arange(0.0, 2001.0, 50.0)
    msd = 20.0 - 4.0 * (elapsed / 1000.0)
    fit = fit_effective_diffusion(elapsed, msd, (400.0, 1800.0))
    assert fit.d_deg2_per_s < 0.0


def test_paired_diffusion_bootstrap_uses_seed_pairing():
    elapsed = np.arange(0.0, 2001.0, 50.0)
    t = elapsed / 1000.0
    # Four seed curves with static D exactly 2 deg^2/s above temporal D.
    temporal = np.asarray([(2.0 * (1.0 + 0.1 * i) * t) for i in range(4)])
    static = temporal + 4.0 * t  # slope +4 => D +2
    est, lo, hi = _bootstrap_paired_diffusion_difference(
        static, temporal, elapsed, (400.0, 1800.0), repeats=500, seed=7
    )
    assert abs(est - 2.0) < 1e-10
    assert lo > 1.99 and hi < 2.01


def test_phase13_scope_respects_primary_anchor():
    manifest = {
        "tau_facilitation_ms": [250.0, 1000.0],
        "tau_recovery_ms": [5.0, 20.0],
        "temporal_modes": ["dynamic", "depletion_only"],
        "primary_confirmatory_anchor": {
            "tau_facilitation_ms": 1000.0,
            "tau_recovery_ms": 20.0,
            "mode": "dynamic",
        },
    }
    assert _phase13_groups_from_manifest(manifest, scope="anchor") == [(1000.0, 20.0, "dynamic")]
    assert len(_phase13_groups_from_manifest(manifest, scope="all")) == 8
