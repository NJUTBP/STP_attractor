from __future__ import annotations

import numpy as np

from config import load_config
from model import (
    aligned_roll_bins,
    apply_activity_shift,
    apply_co_shift,
    apply_trace_shift,
    run_segment,
    simulate,
)


def _cfg():
    return load_config("config/smoke.toml")


def _wrap_deg(angles: np.ndarray, delta: float) -> np.ndarray:
    return np.mod(np.asarray(angles, float) + delta, 360.0)


def _pairs(times: np.ndarray, angles: np.ndarray, t0: float, delta: float = 0.0) -> np.ndarray:
    mask = times >= t0
    out = np.column_stack([times[mask], _wrap_deg(angles[mask], delta)])
    return out[np.lexsort((out[:, 1], out[:, 0]))]


def test_checkpoint_resume_matches_uninterrupted_dynamic():
    cfg = _cfg()
    full = simulate(cfg, condition="dynamic", seed=3, noise_pa=15.0)
    first, state = run_segment(cfg, condition="dynamic", seed=3, noise_pa=15.0, end_ms=80.0)
    resumed, _ = run_segment(
        cfg,
        condition="dynamic",
        seed=3,
        noise_pa=15.0,
        start_state=state,
        end_ms=cfg.simulation.duration_ms,
    )
    assert first.spike_times_e_ms.size <= full.spike_times_e_ms.size
    assert np.array_equal(resumed.spike_times_e_ms, full.spike_times_e_ms)
    assert np.array_equal(resumed.spike_angles_e_deg, full.spike_angles_e_deg)
    assert np.array_equal(resumed.spike_times_i_ms, full.spike_times_i_ms)
    assert np.array_equal(resumed.plfp_gaba_e, full.plfp_gaba_e)
    assert np.array_equal(resumed.mean_efficacy_by_e, full.mean_efficacy_by_e)


def test_checkpoint_resume_matches_uninterrupted_static():
    cfg = _cfg()
    full = simulate(cfg, condition="static_gain", seed=2, noise_pa=20.0, static_gain=0.55)
    _, state = run_segment(
        cfg, condition="static_gain", seed=2, noise_pa=20.0, static_gain=0.55, end_ms=80.0
    )
    resumed, _ = run_segment(
        cfg,
        condition="static_gain",
        seed=2,
        noise_pa=20.0,
        static_gain=0.55,
        start_state=state,
        end_ms=cfg.simulation.duration_ms,
    )
    assert np.array_equal(resumed.spike_times_e_ms, full.spike_times_e_ms)
    assert np.array_equal(resumed.plfp_gaba_e, full.plfp_gaba_e)
    assert state.u is None and state.resources is None


def test_co_shift_is_spatial_equivariance_after_checkpoint():
    cfg = _cfg()
    delta = 22.5
    e_bins, i_bins = aligned_roll_bins(delta, cfg.network.ne, cfg.network.ni)
    assert (e_bins, i_bins) == (4, 1)
    _, state = run_segment(cfg, condition="dynamic", seed=5, noise_pa=0.0, end_ms=80.0)
    unshifted, _ = run_segment(
        cfg, condition="dynamic", seed=5, noise_pa=0.0, start_state=state, end_ms=cfg.simulation.duration_ms
    )
    shifted, _ = run_segment(
        cfg,
        condition="dynamic",
        seed=5,
        noise_pa=0.0,
        start_state=apply_co_shift(state, e_bins, i_bins),
        end_ms=cfg.simulation.duration_ms,
    )
    t0 = 80.0
    assert np.allclose(
        _pairs(unshifted.spike_times_e_ms, unshifted.spike_angles_e_deg, t0, delta),
        _pairs(shifted.spike_times_e_ms, shifted.spike_angles_e_deg, t0),
        atol=1e-12,
    )
    assert np.allclose(
        _pairs(unshifted.spike_times_i_ms, unshifted.spike_angles_i_deg, t0, delta),
        _pairs(shifted.spike_times_i_ms, shifted.spike_angles_i_deg, t0),
        atol=1e-12,
    )
    n0 = int(unshifted.sample_times_ms.searchsorted(t0, side="left"))
    assert np.allclose(unshifted.plfp_gaba_e[n0:], shifted.plfp_gaba_e[n0:])


def test_activity_shift_moves_fast_state_not_stp():
    cfg = _cfg()
    e_bins, i_bins = aligned_roll_bins(22.5, cfg.network.ne, cfg.network.ni)
    _, state = run_segment(cfg, condition="dynamic", seed=1, noise_pa=0.0, end_ms=80.0)
    moved = apply_activity_shift(state, e_bins, i_bins)
    assert np.array_equal(moved.ve, np.roll(state.ve, e_bins))
    assert np.array_equal(moved.s_nmda, np.roll(state.s_nmda, e_bins))
    assert np.array_equal(moved.s_gaba, np.roll(state.s_gaba, i_bins))
    assert np.array_equal(moved.u, state.u)
    assert np.array_equal(moved.resources, state.resources)


def test_trace_shift_moves_stp_not_fast_state():
    cfg = _cfg()
    e_bins, i_bins = aligned_roll_bins(22.5, cfg.network.ne, cfg.network.ni)
    _, state = run_segment(cfg, condition="dynamic", seed=1, noise_pa=0.0, end_ms=80.0)
    moved = apply_trace_shift(state, e_bins, i_bins)
    assert np.array_equal(moved.u, np.roll(state.u, e_bins))
    assert np.array_equal(moved.resources, np.roll(state.resources, e_bins))
    assert np.array_equal(moved.ve, state.ve)
    assert np.array_equal(moved.s_nmda, state.s_nmda)
    assert np.array_equal(moved.vi, state.vi)


def test_static_activity_shift_equals_co_shift():
    cfg = _cfg()
    e_bins, i_bins = aligned_roll_bins(22.5, cfg.network.ne, cfg.network.ni)
    _, state = run_segment(
        cfg, condition="static_gain", seed=0, noise_pa=0.0, static_gain=0.4, end_ms=80.0
    )
    a = apply_activity_shift(state, e_bins, i_bins)
    c = apply_co_shift(state, e_bins, i_bins)
    assert np.array_equal(a.ve, c.ve)
    assert np.array_equal(a.s_gaba, c.s_gaba)
    assert a.u is None and c.u is None
    ra, _ = run_segment(
        cfg, condition="static_gain", seed=0, noise_pa=0.0, static_gain=0.4,
        start_state=a, end_ms=cfg.simulation.duration_ms,
    )
    rc, _ = run_segment(
        cfg, condition="static_gain", seed=0, noise_pa=0.0, static_gain=0.4,
        start_state=c, end_ms=cfg.simulation.duration_ms,
    )
    assert np.array_equal(ra.spike_times_e_ms, rc.spike_times_e_ms)
    assert np.array_equal(ra.plfp_gaba_e, rc.plfp_gaba_e)


def test_plus_minus_co_shift_preserves_spike_times():
    cfg = _cfg()
    e_bins, i_bins = aligned_roll_bins(22.5, cfg.network.ne, cfg.network.ni)
    _, state = run_segment(cfg, condition="dynamic", seed=4, noise_pa=0.0, end_ms=80.0)
    plus, _ = run_segment(
        cfg, condition="dynamic", seed=4, noise_pa=0.0,
        start_state=apply_co_shift(state, e_bins, i_bins),
        end_ms=cfg.simulation.duration_ms,
    )
    minus, _ = run_segment(
        cfg, condition="dynamic", seed=4, noise_pa=0.0,
        start_state=apply_co_shift(state, -e_bins, -i_bins),
        end_ms=cfg.simulation.duration_ms,
    )
    t0 = 80.0
    assert np.array_equal(plus.spike_times_e_ms[plus.spike_times_e_ms >= t0],
                         minus.spike_times_e_ms[minus.spike_times_e_ms >= t0])


def test_misaligned_delta_is_rejected():
    try:
        aligned_roll_bins(10.0, 64, 16)
    except ValueError as exc:
        assert "integer" in str(exc)
    else:
        raise AssertionError("non-lattice delta must raise")


def test_pilot_branch_count_is_248():
    from experiment.phase11_pilot import PILOT_DELTA_DEG, iter_pilot_branches
    branches = iter_pilot_branches()
    assert len(branches) == 248
    e_bins, i_bins = aligned_roll_bins(PILOT_DELTA_DEG, 1024, 256)
    assert (e_bins, i_bins) == (96, 24)


def test_recovery_fraction_uses_abs_delta():
    from analysis.phase11_pilot import directed_displacement, recovery_fraction

    stayed_plus = directed_displacement(33.75, 33.75)
    stayed_minus = directed_displacement(-33.75, -33.75)
    assert float(stayed_plus) == 33.75
    assert float(stayed_minus) == 33.75
    assert abs(float(recovery_fraction(stayed_plus, 33.75))) < 1e-12
    assert abs(float(recovery_fraction(stayed_minus, -33.75))) < 1e-12
    returned = directed_displacement(0.0, -33.75)
    assert float(recovery_fraction(returned, -33.75)) == 1.0
    # The previous bug produced R=2 on a minus-delta no-recovery trial.
    assert float(recovery_fraction(stayed_minus, -33.75)) != 2.0
