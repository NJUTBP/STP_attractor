from __future__ import annotations

import numpy as np

from config import load_config
from experiment.gain_rescue import initial_rescue_scale, corrected_rescue_scale
from model import simulate


def test_rescue_scale_helpers():
    assert np.isclose(initial_rescue_scale(0.7, 0.35), 2.0)
    assert np.isclose(corrected_rescue_scale(2.0, 0.7, 0.56), 2.5)


def test_temporal_gain_scale_one_is_exactly_identity():
    cfg = load_config("config/smoke.toml")
    a = simulate(cfg, condition="depletion_only", seed=7, noise_pa=20.0)
    b = simulate(cfg, condition="depletion_only", seed=7, noise_pa=20.0, temporal_gain_scale=1.0)
    assert np.array_equal(a.spike_times_e_ms, b.spike_times_e_ms)
    assert np.array_equal(a.spike_angles_e_deg, b.spike_angles_e_deg)
    assert np.array_equal(a.plfp_gaba_e, b.plfp_gaba_e)
    assert np.array_equal(a.mean_efficacy_by_e, b.mean_efficacy_by_e)


def test_temporal_gain_scale_is_recorded_and_changes_depletion_efficacy():
    cfg = load_config("config/smoke.toml")
    a = simulate(cfg, condition="depletion_only", seed=1, noise_pa=0.0, temporal_gain_scale=1.0)
    b = simulate(cfg, condition="depletion_only", seed=1, noise_pa=0.0, temporal_gain_scale=1.5)
    assert b.temporal_gain_scale == 1.5
    assert np.isfinite(b.mean_efficacy_by_e).all()
    assert np.allclose(b.mean_efficacy_by_e, 1.5 * a.mean_efficacy_by_e)
    assert np.isclose(b.gain_match_numerator, 1.5 * a.gain_match_numerator)


def test_temporal_gain_scale_rejected_for_static_condition():
    cfg = load_config("config/smoke.toml")
    try:
        simulate(cfg, condition="baseline", seed=0, noise_pa=0.0, temporal_gain_scale=2.0)
    except ValueError as exc:
        assert "temporal_gain_scale" in str(exc)
    else:
        raise AssertionError("non-temporal conditions must reject temporal_gain_scale != 1")
