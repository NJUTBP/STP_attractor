import numpy as np

from config import load_config
from model import simulate


def test_baseline_equals_static_gain_when_pmatch_equals_u():
    cfg = load_config("config/smoke.toml")
    baseline = simulate(cfg, condition="baseline", seed=3, noise_pa=20.0)
    static = simulate(
        cfg,
        condition="static_gain",
        seed=3,
        noise_pa=20.0,
        static_gain=cfg.stf.u_baseline,
    )
    assert np.array_equal(baseline.spike_times_e_ms, static.spike_times_e_ms)
    assert np.array_equal(baseline.spike_angles_e_deg, static.spike_angles_e_deg)
    assert np.array_equal(baseline.plfp_gaba_e, static.plfp_gaba_e)


def test_dynamic_is_ei_only_and_has_gain_match_bookkeeping():
    cfg = load_config("config/smoke.toml")
    dynamic = simulate(cfg, condition="dynamic", seed=0, noise_pa=0.0)
    assert dynamic.mean_efficacy_by_e.shape == (cfg.network.ne,)
    assert np.all(dynamic.mean_efficacy_by_e >= 0.0)
    assert np.all(dynamic.mean_efficacy_by_e <= 1.0)
    assert dynamic.gain_match_denominator >= 0.0


def test_frozen_profile_equals_static_gain_when_profile_is_constant():
    cfg = load_config("config/smoke.toml")
    gain = 0.55
    profile = np.full(cfg.network.ne, gain, dtype=float)
    frozen = simulate(
        cfg,
        condition="frozen_profile",
        seed=4,
        noise_pa=20.0,
        frozen_profile=profile,
    )
    static = simulate(
        cfg,
        condition="static_gain",
        seed=4,
        noise_pa=20.0,
        static_gain=gain,
    )
    assert np.array_equal(frozen.spike_times_e_ms, static.spike_times_e_ms)
    assert np.array_equal(frozen.spike_angles_e_deg, static.spike_angles_e_deg)
    assert np.array_equal(frozen.plfp_gaba_e, static.plfp_gaba_e)
    assert np.array_equal(frozen.mean_efficacy_by_e, profile)


def test_frozen_profile_rejects_wrong_shape():
    cfg = load_config("config/smoke.toml")
    bad = np.full(cfg.network.ne - 1, 0.5)
    try:
        simulate(cfg, condition="frozen_profile", seed=0, noise_pa=0.0, frozen_profile=bad)
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("wrong-shaped frozen_profile should raise ValueError")
