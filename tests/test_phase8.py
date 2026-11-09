from __future__ import annotations

from dataclasses import replace
import numpy as np

from config.phase8 import Phase8SweepConfig
from config.schema import ProjectConfig, STFConfig
from model.stp import DynamicSTFState
from model import simulate


def test_temporal_component_spike_updates_are_distinct():
    cfg = STFConfig(u_baseline=0.4, tau_facilitation_ms=1000.0, tau_recovery_ms=20.0)
    spikes = np.array([True])

    full = DynamicSTFState.initialize(1, cfg)
    full.on_spikes(spikes, cfg, "dynamic")
    assert np.allclose(full.u, [0.64])
    assert np.allclose(full.resources, [0.36])
    assert np.allclose(full.efficacy_for_mode("dynamic", cfg), [0.2304])

    fac = DynamicSTFState.initialize(1, cfg)
    fac.on_spikes(spikes, cfg, "facilitation_only")
    assert np.allclose(fac.u, [0.64])
    assert np.allclose(fac.resources, [1.0])
    assert np.allclose(fac.efficacy_for_mode("facilitation_only", cfg), [0.64])

    dep = DynamicSTFState.initialize(1, cfg)
    dep.on_spikes(spikes, cfg, "depletion_only")
    assert np.allclose(dep.u, [0.4])
    assert np.allclose(dep.resources, [0.6])
    assert np.allclose(dep.efficacy_for_mode("depletion_only", cfg), [0.24])


def test_phase8_default_run_count():
    sweep = Phase8SweepConfig(
        tau_facilitation_ms=(100.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2500.0),
        tau_recovery_ms=(5.0, 10.0, 20.0, 40.0, 80.0, 160.0),
        seeds=(0, 1, 2, 3),
    )
    sweep.validate()
    assert len(sweep.combos) == 42
    assert sweep.expected_run_count() == 2192


def test_new_temporal_modes_run_and_gain_match_on_small_network():
    base = ProjectConfig()
    cfg = replace(
        base,
        network=replace(base.network, ne=64, ni=16),
        simulation=replace(base.simulation, duration_ms=200.0, cue_end_ms=50.0, sampling_rate_hz=200),
        analysis=replace(
            base.analysis,
            theta_start_ms=50.0,
            theta_end_ms=200.0,
            gain_match_start_ms=50.0,
            gain_match_end_ms=180.0,
            pav_target_time_ms=150.0,
            localization_reference_start_ms=50.0,
            localization_reference_end_ms=100.0,
            localization_target_start_ms=120.0,
            localization_target_end_ms=180.0,
        ),
    )
    for mode in ("dynamic", "facilitation_only", "depletion_only"):
        r = simulate(cfg, condition=mode, seed=0, noise_pa=0.0)
        assert r.condition == mode
        assert np.isfinite(r.mean_efficacy_by_e).all()
        assert r.gain_match_denominator >= 0.0
