from __future__ import annotations

import numpy as np

from config import load_config
from model import simulate
from model.stp import DynamicSTFState


def test_mean_u_release_at_baseline_u_equals_depletion_only_exactly():
    cfg=load_config("config/smoke.toml")
    a=simulate(cfg,condition="depletion_only",seed=3,noise_pa=15.0)
    b=simulate(
        cfg,
        condition="mean_u_release",
        seed=3,
        noise_pa=15.0,
        resource_release_u_override=cfg.stf.u_baseline,
    )
    assert np.array_equal(a.spike_times_e_ms,b.spike_times_e_ms)
    assert np.array_equal(a.spike_angles_e_deg,b.spike_angles_e_deg)
    assert np.array_equal(a.plfp_gaba_e,b.plfp_gaba_e)
    assert np.array_equal(a.mean_efficacy_by_e,b.mean_efficacy_by_e)


def test_phase10_hybrid_spike_updates_separate_output_and_release_routes():
    cfg=load_config("config/smoke.toml")
    spikes=np.array([True,False,True,False])

    full=DynamicSTFState.initialize(4,cfg.stf)
    output=DynamicSTFState.initialize(4,cfg.stf)
    release=DynamicSTFState.initialize(4,cfg.stf)
    dep=DynamicSTFState.initialize(4,cfg.stf)

    full.on_spikes(spikes,cfg.stf,"dynamic")
    output.on_spikes(spikes,cfg.stf,"u_output_only")
    release.on_spikes(spikes,cfg.stf,"u_release_only")
    dep.on_spikes(spikes,cfg.stf,"depletion_only")

    # Full, output-only, and release-only all update u dynamically.
    assert np.array_equal(full.u,output.u)
    assert np.array_equal(full.u,release.u)
    assert not np.array_equal(full.u,dep.u)

    # Full and release-only share dynamic-u resource depletion.
    assert np.array_equal(full.resources,release.resources)
    # Output-only and depletion-only share fixed-U resource depletion.
    assert np.array_equal(output.resources,dep.resources)
    assert not np.array_equal(full.resources,output.resources)


def test_resource_release_recording_uses_actual_spike_release_probability():
    cfg=load_config("config/smoke.toml")
    r=simulate(cfg,condition="dynamic",seed=0,noise_pa=0.0)
    assert r.resource_release_u_count > 0
    assert np.isfinite(r.mean_resource_release_u)
    assert cfg.stf.u_baseline <= r.mean_resource_release_u <= 1.0


def test_release_override_is_restricted_to_mean_u_release():
    cfg=load_config("config/smoke.toml")
    try:
        simulate(
            cfg,condition="depletion_only",seed=0,noise_pa=0.0,
            resource_release_u_override=0.6,
        )
    except ValueError as exc:
        assert "resource_release_u_override" in str(exc)
    else:
        raise AssertionError("release override must not silently affect other conditions")
