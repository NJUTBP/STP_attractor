from __future__ import annotations

import numpy as np

from config import load_config
from experiment.phase14_tether import iter_phase14_branches
from model.simulator import axial_profile_summary, simulate
from model.state import RingNetworkState, apply_named_shift


def _dummy_state(n_e: int = 8, n_i: int = 4) -> RingNetworkState:
    return RingNetworkState(
        step=0,
        ve=np.arange(n_e, dtype=float), vi=np.arange(n_i, dtype=float),
        ref_e=np.arange(n_e, dtype=np.int32), ref_i=np.arange(n_i, dtype=np.int32),
        x_nmda=np.arange(n_e, dtype=float)+10, s_nmda=np.arange(n_e, dtype=float)+20,
        s_gaba=np.arange(n_i, dtype=float)+30,
        noise_e=np.arange(n_e, dtype=float)+40, noise_i=np.arange(n_i, dtype=float)+50,
        u=np.arange(n_e, dtype=float)/100+0.4,
        resources=1.0-np.arange(n_e, dtype=float)/100,
        rng_noise_e_state={}, rng_noise_i_state={},
        spike_times_e_ms=[], spike_angles_e_deg=[], spike_times_i_ms=[], spike_angles_i_deg=[],
        sample_times_ms=np.zeros(10), plfp_gaba_e=np.zeros(10), sample_idx=0,
        efficacy_sum=np.zeros(n_e), efficacy_count=0, match_num=0.0, match_den=0.0,
        release_u_sum=0.0, release_u_count=0,
    )


def test_axial_profile_summary_recovers_known_axis():
    theta=np.arange(0.0,360.0,45.0)
    weights=np.zeros_like(theta)
    weights[2]=1.0  # 90 degrees; axial representation returns 90.
    axis,res,mass=axial_profile_summary(weights,theta)
    assert abs(axis-90.0)<1e-12
    assert abs(res-1.0)<1e-12
    assert mass>0


def test_selective_state_surgery_shifts_only_requested_stp_field():
    s=_dummy_state()
    e_bins,i_bins=2,1
    au=apply_named_shift(s,"activity_u_shift",e_bins,i_bins)
    ar=apply_named_shift(s,"activity_r_shift",e_bins,i_bins)
    assert np.array_equal(au.u,np.roll(s.u,e_bins))
    assert np.array_equal(au.resources,s.resources)
    assert np.array_equal(ar.u,s.u)
    assert np.array_equal(ar.resources,np.roll(s.resources,e_bins))
    # Both surgeries shift fast activity.
    assert np.array_equal(au.ve,np.roll(s.ve,e_bins))
    assert np.array_equal(ar.vi,np.roll(s.vi,i_bins))


def test_stp_trace_recorder_is_optional_and_temporal_only():
    cfg=load_config("config/smoke.toml")
    r=simulate(cfg,condition="dynamic",seed=0,noise_pa=0.0,record_stp_trace=True)
    assert r.stp_trace_times_ms.size>0
    assert r.stp_trace_times_ms.size==r.stp_u_axis_deg.size==r.stp_r_axis_deg.size
    assert np.all(np.isfinite(r.stp_u_resultant))
    assert np.all(np.isfinite(r.stp_r_resultant))
    s=simulate(cfg,condition="static_gain",seed=0,noise_pa=0.0,static_gain=0.5,record_stp_trace=True)
    assert s.stp_trace_times_ms.size==0


def test_phase14_branch_design_has_no_duplicates_and_expected_count():
    branches=iter_phase14_branches()
    ids=[b.branch_id for b in branches]
    assert len(ids)==len(set(ids))
    # Per seed: 4 no-shift + 4 conditions*6 activity shifts + dynamic 3 surgeries*4 shifts = 40.
    assert len(branches)==8*40
    assert sum(b.shift_kind=="activity_u_shift" for b in branches)==8*4
    assert sum(b.shift_kind=="activity_r_shift" for b in branches)==8*4


def test_phase14_csv_writer_accepts_summary_only_fields(tmp_path):
    import csv
    from analysis.phase14_tether import _write_csv

    path = tmp_path / "mixed_schema.csv"
    rows = [
        {"condition": "dynamic", "seed": 0, "field_rate_s_inv": 0.1},
        {
            "condition": "dynamic",
            "seed": "summary",
            "field_rate_s_inv": 0.12,
            "ci_low": 0.02,
            "ci_high": 0.20,
        },
    ]
    _write_csv(path, rows)

    with path.open(newline="", encoding="utf-8") as f:
        loaded = list(csv.DictReader(f))

    assert loaded[0]["ci_low"] == ""
    assert loaded[0]["ci_high"] == ""
    assert loaded[1]["ci_low"] == "0.02"
    assert loaded[1]["ci_high"] == "0.2"
