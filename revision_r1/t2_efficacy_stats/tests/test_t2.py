"""Unit tests for T2 (no network simulation)."""
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[3]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from revision_r1.t2_efficacy_stats.metrics import axial_width_deg, u_plus_periodic  # noqa: E402
from revision_r1.t2_efficacy_stats.reconstruct import (efficacy, parse_combo_id, replay_events,  # noqa: E402
                                                     replay_s_nmda, sample_uR)


def _raw(times, ids, ne=8):
    theta = np.arange(ne) * 360.0 / ne
    return {"theta_e_deg": theta, "spike_times_e_ms": np.asarray(times, float),
            "spike_angles_e_deg": theta[np.asarray(ids, int)]}


def test_single_spike_update_dynamic_and_output():
    U = 0.4
    raw = _raw([10.0], [3])
    ev = replay_events(raw, "dynamic", U=U, tau_f=1000.0, tau_r=20.0, t_end_ms=100.0)
    assert np.isclose(ev.u_post[0], U + U * (1 - U))
    assert np.isclose(ev.r_post[0], 1 - (U + U * (1 - U)))
    ev2 = replay_events(raw, "u_output_only", U=U, tau_f=1000.0, tau_r=20.0, t_end_ms=100.0)
    assert np.isclose(ev2.r_post[0], 1 - U)
    ev3 = replay_events(raw, "mean_u_release", U=U, tau_f=1000.0, tau_r=20.0, t_end_ms=100.0, release_override=0.9)
    assert np.isclose(ev3.u_post[0], U) and np.isclose(ev3.r_post[0], 0.1)


def test_sampling_decay():
    U = 0.4
    raw = _raw([10.0], [0])
    ev = replay_events(raw, "dynamic", U=U, tau_f=1000.0, tau_r=20.0, t_end_ms=200.0)
    t = np.array([10.02, 60.02])
    u, r = sample_uR(ev, t, np.array([0, 1]), U=U, tau_f=1000.0, tau_r=20.0, u_dynamic=True)
    assert np.isclose(u[0, 0], ev.u_post[0]) and np.isclose(u[0, 1], U)
    assert np.isclose(u[1, 0], U + (ev.u_post[0] - U) * np.exp(-50 / 1000.0))
    assert np.isclose(r[1, 0], 1 + (ev.r_post[0] - 1) * np.exp(-50 / 20.0))


def test_s_nmda_single_spike_shape():
    raw = _raw([5.0], [2])
    t = np.arange(5.02, 300.0, 1.0)
    s = replay_s_nmda(raw, t, np.array([2]))
    assert s.min() >= 0 and s.max() <= 1
    assert 0.55 < s.max() < 0.66
    assert s[-1] < s.max() * np.exp(-1.5)


def test_efficacy_and_parsing():
    u = np.array([0.9]); r = np.array([0.5])
    assert np.isclose(efficacy("dynamic", u, r, U=0.4), 0.45)
    assert np.isclose(efficacy("u_release_only", u, r, U=0.4, k=2.0), 0.4)
    assert np.isclose(efficacy("static_gain", u, r, U=0.4, p_static=0.7), 0.7)
    assert parse_combo_id("u_0p1_tf_50_tr_800") == (0.1, 50.0, 800.0)
    assert parse_combo_id("tf_1000_tr_20") == (0.4, 1000.0, 20.0)


def test_u_plus_formula_matches_iteration():
    U, tf, f = 0.4, 1000.0, 6.4453125
    u = U
    for _ in range(200):
        u = U + (u - U) * np.exp(-1000.0 / f / tf)
        u = u + U * (1 - u)
    assert np.isclose(u_plus_periodic(f, U, tf), u)
    assert abs(u_plus_periodic(f, U, tf) - 0.894) < 0.001


def test_axial_width():
    th = np.arange(1024) * 360 / 1024
    w = np.exp(-0.5 * (((th - 90 + 90) % 180 - 90) / 10.0) ** 2)
    assert 9.0 < axial_width_deg(w, th) < 11.0
