"""Synthetic-data tests for common/spikes.py, t2 event_stats and P5 firing metrics (no network simulation)."""
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[3]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from revision_r1.common.spikes import assign_cycles, axial_dist, cycle_firing_summary  # noqa: E402
from revision_r1.p5_fig1_diagnostics.firing_metrics import firing_metrics  # noqa: E402

NE = 1024
F = 6.45


def _bump_train(t1=8000.0, half_width=15.0, doublets=False, off_rate=0.0, seed=0, drift_deg_per_s=0.0):
    """Two bumps (90, 270 deg); core neurons fire once per cycle (optionally twice); optional Poisson background."""
    rng = np.random.default_rng(seed)
    theta = np.arange(NE) * 360.0 / NE
    T = 1000.0 / F
    ts, ang = [], []
    for k in range(int(t1 / T)):
        tc = k * T
        center = (90.0 + drift_deg_per_s * tc / 1000.0) % 180.0
        core = np.flatnonzero(axial_dist(theta, center) <= half_width)
        jit = rng.normal(0, 4.0, core.size)
        ts += list(tc + 20.0 + jit); ang += list(theta[core])
        if doublets:
            ts += list(tc + 32.0 + jit); ang += list(theta[core])
    if off_rate > 0:
        n = rng.poisson(off_rate * t1 / 1000.0 * NE)
        ts += list(rng.uniform(0, t1, n)); ang += list(theta[rng.integers(0, NE, n)])
    o = np.argsort(ts)
    return np.asarray(ts)[o], np.asarray(ang)[o]


def test_cycles_one_spike_per_cycle():
    t, a = _bump_train()
    ids = np.rint(a * NE / 360.0).astype(int) % NE
    s = cycle_firing_summary(ids, t, 1000.0, 8000.0, NE)
    assert abs(s["f0_pop_hz"] - F) < 0.3
    assert s["frac_cycle_with_1_spike"] > 0.99
    assert abs(s["spikes_per_cycle_spike_weighted"] - 1.0) < 0.05
    assert s["frac_isi_below_half_period"] < 0.01


def test_cycles_doublets():
    t, a = _bump_train(doublets=True)
    ids = np.rint(a * NE / 360.0).astype(int) % NE
    s = cycle_firing_summary(ids, t, 1000.0, 8000.0, NE)
    assert s["frac_cycle_with_2_spikes"] > 0.95
    assert 0.45 < s["frac_isi_below_half_period"] < 0.55


def test_assign_cycles_counts():
    t, _a = _bump_train()
    cy = assign_cycles(t, 1000.0, 8000.0)
    expected = 7.0 * F - 2                              # partial edge cycles dropped
    assert abs(cy["n_cycles"] - expected) <= 2


def test_p5_silent_background():
    # synthetic bump wider than the 20-deg core, so every core neuron fires once per cycle
    t, a = _bump_train(half_width=25.0)
    row, arr = firing_metrics(t, a, NE, 1000.0, 8000.0)
    assert row["off_median_rate_hz"] == 0.0 and row["off_frac_any_spike"] == 0.0
    assert abs(row["in_mean_rate_hz"] - F) < 0.6
    assert row["bump_frac_cycles_1"] > 0.95 and row["active_pairs_frac_1"] > 0.99
    core = np.mean(axial_dist(np.arange(NE) * 360.0 / NE, 90.0) <= 25.0)
    assert abs(row["active_frac_per_cycle_mean"] - core) < 0.02
    assert row["in_off_ratio"] == float("inf")
    # a bump narrower than the core: core cells at the edge are silent in some cycles (definitional, reported)
    row2, _ = firing_metrics(*_bump_train(half_width=15.0), NE, 1000.0, 8000.0)
    assert row2["bump_frac_cycles_0"] > 0.1 and row2["active_pairs_frac_1"] > 0.99


def test_p5_sparse_background_and_drift():
    t, a = _bump_train(off_rate=0.5, drift_deg_per_s=4.0, seed=3)
    row, _ = firing_metrics(t, a, NE, 1000.0, 8000.0)
    assert 0.3 < row["off_median_rate_hz"] < 0.8
    assert row["off_frac_any_spike"] > 0.9
    assert abs(row["center_change_deg"] - 4.0 * 6.8) < 4.0


def test_event_stats_on_replay():
    from revision_r1.t2_efficacy_stats.event_stats import event_distribution_metrics
    from revision_r1.t2_efficacy_stats.reconstruct import replay_events
    t, a = _bump_train(t1=4000.0)
    raw = {"theta_e_deg": np.arange(NE) * 360.0 / NE, "spike_times_e_ms": t, "spike_angles_e_deg": a}
    ev = replay_events(raw, "dynamic", U=0.4, tau_f=1000.0, tau_r=20.0, t_end_ms=4000.0, ne=NE)
    row, arr = event_distribution_metrics(ev, raw, mode="dynamic", U=0.4, t0=1000.0, t1=4000.0, ne=NE)
    # one spike per ~155 ms with tau_r = 20 ms: R- is essentially fully recovered
    assert row["Rminus_q05"] > 0.99 and row["Rminus_frac_below_0p9"] == 0.0
    assert 0.85 < row["event_amp_mean"] < 0.9
    assert arr["Rminus_hist"].sum() == np.count_nonzero((t >= 1000.0) & (t < 4000.0))
