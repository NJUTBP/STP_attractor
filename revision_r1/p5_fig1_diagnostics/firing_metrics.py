"""P5 (Revision Plan v3 section 10): firing diagnostics of existing runs.  Descriptive only; no thresholds.

Bump membership is time resolved: the axial bump center is computed every 50 ms from the E spikes in a centered
400-ms window, so a bump that wanders under noise does not inflate the off-bump rates.  Because the network has two
bumps 180 deg apart, distances are axial (period 180 deg).
  off-bump: axial distance from the current center > 45 deg      in-bump (core): <= 20 deg
Per neuron, the off-bump rate is (spikes while off-bump) / (time spent off-bump); "off-bump neurons" are those that
spent at least half of the window off-bump.  Cycles come from the E population rate (common/spikes.py).
"""
from __future__ import annotations

import numpy as np

from revision_r1.common.spikes import (assign_cycles, axial_center, axial_dist, center_trace, dominant_frequency,
                                       neuron_index, per_cycle_counts, quantiles)

OFF_DEG = 45.0
IN_DEG = 20.0
BIN_MS = 50.0
CENTER_WINDOW_MS = 400.0
RATE_EDGES = np.r_[0.0, np.logspace(-2, 2, 41)]        # first bin [0, 0.01) Hz holds silent neurons


def plfp_frequency(sample_times_ms, plfp, t0, t1) -> float:
    """The paper's estimate (Welch, Hamming 256, overlap 128, nfft 1024, 200 Hz) when the current proxy exists."""
    from scipy import signal
    t = np.asarray(sample_times_ms, float)
    x = np.asarray(plfp, float)
    m = (t >= t0) & (t <= t1)
    if m.sum() < 256:
        return float("nan")
    f, p = signal.welch(x[m], fs=200.0, window=signal.get_window("hamming", 256), nperseg=256, noverlap=128,
                        nfft=1024)
    sel = (f > 0) & (f <= 50)
    return float(f[sel][np.argmax(p[sel])])


def firing_metrics(spike_t, spike_a, ne: int, t0: float, t1: float, *, sample_times_ms=None, plfp=None,
                   off_deg: float = OFF_DEG, in_deg: float = IN_DEG, bin_ms: float = BIN_MS,
                   center_window_ms: float = CENTER_WINDOW_MS) -> tuple[dict, dict]:
    t_all = np.asarray(spike_t, float)
    a_all = np.asarray(spike_a, float)
    inw = (t_all >= t0) & (t_all < t1)
    t, a = t_all[inw], a_all[inw]
    ids = neuron_index(a, ne)
    theta = np.arange(ne) * 360.0 / ne
    dur_s = (t1 - t0) / 1000.0
    counts = np.bincount(ids, minlength=ne)
    rates = counts / dur_s
    row = {"window_start_ms": t0, "window_end_ms": t1, "n_e_spikes": int(t.size), "all_mean_rate_hz": float(rates.mean()),
           "all_frac_silent": float(np.mean(counts == 0)), "all_frac_ge_0p1hz": float(np.mean(rates >= 0.1)),
           "all_frac_ge_1hz": float(np.mean(rates >= 1.0))}
    row.update(quantiles(rates, (0.1, 0.25, 0.5, 0.75, 0.9, 0.99), prefix="all_rate_"))
    _c, res = axial_center(a)
    row["resultant_all_spikes"] = res
    # time-resolved bump center and membership
    edges = np.arange(t0, t1 + 1e-9, bin_ms)
    mids = edges[:-1] + bin_ms / 2.0
    nb = mids.size
    c, _r, _n = center_trace(t, a, mids, center_window_ms)
    valid = np.isfinite(c)
    row["frac_bins_with_center"] = float(valid.mean()) if nb else float("nan")
    dist = axial_dist(theta[None, :], np.where(valid, c, 0.0)[:, None])
    off = (dist > off_deg) & valid[:, None]
    inb = (dist <= in_deg) & valid[:, None]
    b = np.clip(((t - t0) // bin_ms).astype(np.int64), 0, max(nb - 1, 0))
    off_t = off.sum(axis=0) * bin_ms / 1000.0
    in_t = inb.sum(axis=0) * bin_ms / 1000.0
    off_spk = np.bincount(ids[off[b, ids]], minlength=ne) if t.size else np.zeros(ne, int)
    in_spk = np.bincount(ids[inb[b, ids]], minlength=ne) if t.size else np.zeros(ne, int)
    off_neurons = off_t >= 0.5 * dur_s
    off_rate = off_spk[off_neurons] / off_t[off_neurons]
    row.update(off_n_neurons=int(off_neurons.sum()),
               off_median_rate_hz=float(np.median(off_rate)) if off_rate.size else float("nan"),
               off_mean_rate_hz=float(off_spk.sum() / off_t.sum()) if off_t.sum() > 0 else float("nan"),
               off_q90_rate_hz=float(np.quantile(off_rate, 0.9)) if off_rate.size else float("nan"),
               off_frac_any_spike=float(np.mean(off_spk[off_neurons] > 0)) if off_rate.size else float("nan"),
               off_frac_ge_0p1hz=float(np.mean(off_rate >= 0.1)) if off_rate.size else float("nan"),
               in_mean_rate_hz=float(in_spk.sum() / in_t.sum()) if in_t.sum() > 0 else float("nan"))
    om = row["off_mean_rate_hz"]
    row["in_off_ratio"] = float(row["in_mean_rate_hz"] / om) if (np.isfinite(om) and om > 0) else float("inf")
    if valid.any():
        first, last = c[np.flatnonzero(valid)[0]], c[np.flatnonzero(valid)[-1]]
        row["center_change_deg"] = float(((last - first + 90.0) % 180.0) - 90.0)
    # cycles
    cy = assign_cycles(t, t0, t1)
    row["f0_pop_hz"] = cy["f0_hz"]
    row["n_cycles"] = int(cy["n_cycles"])
    if sample_times_ms is not None and plfp is not None:
        row["f_plfp_hz"] = plfp_frequency(sample_times_ms, plfp, t0, t1)
    arrays = {"rates": rates, "off_rates": off_rate}
    if cy["n_cycles"] >= 3:
        M = per_cycle_counts(ids, cy["cycle"], cy["n_cycles"], ne)
        act = (M > 0).mean(axis=1)
        row.update(quantiles(act, (0.1, 0.5, 0.9), prefix="active_frac_per_cycle_"))
        row["active_frac_per_cycle_mean"] = float(act.mean())
        nz = M[M > 0]                                   # data-driven: every (neuron, cycle) with >= 1 spike
        if nz.size:
            row.update(active_pairs_frac_1=float(np.mean(nz == 1)), active_pairs_frac_2=float(np.mean(nz == 2)),
                       active_pairs_frac_3plus=float(np.mean(nz >= 3)), active_pairs_mean=float(nz.mean()))
        kb = np.clip(((cy["mid_ms"] - t0) // bin_ms).astype(np.int64), 0, nb - 1)
        cm = c[kb]
        okc = np.isfinite(cm)
        if okc.any():
            dc = axial_dist(theta[None, :], cm[okc][:, None])
            Mo = M[okc]
            in_c, off_c = dc <= in_deg, dc > off_deg
            bump = Mo[in_c]
            row.update(bump_frac_cycles_0=float(np.mean(bump == 0)), bump_frac_cycles_1=float(np.mean(bump == 1)),
                       bump_frac_cycles_2=float(np.mean(bump == 2)), bump_frac_cycles_3plus=float(np.mean(bump >= 3)),
                       bump_mean_spikes_per_cycle=float(bump.mean()))
            n_off = off_c.sum(axis=1)
            fo = ((Mo > 0) & off_c).sum(axis=1) / np.maximum(n_off, 1)
            row["off_active_frac_per_cycle_mean"] = float(fo[n_off > 0].mean()) if np.any(n_off > 0) else float("nan")
            arrays["bump_counts"] = np.bincount(np.minimum(bump, 5), minlength=6)
    return row, arrays


def rate_histogram(rates) -> np.ndarray:
    r = np.asarray(rates, float)
    h = np.histogram(np.clip(r, 0.0, RATE_EDGES[-1] - 1e-9), bins=RATE_EDGES)[0]
    return h
