"""Spike-train utilities shared by T2/P2, P1b and P5 (pure NumPy/SciPy; no simulation).

Cycles are defined from the E population rate: the rate in 1-ms bins is band-passed around its dominant frequency
f0, the Hilbert phase is unwrapped, and every spike gets the phase at its time.  Cycle boundaries are placed half a
cycle away from the circular-mean spike phase, so a volley is never split across two cycles.  The first and last
(possibly partial) cycles are discarded.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def neuron_index(angles_deg, ne: int) -> np.ndarray:
    return np.rint(np.asarray(angles_deg, float) * ne / 360.0).astype(np.int64) % ne


def axial_center(angles_deg, weights=None) -> tuple[float, float]:
    """Doubled-angle center in [0, 180) and normalized resultant length."""
    a = np.asarray(angles_deg, float)
    if a.size == 0:
        return float("nan"), 0.0
    w = np.ones_like(a) if weights is None else np.asarray(weights, float)
    tot = w.sum()
    if tot <= 0:
        return float("nan"), 0.0
    z = np.sum(w * np.exp(2j * np.deg2rad(a)))
    return float((np.rad2deg(np.angle(z)) / 2.0) % 180.0), float(np.abs(z) / tot)


def axial_dist(theta_deg, center_deg) -> np.ndarray:
    """|axial distance| in [0, 90] deg (period 180 deg: the two bumps are equivalent)."""
    return np.abs(((np.asarray(theta_deg, float) - np.asarray(center_deg, float) + 90.0) % 180.0) - 90.0)


def center_trace(spike_t, spike_a, grid_ms, window_ms: float = 400.0, min_spikes: int = 5):
    """Axial center of the spikes in a centered window around each grid time (NaN if < min_spikes)."""
    t = np.asarray(spike_t, float)
    order = np.argsort(t, kind="stable")
    t = t[order]
    z = np.exp(2j * np.deg2rad(np.asarray(spike_a, float)[order]))
    cz = np.concatenate([[0j], np.cumsum(z)])
    g = np.asarray(grid_ms, float)
    lo = np.searchsorted(t, g - window_ms / 2.0, side="left")
    hi = np.searchsorted(t, g + window_ms / 2.0, side="left")
    n = hi - lo
    s = cz[hi] - cz[lo]
    c = (np.rad2deg(np.angle(s)) / 2.0) % 180.0
    c = np.where(n >= min_spikes, c, np.nan)
    r = np.where(n > 0, np.abs(s) / np.maximum(n, 1), np.nan)
    return c, r, n


def population_rate(spike_t, t0: float, t1: float, bin_ms: float = 1.0):
    edges = np.arange(t0, t1 + 1e-9, bin_ms)
    counts, _ = np.histogram(np.asarray(spike_t, float), bins=edges)
    return 0.5 * (edges[:-1] + edges[1:]), counts.astype(float)


def dominant_frequency(x, fs: float, fmin: float = 1.0, fmax: float = 50.0) -> float:
    x = np.asarray(x, float)
    if x.size < 64 or not np.any(x != x.mean()):
        return float("nan")
    x = x - x.mean()
    nper = min(x.size, int(round(fs * 1.28)))
    f, p = signal.welch(x, fs=fs, window="hamming", nperseg=nper, noverlap=nper // 2, nfft=max(4 * nper, 1024))
    m = (f >= fmin) & (f <= fmax)
    return float(f[m][np.argmax(p[m])]) if np.any(m) else float("nan")


def assign_cycles(spike_t, t0: float, t1: float, f0: float | None = None, bin_ms: float = 1.0) -> dict:
    """Cycle index of every spike in [t0, t1) (-1 for spikes outside kept cycles).

    Returns dict(cycle=int array aligned with spike_t, n_cycles, f0_hz, first, last, mid_ms (per kept cycle)).
    """
    t = np.asarray(spike_t, float)
    out = dict(cycle=np.full(t.size, -1, dtype=np.int64), n_cycles=0, f0_hz=float("nan"), first=0, last=-1,
               mid_ms=np.empty(0))
    inw = (t >= t0) & (t < t1)
    if np.count_nonzero(inw) < 20:
        return out
    tc, rate = population_rate(t[inw], t0, t1, bin_ms)
    fs = 1000.0 / bin_ms
    f = dominant_frequency(rate, fs) if f0 is None else float(f0)
    if not np.isfinite(f) or f <= 0 or (t1 - t0) < 3000.0 / f:
        return out
    lo, hi = 0.6 * f, min(1.6 * f, 0.45 * fs)
    sos = signal.butter(2, [lo, hi], btype="bandpass", fs=fs, output="sos")
    ph = np.unwrap(np.angle(signal.hilbert(signal.sosfiltfilt(sos, rate - rate.mean()))))
    sp_ph = np.interp(t[inw], tc, ph)
    mean_ph = float(np.angle(np.mean(np.exp(1j * sp_ph))))
    cyc_all = np.floor((ph - mean_ph + np.pi) / (2 * np.pi)).astype(np.int64)
    cyc_sp = np.floor((sp_ph - mean_ph + np.pi) / (2 * np.pi)).astype(np.int64)
    first, last = int(cyc_all.min()) + 1, int(cyc_all.max()) - 1           # drop partial edge cycles
    if last < first:
        return out
    keep = (cyc_sp >= first) & (cyc_sp <= last)
    c = np.full(t.size, -1, dtype=np.int64)
    idx = np.flatnonzero(inw)
    c[idx[keep]] = cyc_sp[keep] - first
    mids = np.array([tc[cyc_all == k].mean() for k in range(first, last + 1)])
    out.update(cycle=c, n_cycles=last - first + 1, f0_hz=f, first=first, last=last, mid_ms=mids)
    return out


def per_cycle_counts(ids, cycle, n_cycles: int, ne: int) -> np.ndarray:
    """Matrix (n_cycles x ne) of spike counts per neuron and kept cycle."""
    ids = np.asarray(ids, np.int64)
    cycle = np.asarray(cycle, np.int64)
    m = cycle >= 0
    out = np.zeros((max(n_cycles, 0), ne), dtype=np.int64)
    if n_cycles > 0 and np.any(m):
        np.add.at(out, (cycle[m], ids[m]), 1)
    return out


def isi_over_period(ids, t, period_ms: float) -> np.ndarray:
    """Inter-spike intervals of every neuron (pooled) divided by the population period."""
    ids = np.asarray(ids, np.int64)
    t = np.asarray(t, float)
    order = np.lexsort((t, ids))
    ids, t = ids[order], t[order]
    same = ids[1:] == ids[:-1]
    isi = (t[1:] - t[:-1])[same]
    return isi / period_ms if np.isfinite(period_ms) and period_ms > 0 else np.full(isi.size, np.nan)


def quantiles(x, qs=(0.05, 0.25, 0.5, 0.75, 0.95), prefix: str = "") -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"{prefix}q{int(round(q * 100)):02d}": float("nan") for q in qs}
    v = np.quantile(x, qs)
    return {f"{prefix}q{int(round(q * 100)):02d}": float(a) for q, a in zip(qs, v)}


def cycle_firing_summary(ids, t, t0: float, t1: float, ne: int, f0: float | None = None) -> dict:
    """Spikes per active neuron per cycle, active fraction per cycle, ISI/period (no bump classification)."""
    ids = np.asarray(ids, np.int64)
    t = np.asarray(t, float)
    inw = (t >= t0) & (t < t1)
    cy = assign_cycles(t[inw], t0, t1, f0)
    out = {"f0_pop_hz": cy["f0_hz"], "n_cycles": int(cy["n_cycles"])}
    if cy["n_cycles"] < 3:
        return out
    M = per_cycle_counts(ids[inw], cy["cycle"], cy["n_cycles"], ne)
    act = M[M > 0]
    out.update(
        pairs_active=int(act.size),
        frac_cycle_with_1_spike=float(np.mean(act == 1)) if act.size else float("nan"),
        frac_cycle_with_2_spikes=float(np.mean(act == 2)) if act.size else float("nan"),
        frac_cycle_with_3plus=float(np.mean(act >= 3)) if act.size else float("nan"),
        mean_spikes_per_active_neuron_cycle=float(act.mean()) if act.size else float("nan"),
        active_fraction_per_cycle_mean=float(np.mean((M > 0).mean(axis=1))),
    )
    per_neuron = M.sum(axis=0) / cy["n_cycles"]
    firing = per_neuron[per_neuron > 0]
    out.update(quantiles(firing, (0.1, 0.5, 0.9), prefix="spikes_per_cycle_per_neuron_"))
    out["spikes_per_cycle_spike_weighted"] = float(np.sum(firing ** 2) / np.sum(firing)) if firing.size else float("nan")
    r = isi_over_period(ids[inw], t[inw], 1000.0 / cy["f0_hz"])
    out.update(quantiles(r, (0.1, 0.5, 0.9), prefix="isi_over_period_"))
    out["frac_isi_below_half_period"] = float(np.mean(r < 0.5)) if r.size else float("nan")
    return out
