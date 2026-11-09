"""v3 additions to T2 (Revision Plan v3, sections 6.5 and 7 item 6): distributions behind the P1 audit argument.

For every spike in the analysis window:
  R-       resource just before the spike (after recovery, before this spike's depletion)
  a        event amplitude the spike would have under the event-scaled convention, without the scale factor k:
           u+ R- (full, output-only), U R- (release-only, mean-release, neither), u+ (facilitation-only)
and per (neuron, cycle): the number of spikes, with cycles from the E population rate (common/spikes.py).
"First" spikes are the first spike of a neuron in a cycle; "later" spikes follow within the same cycle.
"""
from __future__ import annotations

import numpy as np

from revision_r1.common.spikes import assign_cycles, cycle_firing_summary, per_cycle_counts, quantiles

R_EDGES = np.linspace(0.0, 1.0, 101)
A_EDGES = np.linspace(0.0, 1.0, 101)


def event_amplitude(mode: str, u_post, r_pre, U: float) -> np.ndarray:
    u_post = np.asarray(u_post, float)
    r_pre = np.asarray(r_pre, float)
    if mode in ("dynamic", "u_output_only"):
        return u_post * r_pre
    if mode in ("u_release_only", "mean_u_release", "depletion_only"):
        return U * r_pre
    if mode == "facilitation_only":
        return u_post
    raise ValueError(mode)


def event_distribution_metrics(ev, raw: dict, *, mode: str, U: float, t0: float, t1: float, ne: int) -> tuple[dict, dict]:
    """ev: reconstruct.Events of a temporal mode, or None for constant-efficacy modes."""
    ts = np.asarray(raw["spike_times_e_ms"], float)
    ids = np.rint(np.asarray(raw["spike_angles_e_deg"], float) * ne / 360.0).astype(np.int64) % ne
    row = cycle_firing_summary(ids, ts, t0, t1, ne)
    arrays: dict = {}
    if ev is None:
        return row, arrays
    win = (ev.t_spike >= t0) & (ev.t_spike < t1)
    r = ev.r_pre[win]
    a = event_amplitude(mode, ev.u_post[win], r, U)
    row.update(quantiles(r, (0.01, 0.05, 0.25, 0.5, 0.75, 0.95), prefix="Rminus_"))
    row["Rminus_mean"] = float(np.mean(r)) if r.size else float("nan")
    row["Rminus_frac_below_0p9"] = float(np.mean(r < 0.9)) if r.size else float("nan")
    row["Rminus_frac_below_0p99"] = float(np.mean(r < 0.99)) if r.size else float("nan")
    row.update(quantiles(a, (0.05, 0.25, 0.5, 0.75, 0.95), prefix="event_amp_"))
    row["event_amp_mean"] = float(np.mean(a)) if a.size else float("nan")
    row["event_amp_cv"] = float(np.std(a) / np.mean(a)) if a.size and np.mean(a) > 0 else float("nan")
    # first versus later spikes within a cycle
    cy = assign_cycles(ev.t_spike[win], t0, t1)
    if cy["n_cycles"] >= 3:
        c = cy["cycle"]
        ok = c >= 0
        key = ev.ids[win][ok] * (cy["n_cycles"] + 1) + c[ok]
        order = np.lexsort((ev.t_spike[win][ok], key))
        k_sorted = key[order]
        first_sorted = np.r_[True, k_sorted[1:] != k_sorted[:-1]]
        is_first = np.empty(order.size, bool)
        is_first[order] = first_sorted
        r_ok = r[ok]
        row["frac_spikes_later_in_cycle"] = float(np.mean(~is_first))
        row["Rminus_mean_first_in_cycle"] = float(np.mean(r_ok[is_first])) if np.any(is_first) else float("nan")
        row["Rminus_mean_later_in_cycle"] = float(np.mean(r_ok[~is_first])) if np.any(~is_first) else float("nan")
    arrays["Rminus_hist"] = np.histogram(r, bins=R_EDGES)[0].astype(np.int64)
    arrays["event_amp_hist"] = np.histogram(a, bins=A_EDGES)[0].astype(np.int64)
    return row, arrays


def counts_matrix(raw: dict, t0: float, t1: float, ne: int):
    """(n_cycles x ne) spike counts and the cycle dict, for plots of spikes per cycle."""
    ts = np.asarray(raw["spike_times_e_ms"], float)
    ids = np.rint(np.asarray(raw["spike_angles_e_deg"], float) * ne / 360.0).astype(np.int64) % ne
    inw = (ts >= t0) & (ts < t1)
    cy = assign_cycles(ts[inw], t0, t1)
    return per_cycle_counts(ids[inw], cy["cycle"], cy["n_cycles"], ne), cy
