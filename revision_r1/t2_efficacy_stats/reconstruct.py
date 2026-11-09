"""Exact reconstruction of E->I synaptic state from recorded E spikes (T2).

Conventions copied from model/simulator.py and model/stp.py (frozen; not imported for the update rules):
* A spike recorded at time t_s = n*dt (start of step n) updates u and R at the END of step n, i.e. at
  t_s + dt, after the exponential recovery over that step (same as make_figures/dynamics.replay_stp).
* NMDA: s uses the midpoint scheme of the simulator; x decays exactly; x += 1 after the s update of the
  spike step.  The state "at time t" is the state after completing step t/dt - 1.
* Efficacy per mode = model.stp.temporal_efficacy_from_components(...) * temporal_gain_scale.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

DT_MS = 0.02
U_DYNAMIC = frozenset({"dynamic", "facilitation_only", "u_output_only", "u_release_only"})
RESOURCE = frozenset({"dynamic", "depletion_only", "u_output_only", "u_release_only", "mean_u_release"})
TEMPORAL = U_DYNAMIC | RESOURCE
CONSTANT = frozenset({"baseline", "static_gain", "frozen_profile"})
ALL_MODES = TEMPORAL | CONSTANT

_TAU_RE = re.compile(r"tf_(\d+(?:p\d+)?)_tr_(\d+(?:p\d+)?)")
_U_RE = re.compile(r"u_(\d+p\d+|\d+)")


def _num(token: str) -> float:
    return float(token.replace("p", "."))


def parse_combo_id(cid: str | None, default=(0.4, 1000.0, 20.0)) -> tuple[float, float, float]:
    """(U, tau_f, tau_r) from ids such as 'tf_1000_tr_20' or 'u_0p1_tf_50_tr_800'."""
    U, tf, tr = default
    if cid:
        m = _TAU_RE.search(cid)
        if m:
            tf, tr = _num(m.group(1)), _num(m.group(2))
        mu = _U_RE.match(cid)
        if mu:
            U = _num(mu.group(1))
    return float(U), float(tf), float(tr)


def load_trial(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    raw["condition"] = str(np.asarray(raw["condition"]).item())
    raw["seed"] = int(raw["seed"])
    for key in ("noise_pa", "static_gain", "temporal_gain_scale", "resource_release_u_override",
                "gain_match_numerator", "gain_match_denominator", "resource_release_u_sum", "resource_release_u_count"):
        if key in raw:
            raw[key] = float(raw[key])
    raw.setdefault("temporal_gain_scale", 1.0)
    raw.setdefault("resource_release_u_override", float("nan"))
    raw.setdefault("static_gain", float("nan"))
    return raw


def neuron_index(raw: dict, ne: int) -> np.ndarray:
    return np.rint(np.asarray(raw["spike_angles_e_deg"], float) * ne / 360.0).astype(np.int64) % ne


@dataclass
class Events:
    ids: np.ndarray        # E neuron index per event (sorted by neuron, then time)
    t_spike: np.ndarray    # recorded spike time (ms)
    t_eff: np.ndarray      # time at which the update is in the state (t_spike + dt)
    u_pre: np.ndarray
    r_pre: np.ndarray
    u_post: np.ndarray
    r_post: np.ndarray
    rho: np.ndarray        # release fraction actually used to deplete R (0 if no depletion)


def replay_events(raw: dict, mode: str, *, U: float, tau_f: float, tau_r: float, t_end_ms: float,
                  release_override: float | None = None, dt_ms: float = DT_MS, ne: int | None = None) -> Events:
    if mode not in TEMPORAL:
        raise ValueError(f"replay_events only applies to temporal modes, got {mode}")
    if mode == "mean_u_release" and (release_override is None or not np.isfinite(release_override)):
        raise ValueError("mean_u_release needs the release override stored in the NPZ")
    ne = ne or len(raw["theta_e_deg"])
    ids = neuron_index(raw, ne)
    ts = np.asarray(raw["spike_times_e_ms"], float)
    keep = ts + dt_ms <= t_end_ms + 1e-9
    ids, ts = ids[keep], ts[keep]
    order = np.lexsort((ts, ids))
    ids, ts = ids[order], ts[order]
    te = ts + dt_ms
    n = len(ts)
    u_pre = np.empty(n); r_pre = np.empty(n); u_post = np.empty(n); r_post = np.empty(n); rho = np.zeros(n)
    u_dyn = mode in U_DYNAMIC
    depletes = mode in RESOURCE
    bounds = np.searchsorted(ids, np.arange(ne + 1), side="left")
    for i in np.unique(ids):
        a, b = bounds[i], bounds[i + 1]
        u, r, t_last = U, 1.0, 0.0
        for k in range(a, b):
            d = te[k] - t_last
            if u_dyn:
                u = U + (u - U) * np.exp(-d / tau_f)
            r = 1.0 + (r - 1.0) * np.exp(-d / tau_r)
            u_pre[k], r_pre[k] = u, r
            if u_dyn:
                u = u + U * (1.0 - u)
            if depletes:
                if mode in ("dynamic", "u_release_only"):
                    f = u
                elif mode == "mean_u_release":
                    f = float(release_override)
                else:
                    f = U
                rho[k] = f
                r = r - r * f
            u = min(max(u, 0.0), 1.0)
            r = min(max(r, 0.0), 1.0)
            u_post[k], r_post[k] = u, r
            t_last = te[k]
    return Events(ids, ts, te, u_pre, r_pre, u_post, r_post, rho)


def sample_uR(ev: Events, t_ms: np.ndarray, neurons: np.ndarray, *, U: float, tau_f: float, tau_r: float,
              u_dynamic: bool) -> tuple[np.ndarray, np.ndarray]:
    """u(t), R(t) for the listed neurons at sample times (state after the step ending at t)."""
    t_ms = np.asarray(t_ms, float)
    u = np.full((len(t_ms), len(neurons)), U, float)
    r = np.ones((len(t_ms), len(neurons)), float)
    for j, i in enumerate(neurons):
        sel = ev.ids == i
        if not np.any(sel):
            continue
        knots = np.r_[0.0, ev.t_eff[sel]]
        uk = np.r_[U, ev.u_post[sel]]
        rk = np.r_[1.0, ev.r_post[sel]]
        k = np.searchsorted(knots, t_ms + 1e-9, side="right") - 1
        d = np.maximum(0.0, t_ms - knots[k])
        if u_dynamic:
            u[:, j] = U + (uk[k] - U) * np.exp(-d / tau_f)
        r[:, j] = 1.0 + (rk[k] - 1.0) * np.exp(-d / tau_r)
    return u, r


def replay_s_nmda(raw: dict, t_ms: np.ndarray, neurons: np.ndarray, *, dt_ms: float = DT_MS, tau_x: float = 2.0,
                  tau_nmda: float = 100.0, alpha: float = 0.5, ne: int | None = None) -> np.ndarray:
    """s_NMDA of the listed neurons at sample times, integrated exactly as in model/simulator.py."""
    ne = ne or len(raw["theta_e_deg"])
    neurons = np.asarray(neurons, int)
    local = -np.ones(ne, int)
    local[neurons] = np.arange(len(neurons))
    ids = neuron_index(raw, ne)
    steps = np.rint(np.asarray(raw["spike_times_e_ms"], float) / dt_ms).astype(np.int64)
    sel = local[ids] >= 0
    steps, lid = steps[sel], local[ids[sel]]
    order = np.argsort(steps, kind="stable")
    steps, lid = steps[order], lid[order]
    sample_steps = np.rint(np.asarray(t_ms, float) / dt_ms).astype(np.int64) - 1
    if np.any(np.diff(sample_steps) <= 0) or sample_steps[0] < 0:
        raise ValueError("sample times must be increasing and > dt")
    n_end = int(sample_steps[-1]) + 1
    s = np.zeros(len(neurons)); x = np.zeros(len(neurons))
    out = np.empty((len(sample_steps), len(neurons)))
    dec_half = np.exp(-dt_ms / (2.0 * tau_x)); dec = np.exp(-dt_ms / tau_x)
    inv_tau = 1.0 / tau_nmda
    sp_ptr = 0; n_sp = len(steps); out_ptr = 0
    next_sample = sample_steps[0]
    for n in range(n_end):
        k1 = -s * inv_tau + alpha * x * (1.0 - s)
        s_half = s + 0.5 * dt_ms * k1
        x_half = x * dec_half
        k2 = -s_half * inv_tau + alpha * x_half * (1.0 - s_half)
        x *= dec
        s = s + dt_ms * k2
        while sp_ptr < n_sp and steps[sp_ptr] == n:
            x[lid[sp_ptr]] += 1.0
            sp_ptr += 1
        if n == next_sample:
            out[out_ptr] = s
            out_ptr += 1
            if out_ptr < len(sample_steps):
                next_sample = sample_steps[out_ptr]
    return out


def efficacy(mode: str, u, r, *, U: float, k: float = 1.0, p_static: float | None = None, frozen=None):
    """Efficacy multiplier p exactly as used by the simulator (times temporal_gain_scale k)."""
    u = np.asarray(u, float); r = np.asarray(r, float)
    if mode in ("dynamic", "u_output_only"):
        return k * u * r
    if mode == "facilitation_only":
        return k * u
    if mode in ("depletion_only", "u_release_only", "mean_u_release"):
        return k * U * r
    if mode == "baseline":
        return np.full(np.broadcast(u, r).shape, U)
    if mode == "static_gain":
        return np.full(np.broadcast(u, r).shape, float(p_static))
    if mode == "frozen_profile":
        return np.broadcast_to(np.asarray(frozen, float), np.broadcast(u, r).shape).copy()
    raise ValueError(f"unknown mode {mode}")
