"""Continue a checkpointed trial in short chunks and store spatial profiles of u, R and s_NMDA.

Recovery fix: when a continuation extends beyond the SimulationConfig duration used to create
its checkpoint (Module C does this by design), expand the checkpoint's preallocated sampled-current
buffers before calling ``run_segment``.  Without this, Module C reaches the old 8-s allocation and
raises IndexError even though the config duration has been extended.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from model.connectivity import build_connectivity
from model.simulator import run_segment
from phase16.stp_clamp import u_clamp


def with_duration(cfg, end_ms: float):
    if end_ms <= cfg.simulation.duration_ms:
        return cfg
    return replace(cfg, simulation=replace(cfg.simulation, duration_ms=float(end_ms)))


def _ensure_sample_capacity(state, cfg):
    """Return a state whose sampled-current buffers cover cfg.simulation.duration_ms."""
    dt = float(cfg.simulation.dt_ms)
    sample_every = int(round((1000.0 / cfg.simulation.sampling_rate_hz) / dt))
    steps = int(round(cfg.simulation.duration_ms / dt))
    needed = steps // sample_every
    have = len(state.sample_times_ms)
    if have >= needed:
        return state
    out = state.copy()
    t = np.empty(needed, dtype=float)
    p = np.empty(needed, dtype=float)
    t[:have] = out.sample_times_ms
    p[:have] = out.plfp_gaba_e
    # The tail is capacity only; sample_idx marks valid elements.
    t[have:] = np.nan
    p[have:] = np.nan
    out.sample_times_ms = t
    out.plfp_gaba_e = p
    return out


def run_with_snapshots(cfg, *, condition, seed, noise_pa, start_state, end_ms, every_ms=None,
                       clamp_u=False, noise_draw=None, **kw):
    """Return (final_result, final_state, snaps). ``every_ms=None`` runs one segment without profiles."""
    cfg = with_duration(cfg, end_ms)
    start_state = _ensure_sample_capacity(start_state, cfg)
    dt = cfg.simulation.dt_ms
    conn = build_connectivity(cfg.network)
    t_now = start_state.step * dt
    if every_ms is None:
        with u_clamp(clamp_u):
            res, state = run_segment(cfg, condition=condition, seed=seed, noise_pa=noise_pa, start_state=start_state,
                                     end_ms=end_ms, connectivity=conn, noise_draw=noise_draw, **kw)
        return res, state, None
    snaps = {"t": [], "u": [], "R": [], "s": []}
    state, first, res = start_state, True, None
    while t_now < end_ms - 1e-9:
        t_next = min(end_ms, round((t_now + every_ms) / dt) * dt)
        with u_clamp(clamp_u):
            res, state = run_segment(cfg, condition=condition, seed=seed, noise_pa=noise_pa, start_state=state,
                                     end_ms=t_next, connectivity=conn,
                                     noise_draw=noise_draw if first else None, **kw)
        first, t_now = False, t_next
        snaps["t"].append(t_now)
        snaps["s"].append(state.s_nmda.astype(np.float32))
        if state.u is not None:
            snaps["u"].append(state.u.astype(np.float32))
            snaps["R"].append(state.resources.astype(np.float32))
    out = {"t": np.asarray(snaps["t"]), "s": np.stack(snaps["s"])}
    if snaps["u"]:
        out["u"], out["R"] = np.stack(snaps["u"]), np.stack(snaps["R"])
    return res, state, out
