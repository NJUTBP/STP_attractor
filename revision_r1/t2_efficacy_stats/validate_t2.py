"""T2 validation against the frozen simulator (short runs; code check only, not a scientific result).

Runs the reference configuration, seed 0, 0 pA, to 1000 ms, then in 1-ms segments to --t-end, recording the true
u, R and s_NMDA.  Reconstruction from the recorded spikes must match: u, R < 1e-9; s < 1e-6; M1 (sampled every dt)
equal to the simulator's own gain-match ratio to < 1e-9 relative; mean release fraction equal to the NPZ value.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np  # noqa: E402

MODES = {  # mode: (temporal_gain_scale, release_override, static_gain)
    "dynamic": (1.0, None, None),
    "u_output_only": (0.8721143, None, None),
    "u_release_only": (2.1329120, None, None),
    "mean_u_release": (2.1339830, 0.8887143, None),
    "depletion_only": (2.0, None, None),
    "static_gain": (1.0, None, 0.7042313),
}


def run_mode(cfg, conn, mode, t_end):
    from model.simulator import run_segment
    k, ov, sg = MODES[mode]
    kw = dict(condition=mode, seed=0, noise_pa=0.0, connectivity=conn, static_gain=sg,
              temporal_gain_scale=k, resource_release_u_override=ov)
    res, state = run_segment(cfg, end_ms=1000.0, **kw)
    times, U_, R_, S_ = [], [], [], []
    t = 1000.0
    while t < t_end - 1e-9:
        t = round(t + 1.0, 6)
        res, state = run_segment(cfg, start_state=state, end_ms=t, **kw)
        times.append(t); S_.append(state.s_nmda.copy())
        if state.u is not None:
            U_.append(state.u.copy()); R_.append(state.resources.copy())
    return res, np.array(times), (np.array(U_) if U_ else None), (np.array(R_) if R_ else None), np.array(S_)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t-end", type=float, default=1200.0)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args(argv)
    from config import load_config
    from io_utils import save_result
    from model.connectivity import build_connectivity
    from revision_r1.t2_efficacy_stats.metrics import compute_trial_metrics
    from revision_r1.t2_efficacy_stats.reconstruct import (TEMPORAL, U_DYNAMIC, load_trial, replay_events,
                                                         replay_s_nmda, sample_uR)
    cfg = load_config(CODE_ROOT / "config" / "phase4_causal.toml")
    conn = build_connectivity(cfg.network)
    U, tf, tr = cfg.stf.u_baseline, cfg.stf.tau_facilitation_ms, cfg.stf.tau_recovery_ms
    report, ok_all = {}, True
    for mode in a.modes.split(","):
        res, times, Ut, Rt, St = run_mode(cfg, conn, mode, a.t_end)
        with tempfile.TemporaryDirectory() as tmp:
            raw = load_trial(save_result(res, Path(tmp) / "seed_0000.npz"))
        ne = len(conn.theta_e_deg)
        allN = np.arange(ne)
        s_rec = replay_s_nmda(raw, times, allN, dt_ms=cfg.simulation.dt_ms, tau_x=cfg.network.tau_x_ms,
                              tau_nmda=cfg.network.tau_nmda_ms, alpha=cfg.network.alpha_nmda_khz, ne=ne)
        r = {"max_abs_err_s": float(np.max(np.abs(s_rec - St)))}
        if mode in TEMPORAL:
            ev = replay_events(raw, mode, U=U, tau_f=tf, tau_r=tr, t_end_ms=a.t_end,
                               release_override=MODES[mode][1], dt_ms=cfg.simulation.dt_ms, ne=ne)
            u_rec, R_rec = sample_uR(ev, times, allN, U=U, tau_f=tf, tau_r=tr, u_dynamic=mode in U_DYNAMIC)
            r["max_abs_err_u"] = float(np.max(np.abs(u_rec - Ut)))
            r["max_abs_err_R"] = float(np.max(np.abs(R_rec - Rt)))
        row, _ = compute_trial_metrics(raw, mode=mode, U=U, tau_f=tf, tau_r=tr, w_out=conn.ei_presynaptic_weight_sum,
                                       theta_e=conn.theta_e_deg, ni=cfg.network.ni, window=(1000.0, a.t_end),
                                       sample_ms=cfg.simulation.dt_ms, dt_ms=cfg.simulation.dt_ms)
        r["M1"] = row["M1_p_drive_weighted"]; r["npz_p_match"] = row["npz_p_match"]
        if mode in TEMPORAL:
            ref_m1 = r["npz_p_match"]          # the simulator accumulates the match only for temporal modes
        else:
            ref_m1 = MODES[mode][2] if mode == "static_gain" else U   # constant efficacy: compare with the set value
        r["M1_reference"] = ref_m1
        r["rel_err_M1"] = abs(r["M1"] - ref_m1) / abs(ref_m1)
        r["rho_spike_mean"] = row["M5_rho_spike_mean"]; r["npz_release_u_mean"] = row["npz_release_u_mean"]
        ok = (r["max_abs_err_s"] < 1e-6 and r["rel_err_M1"] < 1e-9
              and r.get("max_abs_err_u", 0.0) < 1e-9 and r.get("max_abs_err_R", 0.0) < 1e-9)
        if mode in TEMPORAL and mode not in ("facilitation_only",) and np.isfinite(r["npz_release_u_mean"]):
            ok = ok and abs(r["rho_spike_mean"] - r["npz_release_u_mean"]) < 1e-9
        r["PASS"] = bool(ok); ok_all &= ok
        report[mode] = r
        print(mode, json.dumps(r), flush=True)
    report["ALL_PASS"] = bool(ok_all)
    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("ALL_PASS" if ok_all else "VALIDATION_FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
