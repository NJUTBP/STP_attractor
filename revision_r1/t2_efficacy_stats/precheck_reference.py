"""v3 section 6.5 precheck on EXISTING continuous-gain reference trials (retrospective diagnostic, no simulation).

For the full model at the reference point it reports, from an exact replay of u and R from the recorded spikes:
  * spikes per (active neuron, theta cycle) and active fraction per cycle,
  * the resource R- just before each spike, overall and for first versus later spikes within a cycle,
  * the event amplitude u+ R- each spike would have under the event-scaled convention.
These numbers inform the reading of P1b but do not change any P1b decision rule (PREREGISTRATION section 6.5).

  python revision_r1/p1b_event_scaled/precheck_reference.py ^
      --npz-dirs results/phase6_corrected_full/theta/dynamic ^
                 results/phase10_facilitation_routes/combos/tf_1000_tr_20/probe_full/dynamic ^
      --out results/revision_r1/p1b/precheck
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(CODE_ROOT / "config" / "phase4_causal.toml"))
    ap.add_argument("--window", nargs=2, type=float, default=(1000.0, 8000.0))
    a = ap.parse_args(argv)
    from config import load_config
    from revision_r1.t2_efficacy_stats.event_stats import A_EDGES, R_EDGES, counts_matrix, event_distribution_metrics
    from revision_r1.t2_efficacy_stats.reconstruct import load_trial, replay_events
    cfg = load_config(a.config)
    U, tf, tr = cfg.stf.u_baseline, cfg.stf.tau_facilitation_ms, cfg.stf.tau_recovery_ms
    ne = cfg.network.ne
    t0, t1 = a.window
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    r_hist = np.zeros(R_EDGES.size - 1, np.int64)
    a_hist = np.zeros(A_EDGES.size - 1, np.int64)
    count_dist = {}
    for d in a.npz_dirs:
        files = sorted(Path(d).glob("seed_*.npz"))
        if not files:
            print(f"WARNING: no seed_*.npz in {d}")
        for f in files:
            raw = load_trial(f)
            if raw["condition"] != "dynamic" or abs(raw["temporal_gain_scale"] - 1.0) > 1e-12 or raw["noise_pa"] != 0:
                continue
            if raw["seed"] in seen:            # the same seed in two folders is the same deterministic trial
                continue
            seen.add(raw["seed"])
            ev = replay_events(raw, "dynamic", U=U, tau_f=tf, tau_r=tr, t_end_ms=t1, dt_ms=cfg.simulation.dt_ms, ne=ne)
            row, arr = event_distribution_metrics(ev, raw, mode="dynamic", U=U, t0=t0, t1=t1, ne=ne)
            row.update(seed=raw["seed"], path=str(f))
            rows.append(row)
            r_hist += arr.get("Rminus_hist", 0)
            a_hist += arr.get("event_amp_hist", 0)
            M, _cy = counts_matrix(raw, t0, t1, ne)
            vals, cnt = np.unique(M[M > 0], return_counts=True)
            for v, c in zip(vals, cnt):
                count_dist[int(v)] = count_dist.get(int(v), 0) + int(c)
    if not rows:
        raise SystemExit("no continuous full 0-pA trials found")
    df = pd.DataFrame(rows).sort_values("seed")
    df.to_csv(out / "precheck_per_trial.csv", index=False)
    pd.DataFrame({"R_lo": R_EDGES[:-1], "R_hi": R_EDGES[1:], "n_spikes": r_hist}).to_csv(out / "Rminus_hist.csv", index=False)
    pd.DataFrame({"a_lo": A_EDGES[:-1], "a_hi": A_EDGES[1:], "n_spikes": a_hist}).to_csv(out / "event_amp_hist.csv",
                                                                                         index=False)
    tot = sum(count_dist.values())
    summary = {
        "n_trials": int(len(df)), "seeds": [int(s) for s in df.seed],
        "window_ms": [t0, t1],
        "spikes_per_active_neuron_cycle": {str(k): count_dist[k] / tot for k in sorted(count_dist)},
        "mean_over_trials": {c: float(df[c].mean()) for c in df.columns
                             if c not in ("seed", "path") and np.issubdtype(df[c].dtype, np.number)},
        "note": "retrospective diagnostic of the continuous-gain convention; does not change P1b rules",
    }
    (out / "precheck_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
        ax[0].step(R_EDGES[:-1], r_hist / max(r_hist.sum(), 1), where="post")
        ax[0].set_yscale("log"); ax[0].set_xlabel("R- at spikes"); ax[0].set_ylabel("fraction of spikes")
        ks = sorted(count_dist)
        ax[1].bar(ks, [count_dist[k] / tot for k in ks])
        ax[1].set_xlabel("spikes per active neuron per cycle"); ax[1].set_ylabel("fraction of (neuron, cycle)")
        fig.suptitle(f"continuous full, reference point, {len(df)} trials", fontsize=9)
        fig.tight_layout(); fig.savefig(out / "precheck.png", dpi=160); plt.close(fig)
    except Exception:
        pass
    keys = ["Rminus_mean", "Rminus_q01", "Rminus_q05", "Rminus_frac_below_0p9", "frac_spikes_later_in_cycle",
            "Rminus_mean_first_in_cycle", "Rminus_mean_later_in_cycle", "event_amp_mean", "event_amp_cv",
            "frac_cycle_with_1_spike", "mean_spikes_per_active_neuron_cycle", "f0_pop_hz"]
    print(json.dumps({k: summary["mean_over_trials"].get(k) for k in keys}, indent=2))
    print("spikes per active neuron-cycle:", json.dumps(summary["spikes_per_active_neuron_cycle"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
