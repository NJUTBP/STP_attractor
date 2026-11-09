"""T2 runner: inventory of existing NPZ -> rule-based selection -> metrics per trial (no new simulations).

From the code root:
  python revision_r1/t2_efficacy_stats/run_t2.py --roots results/phase10_facilitation_routes \
      results/phase8_mechanism_map results/phase9_gain_rescued_depletion --out results/revision_r1/t2_efficacy --plan
  ... same command with --execute --workers 8
--plan writes inventory.csv and selection.csv only.  Selection: noise 0 pA, stages not starting with "cal",
requested modes, runs covering the window; duplicates (same combo/condition/seed/scale/override) kept once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_MODES = ("dynamic", "u_output_only", "u_release_only", "mean_u_release", "depletion_only",
                 "static_gain", "frozen_profile", "baseline")
STAGE_PRIORITY = {"theta": 0, "probe_full": 1}


def inventory(roots) -> pd.DataFrame:
    rows = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            raise SystemExit(f"MISSING_INPUT: {root}")
        for p in sorted(root.rglob("seed_*.npz")):
            parts = p.relative_to(root).parts
            cid = parts[parts.index("combos") + 1] if "combos" in parts else ""
            with np.load(p, allow_pickle=False) as z:
                st = z["sample_times_ms"]
                rows.append({
                    "path": str(p), "root": str(root), "combo_id": cid,
                    "stage": parts[-3] if len(parts) >= 3 else "", "label": parts[-2] if len(parts) >= 2 else "",
                    "condition": str(z["condition"].item()), "seed": int(z["seed"]), "noise_pa": float(z["noise_pa"]),
                    "temporal_gain_scale": float(z["temporal_gain_scale"]) if "temporal_gain_scale" in z.files else 1.0,
                    "release_override": float(z["resource_release_u_override"]) if "resource_release_u_override" in z.files else float("nan"),
                    "static_gain": float(z["static_gain"]),
                    "t_last_sample_ms": float(st[-1]) if st.size else float("nan"),
                    "size": p.stat().st_size,
                })
    return pd.DataFrame(rows)


def select(inv: pd.DataFrame, *, modes, settings, window_end: float) -> pd.DataFrame:
    s = inv[(inv.noise_pa == 0.0) & ~inv.stage.str.lower().str.startswith("cal")
            & inv.condition.isin(modes) & (inv.t_last_sample_ms >= window_end - 1e-6)].copy()
    if settings and settings != ["all"]:
        s = s[s.combo_id.isin(settings)]
    s["prio"] = s.stage.map(STAGE_PRIORITY).fillna(9)
    s = s.sort_values(["prio", "path"])
    dkey = pd.DataFrame({"c": s.combo_id, "m": s.condition, "seed": s.seed, "k": s.temporal_gain_scale.round(9),
                         "ov": s.release_override.round(9).fillna(-1), "sg": s.static_gain.round(9).fillna(-1)})
    s = s[~dkey.duplicated(keep="first").to_numpy()]
    return s.drop(columns="prio").reset_index(drop=True)


def _trial_key(row) -> str:
    return hashlib.sha1(f"{row['path']}|{row['size']}".encode()).hexdigest()[:16]


def _work(task):
    row, out_dir, cfg_path, default_combo, window, sample_ms = task
    from config import load_config
    from model.connectivity import build_connectivity
    from revision_r1.t2_efficacy_stats.metrics import compute_trial_metrics
    from revision_r1.t2_efficacy_stats.reconstruct import load_trial, parse_combo_id
    cfg = load_config(cfg_path)
    conn = build_connectivity(cfg.network)
    U, tf, tr = parse_combo_id(row["combo_id"] or default_combo,
                               default=(cfg.stf.u_baseline, cfg.stf.tau_facilitation_ms, cfg.stf.tau_recovery_ms))
    raw = load_trial(row["path"])
    t_start = time.time()
    metrics, arrays = compute_trial_metrics(
        raw, mode=row["condition"], U=U, tau_f=tf, tau_r=tr, w_out=conn.ei_presynaptic_weight_sum,
        theta_e=conn.theta_e_deg, ni=cfg.network.ni, window=window, sample_ms=sample_ms, dt_ms=cfg.simulation.dt_ms,
        nmda={"tau_x": cfg.network.tau_x_ms, "tau_nmda": cfg.network.tau_nmda_ms, "alpha": cfg.network.alpha_nmda_khz})
    out = {k: row[k] for k in ("path", "combo_id", "stage", "label", "condition", "seed", "temporal_gain_scale",
                               "release_override", "static_gain")}
    out.update({"U": U, "tau_f_ms": tf, "tau_r_ms": tr, "seconds": round(time.time() - t_start, 2)})
    out.update(metrics)
    key = _trial_key(row)
    np.savez_compressed(Path(out_dir) / "per_trial" / f"{key}.npz", **arrays)
    (Path(out_dir) / "per_trial" / f"{key}.json").write_text(json.dumps(out, default=float), encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(CODE_ROOT / "config" / "phase4_causal.toml"))
    ap.add_argument("--default-combo", default="tf_1000_tr_20", help="for runs stored outside combos/<id>/")
    ap.add_argument("--modes", default=",".join(DEFAULT_MODES))
    ap.add_argument("--settings", default="all", help="comma-separated combo ids or 'all'")
    ap.add_argument("--window", nargs=2, type=float, default=(1000.0, 7000.0))
    ap.add_argument("--sample-ms", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="smoke runs: analyse only the first N selected trials")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--plan", action="store_true", help="inventory + selection only (default)")
    g.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out); (out / "per_trial").mkdir(parents=True, exist_ok=True)
    inv = inventory(a.roots)
    inv.to_csv(out / "inventory.csv", index=False)
    sel = select(inv, modes=a.modes.split(","), settings=a.settings.split(","), window_end=a.window[1])
    if a.limit:
        sel = sel.head(a.limit)
    sel.to_csv(out / "selection.csv", index=False)
    print(f"inventory {len(inv)} NPZ; selected {len(sel)} trials")
    print(sel.groupby(["condition"]).size().to_string())
    if not a.execute:
        print("PLAN ONLY (use --execute to compute).")
        return 0
    from revision_r1.common.manifest import write_manifest
    write_manifest(out, task="T2_efficacy_stats", argv=sys.argv, code_root=CODE_ROOT,
                   extra={"roots": a.roots, "n_selected": int(len(sel)), "window": list(a.window), "sample_ms": a.sample_ms})
    todo, rows = [], []
    for _, r in sel.iterrows():
        j = out / "per_trial" / f"{_trial_key(r)}.json"
        if j.exists():
            rows.append(json.loads(j.read_text(encoding="utf-8")))
        else:
            todo.append((r.to_dict(), str(out), a.config, a.default_combo, tuple(a.window), a.sample_ms))
    print(f"{len(rows)} already done, {len(todo)} to compute")
    if a.workers <= 1:
        for i, t in enumerate(todo, 1):
            rows.append(_work(t)); print(f"[{i}/{len(todo)}] {t[0]['combo_id']} {t[0]['condition']} seed {t[0]['seed']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(_work, t) for t in todo]
            for i, f in enumerate(as_completed(futs), 1):
                rows.append(f.result())
                if i % 10 == 0 or i == len(todo):
                    print(f"[{i}/{len(todo)}]", flush=True)
    pd.DataFrame(rows).sort_values(["combo_id", "condition", "seed"]).to_csv(out / "per_trial_metrics.csv", index=False)
    print(f"wrote {out / 'per_trial_metrics.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
