"""R1-T4 runner: oscillation frequency at an imposed constant release fraction rho (mean_u_release mode).

  python revision_r1/t4_release_fraction/run_t4.py --plan
  python revision_r1/t4_release_fraction/run_t4.py --smoke
  python revision_r1/t4_release_fraction/run_t4.py --execute --author-approved --workers 20
Calibration: <= 3 waves to the cell's target p (stop at 1 %), then a final wave if needed; valid if <= 2 %.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RHO_MAIN = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.89, 0.95)
RHO_SENS = (0.40, 0.60, 0.80, 0.89)
RHO_LONG = (0.30, 0.50, 0.70, 0.89)
TR_MAIN = (5.0, 20.0, 80.0)
TR_LONG = 400.0
SEEDS = (0, 1, 2, 3)
P_REF = 0.7042313040351903
K_REF, RHO_REF = 2.1339830, 0.8887143          # Phase-10 mean-release scale and rho at the reference point
TOL_STOP, TOL_VALID, MAX_CAL_WAVES = 0.01, 0.02, 3
CONFIG = CODE_ROOT / "config" / "phase4_causal.toml"
PHASE10 = CODE_ROOT / "model_sweep_tables" / "phase10-analysis" / "combo_summary.csv"



V3_ARCHIVED = ("T4 is archived under Revision Plan v3: its preregistration is void and it must not be executed. "
               "See revision_r1/README_zh.md.")

def row_targets(csv=PHASE10) -> dict:
    d = pd.read_csv(csv)
    out = {}
    for tr in TR_MAIN:
        r = d[np.isclose(d.tau_recovery_ms, tr)]
        out[tr] = {"main": float(r[np.isclose(r.tau_facilitation_ms, 1000.0)].full_p_match_theta.iloc[0]),
                   "lo": float(r.full_p_match_theta.min()), "hi": float(r.full_p_match_theta.max())}
    return out


def build_cells(targets: dict) -> list[dict]:
    cells = []
    for tr in TR_MAIN:
        cells += [{"rho": r, "tau_r": tr, "p_target": targets[tr]["main"], "family": "main"} for r in RHO_MAIN]
        for key in ("lo", "hi"):
            cells += [{"rho": r, "tau_r": tr, "p_target": targets[tr][key], "family": f"sens_{key}"} for r in RHO_SENS]
    cells += [{"rho": r, "tau_r": TR_LONG, "p_target": P_REF, "family": "long"} for r in RHO_LONG]
    for c in cells:
        c["cell_id"] = f"rho_{c['rho']:.2f}_tr_{c['tau_r']:g}_p_{c['p_target']:.4f}".replace(".", "p")
        c["k_start"] = K_REF * (RHO_REF / c["rho"]) * (c["p_target"] / P_REF)
    return cells


def _sim(task):
    cell, seed, wave, k, out, t_end = task
    from config import load_config
    from io_utils import load_result, save_result
    from model import simulate
    path = Path(out) / "cells" / cell["cell_id"] / f"wave{wave}" / f"seed_{seed:04d}.npz"
    if path.exists():
        res = load_result(path)
        if abs(res.temporal_gain_scale - k) > 1e-12 * max(1.0, abs(k)) or abs(res.resource_release_u_override - cell["rho"]) > 1e-12:
            raise SystemExit(f"PROVENANCE_CONFLICT: {path} was run with other parameters")
    else:
        cfg = load_config(CONFIG)
        cfg = replace(cfg, stf=replace(cfg.stf, tau_recovery_ms=float(cell["tau_r"])))
        res = simulate(cfg, condition="mean_u_release", seed=int(seed), noise_pa=0.0, temporal_gain_scale=float(k),
                       resource_release_u_override=float(cell["rho"]), end_ms=float(t_end))
        save_result(res, path)
    return cell["cell_id"], seed, wave, float(res.gain_match_numerator), float(res.gain_match_denominator), str(path)


def _final_metrics(paths, window=(1000.0, 8000.0)):
    from revision_r1.t2_efficacy_stats.metrics import psd_frequency_from_proxy
    from revision_r1.t2_efficacy_stats.reconstruct import load_trial
    from revision_r1.common.spikes import axial_center
    freqs, locs = [], []
    for p in paths:
        raw = load_trial(p)
        freqs.append(psd_frequency_from_proxy(raw, window[0], window[1]))
        ts, ang = np.asarray(raw["spike_times_e_ms"]), np.asarray(raw["spike_angles_e_deg"])
        _, r1 = axial_center(ang[(ts >= 1000) & (ts < 2000)])
        _, r2 = axial_center(ang[(ts >= 6500) & (ts < 7500)])
        locs.append(r2 / r1 if r1 > 0 else np.nan)
    return freqs, locs


def run(cells, out: Path, workers: int, t_end: float, seeds=SEEDS) -> pd.DataFrame:
    state = {c["cell_id"]: {"cell": c, "k": c["k_start"], "done": False, "log": []} for c in cells}
    for wave in range(1, MAX_CAL_WAVES + 2):
        pending = [s for s in state.values() if not s["done"]]
        if not pending:
            break
        tasks = [(s["cell"], seed, wave, s["k"], str(out), t_end) for s in pending for seed in seeds]
        res = {}
        if workers <= 1:
            for t in tasks:
                r = _sim(t); res.setdefault(r[0], []).append(r)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for f in as_completed([ex.submit(_sim, t) for t in tasks]):
                    r = f.result(); res.setdefault(r[0], []).append(r)
        for s in pending:
            rs = res[s["cell"]["cell_id"]]
            p = sum(r[3] for r in rs) / sum(r[4] for r in rs)
            err = abs(p - s["cell"]["p_target"]) / s["cell"]["p_target"]
            s["log"].append({"wave": wave, "k": s["k"], "p": p, "rel_err": err})
            if err <= TOL_STOP or wave == MAX_CAL_WAVES + 1:
                s["done"], s["final_wave"], s["final_paths"] = True, wave, sorted(r[5] for r in rs)
            else:
                s["k"] = s["k"] * s["cell"]["p_target"] / p
        print(f"wave {wave}: {sum(s['done'] for s in state.values())}/{len(state)} cells final", flush=True)
    rows = []
    for s in state.values():
        last = s["log"][-1]
        freqs, locs = _final_metrics(s["final_paths"]) if t_end >= 8000 else ([np.nan] * len(seeds), [np.nan] * len(seeds))
        fr = np.asarray(freqs, float)
        rows.append({**{k: s["cell"][k] for k in ("cell_id", "family", "rho", "tau_r", "p_target")},
                     "final_wave": s["final_wave"], "k_final": last["k"], "p_final": last["p"],
                     "rel_err": last["rel_err"], "calibration_valid": bool(last["rel_err"] <= TOL_VALID),
                     "freq_hz": float(np.nanmean(fr)), "freq_sd_hz": float(np.nanstd(fr)),
                     "theta_fraction": float(np.mean((fr >= 4) & (fr <= 12))),
                     "localization_min": float(np.nanmin(locs)) if np.isfinite(locs).any() else np.nan,
                     **{f"freq_seed{sd}": f for sd, f in zip(seeds, freqs)},
                     "waves": json.dumps(s["log"])})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(CODE_ROOT / "results" / "revision_r1" / "t4_release_fraction"))
    ap.add_argument("--workers", type=int, default=1)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--plan", action="store_true")
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--execute", action="store_true")
    ap.add_argument("--author-approved", action="store_true")
    a = ap.parse_args(argv)
    targets = row_targets()
    cells = build_cells(targets)
    if a.smoke:
        import tempfile
        c = [x for x in cells if x["family"] == "main" and np.isclose(x["rho"], 0.89) and x["tau_r"] == 20.0][0]
        df = run([c], Path(tempfile.mkdtemp()), 1, 1300.0, seeds=(0,))
        print(df.drop(columns=["waves"]).to_string()); print("SMOKE_OK")
        return 0
    print("row targets (full p_match from Phase 10):", json.dumps(targets))
    print(pd.DataFrame(cells)[["cell_id", "family", "rho", "tau_r", "p_target", "k_start"]].to_string())
    print(f"{len(cells)} cells x {len(SEEDS)} seeds; up to {MAX_CAL_WAVES + 1} waves of 8-s runs")
    if not a.execute:
        print("PLAN ONLY")
        return 0
    from revision_r1.common.manifest import write_manifest
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    write_manifest(out, task="T4_release_fraction", argv=sys.argv, code_root=CODE_ROOT,
                   extra={"targets": {str(k): v for k, v in targets.items()}, "n_cells": len(cells), "seeds": list(SEEDS)})
    df = run(cells, out, a.workers, 8000.0)
    df.to_csv(out / "cells.csv", index=False)
    print(df.drop(columns=["waves"]).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
