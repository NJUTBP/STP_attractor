#!/usr/bin/env python3
"""Run Phase-16 experiments (reduced-model tests of the spatial mechanism).

Modules
  A  u-clamp force field: activity-only shifts over a grid of offsets, normal versus clamped u, 600 ms.
  B  well kinematics: activity-only (and, for the full model, activity+u / activity+R) shifts with
     10-ms profiles of u, R and s_NMDA, 2.5 s.
  C  relative versus joint wandering: noisy continuations (30, 60 pA) of full and static models with
     20-ms profiles, 10 s.
  D  timescale scaling: shifts at (tf, tr) = (1000,5), (1000,20), (1000,80), (250,20), (2500,20);
     output-only scale factors are read from the Phase-10 manifests.

Example
  python phase16/run_phase16.py --module A --output results/phase16 --workers 8 \
      --reuse-checkpoints results/phase15_tether_confirmation
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402
from experiment.phase11_pilot import checkpoint_ms_for_seed, checkpoint_path, mode_kwargs  # noqa: E402
from model.simulator import run_segment  # noqa: E402
from model.state import aligned_roll_bins, apply_named_shift, load_checkpoint, save_checkpoint  # noqa: E402
from phase16.snapshots import run_with_snapshots, with_duration  # noqa: E402

SEEDS = tuple(range(200, 216))
GRID_A = (2.8125, 5.625, 11.25, 16.875, 22.5, 33.75, 45.0)
ROUTES = ("dynamic", "u_output_only", "u_release_only")
COMBOS_D = ((1000.0, 5.0), (1000.0, 20.0), (1000.0, 80.0), (250.0, 20.0), (2500.0, 20.0))


def cid(tf, tr):
    return f"tf_{tf:g}_tr_{tr:g}"


def scale_kwargs(cond, tf, tr, phase10_root):
    if cond == "facilitation_only":
        return {"temporal_gain_scale": 1.0, "resource_release_u_override": None, "static_gain": None}
    if (tf, tr) == (1000.0, 20.0) or cond in ("dynamic", "static_gain"):
        if cond == "static_gain" and (tf, tr) != (1000.0, 20.0):
            raise ValueError("static control is only calibrated at the anchor")
        return mode_kwargs(cond)
    if phase10_root is None:
        raise ValueError("--phase10-root is required for route controls away from the anchor")
    man = json.loads((Path(phase10_root) / "combos" / cid(tf, tr) / "manifest.json").read_text())
    return {"temporal_gain_scale": float(man["modes"][cond]["final_scale"]),
            "resource_release_u_override": None, "static_gain": None}


def build_specs(args):
    seeds = args.seeds if args.seeds is not None else (SEEDS[:8] if args.module == "D" else SEEDS)
    grid = args.deltas if args.deltas is not None else None
    specs = []
    def add(module, cond, seed, kind, delta, T, snap, mode="normal", noise=0.0, draw=None, tf=1000.0, tr=20.0):
        specs.append(dict(module=module, condition=cond, seed=int(seed), kind=kind, delta=float(delta), T=float(T),
                          snap=snap, mode=mode, noise=float(noise), draw=draw, tf=tf, tr=tr))
    for seed in seeds:
        if args.module == "A":
            for cond in ROUTES:
                for mode in ("normal", "clamp"):
                    add("A", cond, seed, "no_shift", 0.0, 600.0, None, mode)
                    for d in (grid or GRID_A):
                        for s in (1, -1):
                            add("A", cond, seed, "activity_shift", s * d, 600.0, None, mode)
        elif args.module == "B":
            conds = ROUTES + (("facilitation_only",) if args.include_facilitation_only else ())
            for cond in conds:
                add("B", cond, seed, "no_shift", 0.0, 2500.0, 10.0)
                for d in (grid or (11.25, 22.5)):
                    for s in (1, -1):
                        add("B", cond, seed, "activity_shift", s * d, 2500.0, 10.0)
            for kind in ("activity_u_shift", "activity_r_shift"):
                for d in (grid or (11.25, 22.5)):
                    for s in (1, -1):
                        add("B", "dynamic", seed, kind, s * d, 2500.0, 10.0)
        elif args.module == "C":
            for cond in ("dynamic", "static_gain"):
                for noise in (30.0, 60.0):
                    for draw in (0, 1):
                        add("C", cond, seed, "no_shift", 0.0, 10000.0, 20.0, noise=noise, draw=draw)
        elif args.module == "D":
            for tf, tr in COMBOS_D:
                for cond in ("dynamic", "u_output_only"):
                    add("D", cond, seed, "no_shift", 0.0, 2500.0, None, tf=tf, tr=tr)
                    for d in (grid or (11.25, 22.5)):
                        for s in (1, -1):
                            add("D", cond, seed, "activity_shift", s * d, 2500.0, None, tf=tf, tr=tr)
                for kind in ("activity_u_shift", "activity_r_shift"):
                    for d in (grid or (11.25, 22.5)):
                        for s in (1, -1):
                            add("D", "dynamic", seed, kind, s * d, 2500.0, None, tf=tf, tr=tr)
    if args.t_ms is not None:
        for sp in specs:
            sp["T"] = float(args.t_ms)
    return specs


def branch_id(sp):
    d = sp["delta"]; sign = "p" if d >= 0 else "m"
    tag = f"{sp['kind']}__{sign}{abs(d):g}".replace(".", "p") if sp["kind"] != "no_shift" else "no_shift"
    extra = f"__{sp['mode']}" if sp["module"] == "A" else ""
    if sp["module"] == "C":
        extra = f"__noise{sp['noise']:g}__draw{sp['draw']}"
    return f"{cid(sp['tf'], sp['tr'])}__{sp['condition']}__seed{sp['seed']:04d}__{tag}{extra}"


def cfg_for(args, sp):
    cfg = load_config(args.config)
    if (sp["tf"], sp["tr"]) != (cfg.stf.tau_facilitation_ms, cfg.stf.tau_recovery_ms):
        cfg = replace(cfg, stf=replace(cfg.stf, tau_facilitation_ms=sp["tf"], tau_recovery_ms=sp["tr"]))
    return cfg


def t0_for(args, seed):
    return float(args.t0_ms) if args.t0_ms is not None else checkpoint_ms_for_seed(seed)


def ckpt_for(args, sp):
    if sp["module"] != "D":
        for r in args.reuse_checkpoints:
            p = checkpoint_path(Path(r), sp["condition"], sp["seed"])
            if p.exists():
                return p
    return Path(args.output) / "checkpoints" / cid(sp["tf"], sp["tr"]) / sp["condition"] / f"seed_{sp['seed']:04d}.pkl"


def _make_checkpoint(payload):
    args, sp, path = payload
    path = Path(path)
    if path.exists():
        return str(path)
    cfg = cfg_for(args, sp)
    t0 = t0_for(args, sp["seed"])
    kw = scale_kwargs(sp["condition"], sp["tf"], sp["tr"], args.phase10_root)
    _res, state = run_segment(cfg, condition=sp["condition"], seed=sp["seed"], noise_pa=0.0, end_ms=t0, **kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(state, path)
    return str(path)


def _run_branch(payload):
    args, sp, ckpt, out = payload
    out = Path(out)
    if out.exists() and not args.overwrite:
        return str(out)
    cfg = cfg_for(args, sp)
    state = load_checkpoint(ckpt)
    t0 = state.step * cfg.simulation.dt_ms
    if abs(t0 - t0_for(args, sp["seed"])) > 1e-6:
        raise RuntimeError(f"checkpoint time {t0} does not match protocol time for seed {sp['seed']}")
    if sp["kind"] == "no_shift":
        br = state.copy()
    else:
        e, i = aligned_roll_bins(sp["delta"], cfg.network.ne, cfg.network.ni)
        br = apply_named_shift(state, sp["kind"], e, i)
    kw = scale_kwargs(sp["condition"], sp["tf"], sp["tr"], args.phase10_root)
    res, _state, snaps = run_with_snapshots(
        cfg, condition=sp["condition"], seed=sp["seed"], noise_pa=sp["noise"], start_state=br,
        end_ms=t0 + sp["T"], every_ms=sp["snap"], clamp_u=(sp["mode"] == "clamp"), noise_draw=sp["draw"], **kw)
    m = res.spike_times_e_ms > t0
    payload = dict(spike_t=(res.spike_times_e_ms[m] - t0).astype(np.float32),
                   spike_a=res.spike_angles_e_deg[m].astype(np.float32),
                   theta_e=res.theta_e_deg.astype(np.float32),
                   meta=json.dumps({**sp, "t0_ms": t0, "scale_kwargs": kw}))
    if snaps is not None:
        payload.update(snap_t=(snaps["t"] - t0).astype(np.float32), snap_s=snaps["s"])
        if "u" in snaps:
            payload.update(snap_u=snaps["u"], snap_R=snaps["R"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    return str(out)


def _pool(fn, items, workers, label):
    if workers <= 1:
        for k, it in enumerate(items, 1):
            fn(it)
            print(f"[{label}] {k}/{len(items)}", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, it) for it in items]
        for k, f in enumerate(as_completed(futs), 1):
            f.result()
            if k % 10 == 0 or k == len(items):
                print(f"[{label}] {k}/{len(items)}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--module", choices=list("ABCD"), required=True)
    p.add_argument("--config", default=str(ROOT / "config" / "phase4_causal.toml"))
    p.add_argument("--output", default=str(ROOT / "results" / "phase16"))
    p.add_argument("--seeds", type=lambda s: tuple(int(x) for x in s.split(",")), default=None,
                   help="default: 200-215 (modules A-C), 200-207 (module D)")
    p.add_argument("--deltas", type=lambda s: tuple(float(x) for x in s.split(",")), default=None,
                   help="override the shift magnitudes (smoke tests)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--reuse-checkpoints", nargs="*", default=[])
    p.add_argument("--phase10-root", default=None)
    p.add_argument("--include-facilitation-only", action="store_true")
    p.add_argument("--t0-ms", type=float, default=None, help="override checkpoint time (smoke tests only)")
    p.add_argument("--t-ms", type=float, default=None, help="override continuation length (smoke tests only)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    specs = build_specs(args)
    root = Path(args.output) / f"module_{args.module}"
    ck = {}
    for sp in specs:
        ck.setdefault(str(ckpt_for(args, sp)), sp)
    print(f"module {args.module}: {len(specs)} branches, {len(ck)} checkpoints", flush=True)
    if args.dry_run:
        return
    _pool(_make_checkpoint, [(args, sp, path) for path, sp in ck.items()], args.workers, "checkpoints")
    tasks = [(args, sp, str(ckpt_for(args, sp)), str(root / "branches" / f"{branch_id(sp)}.npz")) for sp in specs]
    _pool(_run_branch, tasks, args.workers, f"module {args.module}")
    (root / "run_manifest.json").write_text(json.dumps({"config": args.config, "seeds": sorted({sp["seed"] for sp in specs}),
                                                       "n_branches": len(specs)}, indent=2))


if __name__ == "__main__":
    main()
