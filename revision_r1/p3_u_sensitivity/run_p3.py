"""P3 runner: U sensitivity within the continuous-gain model (PREREGISTRATION.md).  Reuses the Phase-17 machinery
(experiment/phase17_physiological.py: lock-step calibration, module-D restoration) with p3_config.toml.

From the code root:
  python revision_r1/p3_u_sensitivity/run_p3.py --plan
  python revision_r1/p3_u_sensitivity/run_p3.py --stage full --execute --author-approved --workers 20
  python revision_r1/p3_u_sensitivity/run_p3.py --stage routes --execute --author-approved --workers 20
  python revision_r1/p3_u_sensitivity/analyze_p3.py
  python revision_r1/p3_u_sensitivity/run_p3.py --stage restoration --execute --author-approved --workers 20
  python revision_r1/p3_u_sensitivity/analyze_p3.py

full         raw full model (frozen conductances) at every U, seeds 0-3; static control with p = that U's own p_match;
             interpretability screen; U = 0.4 anchor check against the frozen reference (FAIL stops the run)
routes       output-only, release-only, mean-release calibrated to that U's own full p_match (<= 3 waves), only at
             interpretable U
restoration  module-D protocol (+-11.25 deg activity-only shifts, F(2 s)) for full, output-only, release-only at
             [p3].restoration_u, only where interpretable with valid route calibrations
Every stage resumes: finished files are reused.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np  # noqa: E402

from experiment.phase17_physiological import (_run_parallel, _seed_paths, calibrate, combo_id,  # noqa: E402
                                              config_for, load_phase17_config, pooled_gain, pooled_release_u,
                                              read_manifest, run_restoration_items, write_manifest)
from revision_r1.p3_u_sensitivity.rules_p3 import anchor_check, interpretable, load_p3_section  # noqa: E402

MIN_PER_8S_RUN = 2.3


def _static_task(task):
    from io_utils import load_result, save_result
    from model import simulate
    cfg, seed, p, path = task
    path = Path(path)
    if path.exists():
        r = load_result(path)
        if r.static_gain is None or abs(r.static_gain - p) > 1e-12:
            raise RuntimeError(f"{path} exists with static_gain {r.static_gain}, planned {p}; refusing to overwrite")
        return str(path)
    r = simulate(cfg, condition="static_gain", seed=int(seed), noise_pa=0.0, static_gain=float(p))
    save_result(r, path)
    return str(path)


def _pool(fn, tasks, workers, label):
    if not tasks:
        return []
    out = []
    if workers <= 1:
        for k, t in enumerate(tasks, 1):
            out.append(fn(t)); print(f"[{label}] {k}/{len(tasks)}", flush=True)
        return out
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, f in enumerate(as_completed([ex.submit(fn, t) for t in tasks]), 1):
            out.append(f.result())
            if k % 5 == 0 or k == len(tasks):
                print(f"[{label}] {k}/{len(tasks)}", flush=True)
    return out


def trial_rows(cfg, theta_dir, seeds, smoke: bool = False) -> list[dict]:
    from io_utils import load_result
    from revision_r1.common.drive_rules import trial_metrics
    return [dict(seed=int(s), **trial_metrics(cfg, load_result(p), smoke=smoke))
            for s, p in zip(seeds, _seed_paths(theta_dir, seeds))]


def combos(cfg17):
    return [(combo_id(u, tf, tr), u, tf, tr) for (u, tf, tr) in cfg17.combos]


def log_run(out: Path, argv, stage):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), argv=list(argv), stage=stage,
                                python=platform.python_version(), numpy=np.__version__,
                                platform=platform.platform())) + "\n")


# ============================================================================== stages
def stage_full(cfg17, p3, workers):
    from config import load_config
    out = p3["output"]
    base = load_config(cfg17.base_config)
    seeds = cfg17.seeds
    tasks = []
    for cid, u, tf, tr in combos(cfg17):
        cfg = config_for(base, u, tf, tr)
        for p in _seed_paths(out / "combos" / cid / "raw" / "full" / "theta", seeds):
            tasks.append((cfg, "dynamic", int(p.stem.split("_")[1]), cfg17.theta_noise_pa, 1.0, None, str(p), True))
    contrib = _run_parallel(tasks, workers, "p3-full")
    static_tasks = []
    for cid, u, tf, tr in combos(cfg17):
        cfg = config_for(base, u, tf, tr)
        d = out / "combos" / cid / "raw" / "full" / "theta"
        paths = _seed_paths(d, seeds)
        p_match = pooled_gain(paths, contrib)
        rel_u = pooled_release_u(paths, contrib)
        rows = trial_rows(cfg, d, seeds, p3["smoke"])
        man = read_manifest(out, cid)
        man.update(combo_id=cid, u_baseline=u, tau_facilitation_ms=tf, tau_recovery_ms=tr)
        man.setdefault("raw", {})["full"] = dict(final_scale=1.0, theta_dir=str(d), final_p_match=p_match,
                                                 mean_release_u=rel_u, interpretability=interpretable(rows, p3),
                                                 per_seed=rows)
        write_manifest(out, cid, man)
        print(f"[p3-full] U={u:g}: f={man['raw']['full']['interpretability']['frequency_hz_mean']:.3f} Hz  "
              f"p_match={p_match:.5f}  u_rel={rel_u:.4f}  interpretable={man['raw']['full']['interpretability']['interpretable']}")
        for s, p in zip(seeds, _seed_paths(out / "combos" / cid / "raw" / "static" / "theta", seeds)):
            static_tasks.append((cfg, s, p_match, str(p)))
    _pool(_static_task, static_tasks, workers, "p3-static")
    # anchor check at the reference U
    ref = json.loads(Path(cfg17.anchor_reference).read_text(encoding="utf-8"))
    cid0 = combo_id(float(p3["reference_u"]), 1000.0, 20.0)
    f0 = read_manifest(out, cid0)["raw"]["full"]
    chk = anchor_check({r["seed"]: r["frequency_hz"] for r in f0["per_seed"]},
                       {r["seed"]: r["shift_1_8_deg"] for r in f0["per_seed"]},
                       f0["final_p_match"], f0["mean_release_u"], ref)
    (out / "anchor_check.json").write_text(json.dumps(chk, indent=2), encoding="utf-8")
    print(f"[p3-anchor] U = {p3['reference_u']}: {chk['status']}")
    if chk["status"] == "FAIL" and not p3["smoke"]:
        raise SystemExit("ANCHOR_FAIL: U = 0.4 does not reproduce the frozen reference; stop and report anchor_check.json")


def stage_routes(cfg17, p3, workers):
    from config import load_config
    out = p3["output"]
    if not (out / "anchor_check.json").exists():
        raise SystemExit("run --stage full first")
    if json.loads((out / "anchor_check.json").read_text(encoding="utf-8"))["status"] == "FAIL" and not p3["smoke"]:
        raise SystemExit("anchor check failed")
    base = load_config(cfg17.base_config)
    items = []
    for cid, u, tf, tr in combos(cfg17):
        man = read_manifest(out, cid)
        full = man["raw"]["full"]
        if not full["interpretability"]["interpretable"] and not p3["smoke"]:
            print(f"[p3-routes] U={u:g} not interpretable -> routes skipped (pre-registered)")
            continue
        cfg = config_for(base, u, tf, tr)
        for mode in cfg17.hybrid_modes:
            init = 1.0 if mode == "u_output_only" else float(full["mean_release_u"]) / u
            override = float(full["mean_release_u"]) if mode == "mean_u_release" else None
            items.append(dict(key=(cid, mode), cfg=cfg, condition=mode, target=float(full["final_p_match"]),
                              init_scale=init, override=override,
                              dir=out / "combos" / cid / "raw" / "routes" / mode))
    if not items:
        print("no interpretable U values: nothing to calibrate")
        return
    st = calibrate(items, cfg17, workers, True, "p3-routes")
    tasks = []
    for it in items:
        s = st[it["key"]]
        if not s["converged"]:
            for p in _seed_paths(Path(s["theta_dir"]), cfg17.seeds):
                tasks.append((it["cfg"], it["condition"], int(p.stem.split("_")[1]), cfg17.theta_noise_pa,
                              s["final_scale"], it["override"], str(p), True))
    _run_parallel(tasks, workers, "p3-routes-final")
    for it in items:
        cid, mode = it["key"]
        s = st[it["key"]]
        paths = _seed_paths(Path(s["theta_dir"]), cfg17.seeds)
        s["final_p_match"] = pooled_gain(paths)
        s["target"] = it["target"]
        s["final_relative_error"] = abs(s["final_p_match"] - it["target"]) / it["target"]
        s["valid"] = bool(s["final_relative_error"] <= cfg17.calibration_validity)
        s["mean_release_u"] = pooled_release_u(paths)
        s["override"] = it["override"]
        man = read_manifest(out, cid)
        man["raw"].setdefault("routes", {})[mode] = {k: (str(v) if isinstance(v, Path) else v) for k, v in s.items()}
        write_manifest(out, cid, man)
        print(f"[p3-routes] {cid} {mode}: k = {s['final_scale']:.5f}  rel.err {s['final_relative_error']:.4f}  "
              f"valid {s['valid']}")


def stage_restoration(cfg17, p3, workers):
    from config import load_config
    out = p3["output"]
    base = load_config(cfg17.base_config)
    items = []
    for u in p3["restoration_u"]:
        cid = combo_id(float(u), 1000.0, 20.0)
        man = read_manifest(out, cid)
        if not man or (not man["raw"]["full"]["interpretability"]["interpretable"] and not p3["smoke"]):
            print(f"[p3-restoration] U={u}: not interpretable -> skipped (pre-registered)")
            continue
        routes = man["raw"].get("routes", {})
        need = [c for c in cfg17.restoration["conditions"] if c != "dynamic"]
        if not all(c in routes and (routes[c].get("valid") or p3["smoke"]) for c in need):
            print(f"[p3-restoration] U={u}: route calibration missing or invalid -> skipped (pre-registered)")
            continue
        cfg = config_for(base, float(u), 1000.0, 20.0)
        for cond in cfg17.restoration["conditions"]:
            scale = 1.0 if cond == "dynamic" else float(routes[cond]["final_scale"])
            items.append((cid, "raw", cond, cfg, scale))
    if items:
        run_restoration_items(cfg17, out, items, workers, root=out / "restoration")


def print_plan(cfg17, p3):
    n_u, s, w = len(cfg17.combos), len(cfg17.seeds), cfg17.max_calibration_waves
    m = len(cfg17.hybrid_modes)
    full, static = n_u * s, n_u * s
    routes_lo, routes_hi = n_u * m * s, n_u * m * (w * s + s)
    R = cfg17.restoration
    nr, nc = len(p3["restoration_u"]), len(R["conditions"])
    ck = nr * nc * len(R["seeds"])
    br = nr * nc * len(R["seeds"]) * (1 + 2 * len(R["deltas_deg"]))
    eq_rest = ck * 3.15 / 8 + br * float(R["continuation_ms"]) / 8000.0
    print(f"P3 U sensitivity: U = {list(cfg17.u_baseline)}, tau_f = 1000 ms, tau_r = 20 ms, seeds {list(cfg17.seeds)}")
    print(f"  full (raw) {full} + static {static} runs; routes {routes_lo}-{routes_hi} runs (interpretable U only)")
    print(f"  CPU: full+static {(full + static) * MIN_PER_8S_RUN / 60:.1f} h; routes "
          f"{routes_lo * MIN_PER_8S_RUN / 60:.1f}-{routes_hi * MIN_PER_8S_RUN / 60:.1f} h; restoration "
          f"{eq_rest * MIN_PER_8S_RUN / 60:.1f} h ({ck} checkpoints + {br} branches) at {MIN_PER_8S_RUN} min per 8-s run")
    print(f"  output: {p3['output']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).with_name("p3_config.toml")))
    ap.add_argument("--stage", choices=["full", "routes", "restoration"])
    ap.add_argument("--workers", type=int, default=1)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--plan", action="store_true")
    g.add_argument("--execute", action="store_true")
    ap.add_argument("--author-approved", action="store_true")
    a = ap.parse_args(argv)
    cfg17 = load_phase17_config(a.config)
    p3 = load_p3_section(a.config)
    if a.plan or not a.execute:
        print_plan(cfg17, p3)
        return 0
    if not a.stage:
        raise SystemExit("--execute needs --stage full, routes or restoration")
    log_run(p3["output"], sys.argv, a.stage)
    {"full": stage_full, "routes": stage_routes, "restoration": stage_restoration}[a.stage](cfg17, p3, a.workers)
    print("done:", a.stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
