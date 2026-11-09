"""P2b runner: C3 (total-drive) matched controls in the continuous-gain convention (PREREGISTRATION.md).

From the code root (the folder that contains model/, config/, revision_r1/):

  python revision_r1/run_tests.py
  python revision_r1/p2b_drive_matched/run_p2b.py --plan
  python revision_r1/p2b_drive_matched/run_p2b.py --smoke --workers 4
  python revision_r1/p2b_drive_matched/run_p2b.py --execute --author-approved --workers 24
  python revision_r1/p2b_drive_matched/analyze_p2b.py

Steps of --execute:
  1  D_target from the existing continuous full runs (target_dir, seeds 0-3), with the P1b QC on p_match
  2  check the existing full runs of the frequency seeds (full_frequency_dir); the full model is never re-run
  3  wave 0 of output-only and release-only = the existing C1 runs (their C3 drive is read exactly from the NPZ)
  4  lock-step calibration of k (output-only, release-only) and p (static) to D_target, <= max_waves waves
  5  all frequency seeds at the final values
Re-running the command resumes: finished files are reused; a finished file made with other parameters stops the run.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import sys
from dataclasses import replace
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np  # noqa: E402

from revision_r1.common.drive_rules import _pool, _sim_task  # noqa: E402
from revision_r1.p2b_drive_matched.rules_p2b import (  # noqa: E402
    CAL_MODELS, MODELS, calibrate_core, check_existing_run, drive_of_paths,
    load_p2b_config, run_kwargs, seed_path)

MIN_PER_8S_RUN = 2.3      # single-core minutes per 8-s trial on the author's machine (for --plan only)
P_MATCH_REFERENCE = 0.7042313040351903   # pooled p_match of the target runs (P1b QC2, anchor_reference)


# ============================================================================== state and manifest
def write_state(out: Path, st: dict) -> None:
    p = Path(out) / "p2b_stage.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(st, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def read_state(out: Path) -> dict:
    p = Path(out) / "p2b_stage.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def write_manifest(out: Path, argv, extra: dict) -> None:
    import scipy
    Path(out).mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    m = dict(task="P2b_drive_matched_continuous", utc=stamp, argv=list(argv), python=sys.version.split()[0],
             numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(), **extra)
    (Path(out) / f"run_manifest_{stamp}.json").write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")


# ============================================================================== smoke settings
def smoke_settings(cfg, pc, duration_ms: float):
    """Short trials with analysis windows scaled to the duration (plumbing check only; no verdicts)."""
    D = float(duration_ms)
    cue_end = cfg.simulation.cue_end_ms
    if D < cue_end + 200.0:
        raise SystemExit(f"--smoke-ms must be at least cue end + 200 ms = {cue_end + 200.0:g} ms")
    dt = cfg.simulation.dt_ms
    grid = lambda x: round(round(x / dt) * dt, 6)  # noqa: E731
    sim = replace(cfg.simulation, duration_ms=D)
    ana = replace(cfg.analysis, theta_start_ms=grid(D / 3), theta_end_ms=D, gain_match_start_ms=grid(D / 3),
                  gain_match_end_ms=grid(0.9 * D), localization_reference_start_ms=grid(D / 3),
                  localization_reference_end_ms=grid(D / 2), localization_target_start_ms=grid(0.8 * D),
                  localization_target_end_ms=D, pav_target_time_ms=grid(0.9 * D))
    cfg_s = replace(cfg, simulation=sim, analysis=ana)
    out = pc.output / "smoke"
    pc_s = replace(pc, output=out, target_dir=out / "continuous_full", full_frequency_dir=out / "continuous_full",
                   wave0_dirs={m: out / "wave0" / m for m in ("output_only", "release_only")},
                   seeds_calibration=(0,), seeds_frequency=(0, 1), calibration=dict(pc.calibration, max_waves=1))
    return cfg_s, pc_s


# ============================================================================== steps
def compute_target(pc, cfg, workers: int, smoke: bool) -> dict:
    from io_utils import load_result
    if smoke:                                      # fresh short continuous full runs (target and frequency seeds)
        tasks = [(cfg, dict(convention="continuous", condition="dynamic"), s, seed_path(pc.target_dir, s),
                  {"task": "P2b", "model": "full", "stage": "smoke_target"}) for s in pc.seeds_frequency]
        _pool(_sim_task, tasks, workers, "smoke-full")
    paths = [seed_path(pc.target_dir, s) for s in pc.seeds_calibration]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"MISSING_TARGET_RUNS: set [paths].target_dir in p2b_config.toml: {missing}")
    rs = [load_result(p) for p in paths]
    for r, s in zip(rs, pc.seeds_calibration):
        check_existing_run(r, "dynamic", s, k_expected=1.0, rel_tol=1e-12)
    D = drive_of_paths(paths, cfg)
    p_match = sum(r.gain_match_numerator for r in rs) / sum(r.gain_match_denominator for r in rs)
    if not smoke and abs(p_match - P_MATCH_REFERENCE) > 1e-6:
        raise SystemExit(f"QC2: pooled p_match of the target runs {p_match:.9f} != {P_MATCH_REFERENCE:.9f}")
    return dict(D_target=float(D), pooled_p_match=float(p_match), files=[str(p) for p in paths])


def check_full_frequency(pc) -> dict:
    from io_utils import load_result
    paths = {str(s): str(seed_path(pc.full_frequency_dir, s)) for s in pc.seeds_frequency}
    missing = [p for p in paths.values() if not Path(p).exists()]
    if missing:
        raise SystemExit(f"MISSING_FULL_RUNS: set [paths].full_frequency_dir in p2b_config.toml: {missing}")
    for s, p in paths.items():
        check_existing_run(load_result(p), "dynamic", int(s), k_expected=1.0, rel_tol=1e-12)
    return paths


def wave0_history(pc, cfg, workers: int, smoke: bool) -> dict:
    """Wave 0 of the two route models: their existing C1 runs (seeds = calibration seeds)."""
    from io_utils import load_result
    out = {}
    for model, key in (("output_only", "output_only_k_c1"), ("release_only", "release_only_k_c1")):
        cond = MODELS[model][0]
        d = pc.wave0_dirs[model]
        k_cfg = float(pc.initial[key])
        if smoke:
            tasks = [(cfg, run_kwargs(model, k_cfg), s, seed_path(d, s), {"task": "P2b", "model": model,
                                                                          "stage": "smoke_wave0"})
                     for s in pc.seeds_calibration]
            _pool(_sim_task, tasks, workers, f"smoke-wave0-{model}")
        paths = [seed_path(d, s) for s in pc.seeds_calibration]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"MISSING_WAVE0_RUNS for {model}: {missing}")
        ks = []
        for p, s in zip(paths, pc.seeds_calibration):
            r = load_result(p)
            check_existing_run(r, cond, s, k_expected=k_cfg, rel_tol=1e-6)
            ks.append(float(r.temporal_gain_scale))
        if max(ks) - min(ks) > 1e-12 * max(ks):
            raise SystemExit(f"wave-0 files of {model} use different k: {ks}")
        D = drive_of_paths(paths, cfg)
        out[model] = dict(wave=0, param=ks[0], D=float(D), dir=str(d))
    return out


def make_wave_runner(pc, cfg, workers: int, label: str):
    def run_wave(todo: list[dict], wave: int):
        tasks, dirs = [], []
        for it in todo:
            wdir = Path(it["dir"]) / f"wave{wave}"
            kw = run_kwargs(it["model"], it["param"])
            for s in pc.seeds_calibration:
                tasks.append((cfg, kw, s, seed_path(wdir, s), {"task": "P2b", "model": it["model"],
                                                               "stage": f"wave{wave}"}))
            dirs.append(wdir)
        _pool(_sim_task, tasks, workers, f"{label}-wave{wave}")
        res = []
        for it, wdir in zip(todo, dirs):
            D = drive_of_paths([seed_path(wdir, s) for s in pc.seeds_calibration], cfg)
            rel = abs(D - it["target"]) / it["target"]
            print(f"[{label}] {it['model']} wave {wave}: param {it['param']:.7g}  D {D:.6g}  target "
                  f"{it['target']:.6g}  rel.err {rel:.4f}", flush=True)
            res.append((D, wdir))
        return res
    return run_wave


def finals(items: list[dict], pc, cfg, workers: int) -> list[dict]:
    tasks = []
    for it in items:
        kw = run_kwargs(it["model"], it["final_param"])
        it["final_kw"] = kw
        it["final_paths"] = {}
        for s in pc.seeds_frequency:
            if it["converged_dir"] and s in pc.seeds_calibration:
                p = seed_path(it["converged_dir"], s)
            else:
                p = seed_path(Path(it["dir"]) / "final", s)
                tasks.append((cfg, kw, s, p, {"task": "P2b", "model": it["model"], "stage": "final"}))
            it["final_paths"][str(s)] = str(p)
    _pool(_sim_task, tasks, workers, "finals")
    for it in items:
        D = drive_of_paths([it["final_paths"][str(s)] for s in pc.seeds_calibration], cfg)
        it["final_D_calibration_seeds"] = float(D)
        it["final_rel_error"] = float(abs(D - it["target"]) / it["target"])
        it["valid"] = bool(it["final_rel_error"] <= float(pc.calibration["validity"]))
    return items


def stage_all(pc, cfg, workers: int, smoke: bool = False) -> dict:
    out = pc.output
    st = read_state(out)
    st["config"] = dict(seeds_calibration=pc.seeds_calibration, seeds_frequency=pc.seeds_frequency,
                        initial=pc.initial, calibration=pc.calibration, decision=pc.decision, smoke=smoke)
    st["target"] = compute_target(pc, cfg, workers, smoke)
    D_t = st["target"]["D_target"]
    print(f"[target] D_target = {D_t:.6g}  (pooled p_match {st['target']['pooled_p_match']:.7f})", flush=True)
    st["full_frequency_paths"] = check_full_frequency(pc)
    w0 = wave0_history(pc, cfg, workers, smoke)
    for m, h in w0.items():
        h["rel_error"] = abs(h["D"] - D_t) / D_t
        print(f"[wave0] {m}: k {h['param']:.7g}  D {h['D']:.6g}  ({h['D'] / D_t:.3f} x target)", flush=True)
    st["wave0"] = w0
    write_state(out, st)
    items = []
    for m in CAL_MODELS:
        it = dict(model=m, target=D_t, dir=str(out / "matched" / m))
        if m in w0:
            it.update(param=w0[m]["param"], history=[dict(w0[m])])
        else:
            it.update(param=float(pc.initial["static_p0"]), history=[])
        items.append(it)
    items = calibrate_core(items, make_wave_runner(pc, cfg, workers, "cal"), pc.calibration)
    st["models"] = items
    write_state(out, st)
    items = finals(items, pc, cfg, workers)
    st["models"] = items
    st["status"] = "P2B_COMPLETE"
    write_state(out, st)
    for it in items:
        print(f"[{it['model']}] param* = {it['final_param']:.7g}  rel.err {it['final_rel_error']:.4f}  "
              f"valid {it['valid']}", flush=True)
    print("P2b runs complete. Next: python revision_r1/p2b_drive_matched/analyze_p2b.py"
          + (" --smoke" if smoke else ""))
    return st


def print_plan(pc) -> None:
    nc, nf = len(pc.seeds_calibration), len(pc.seeds_frequency)
    w = int(pc.calibration["max_waves"])
    n_models = len(CAL_MODELS)
    lo = n_models * nc + n_models * (nf - nc)                # one wave each, converged
    hi = n_models * w * nc + n_models * nf                   # all waves, finals on every frequency seed
    print("P2b plan (8-s, 0-pA runs at the reference point, continuous-gain convention)")
    print(f"  target      : existing files in {pc.target_dir} (seeds {pc.seeds_calibration}); no new runs")
    print(f"  full model  : existing files in {pc.full_frequency_dir} (seeds {pc.seeds_frequency}); no new runs")
    print(f"  wave 0      : existing C1 runs of output-only and release-only; no new runs")
    print(f"  calibration : output-only k, release-only k, static p (start {pc.initial['static_p0']}), "
          f"<= {w} waves x {nc} seeds")
    print(f"  runs        : {lo}-{hi} new 8-s runs, {lo * MIN_PER_8S_RUN / 60:.1f}-{hi * MIN_PER_8S_RUN / 60:.1f} "
          f"core-hours; with >= {n_models * nc} workers about {(w + 1) * MIN_PER_8S_RUN:.0f}-"
          f"{(w + 2) * MIN_PER_8S_RUN:.0f} min wall time")
    print(f"  output      : {pc.output}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="P2b TOML (default: p2b_config.toml next to this file)")
    ap.add_argument("--workers", type=int, default=1)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--plan", action="store_true")
    g.add_argument("--smoke", action="store_true", help="short trials in a separate folder; no verdicts")
    g.add_argument("--execute", action="store_true")
    ap.add_argument("--author-approved", action="store_true")
    ap.add_argument("--smoke-ms", type=float, default=1200.0)
    a = ap.parse_args(argv)
    from config import load_config
    pc = load_p2b_config(a.config)
    cfg = load_config(pc.base_config)
    if (cfg.stf.u_baseline, cfg.stf.tau_facilitation_ms, cfg.stf.tau_recovery_ms) != (0.4, 1000.0, 20.0):
        raise SystemExit("base config is not the reference point")
    if a.plan or not (a.smoke or a.execute):
        print_plan(pc)
        return 0
    if a.smoke:
        cfg_s, pc_s = smoke_settings(cfg, pc, a.smoke_ms)
        write_manifest(pc_s.output, sys.argv, {"mode": "smoke", "smoke_ms": a.smoke_ms})
        stage_all(pc_s, cfg_s, a.workers, smoke=True)
        print(f"SMOKE_DONE: {pc_s.output}  (plumbing check only; numbers are not results)")
        return 0
    write_manifest(pc.output, sys.argv, {"workers": a.workers})
    stage_all(pc, cfg, a.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
