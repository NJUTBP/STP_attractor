from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np

from config.phase8 import Phase8SweepConfig
from config.schema import ProjectConfig
from io_utils import load_result, save_result, write_json
from model import simulate

TEMPORAL_MODES = ("dynamic", "facilitation_only", "depletion_only")


def combo_id(tau_f_ms: float, tau_r_ms: float) -> str:
    return f"tf_{tau_f_ms:g}_tr_{tau_r_ms:g}".replace(".", "p")


def config_for_combo(base_cfg: ProjectConfig, tau_f_ms: float, tau_r_ms: float) -> ProjectConfig:
    return replace(
        base_cfg,
        stf=replace(
            base_cfg.stf,
            tau_facilitation_ms=float(tau_f_ms),
            tau_recovery_ms=float(tau_r_ms),
        ),
    )


def _result_path(root: Path, cid: str, regime: str, label: str, seed: int) -> Path:
    return root / "combos" / cid / regime / label / f"seed_{seed:04d}.npz"


def _baseline_path(root: Path, regime: str, seed: int) -> Path:
    return root / "baseline" / regime / f"seed_{seed:04d}.npz"


def _simulate_task(
    cfg: ProjectConfig,
    condition: str,
    seed: int,
    noise_pa: float,
    static_gain: float | None,
    frozen_profile: np.ndarray | None,
    path: str,
    resume: bool,
) -> tuple[str, float, float]:
    out = Path(path)
    if resume and out.exists():
        old = load_result(out)
        return path, old.gain_match_numerator, old.gain_match_denominator
    result = simulate(
        cfg,
        condition=condition,
        seed=int(seed),
        noise_pa=float(noise_pa),
        static_gain=static_gain,
        frozen_profile=frozen_profile,
    )
    save_result(result, out)
    return path, result.gain_match_numerator, result.gain_match_denominator


def _run_parallel(tasks: list[tuple], workers: int, label: str) -> dict[str, tuple[float, float]]:
    outputs: dict[str, tuple[float, float]] = {}
    total = len(tasks)
    if total == 0:
        return outputs
    if workers <= 1:
        for i, task in enumerate(tasks, 1):
            path, num, den = _simulate_task(*task)
            outputs[path] = (num, den)
            if i % 25 == 0 or i == total:
                print(f"[{label}] {i}/{total}")
        return outputs
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_simulate_task, *task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            path, num, den = future.result()
            outputs[path] = (num, den)
            if i % 50 == 0 or i == total:
                print(f"[{label}] {i}/{total}")
    return outputs


def _pooled_match(paths: Iterable[Path], contributions: dict[str, tuple[float, float]]) -> float:
    num = 0.0
    den = 0.0
    for path in paths:
        key = str(path)
        if key in contributions:
            n, d = contributions[key]
        else:
            r = load_result(path)
            n, d = r.gain_match_numerator, r.gain_match_denominator
        num += n
        den += d
    if den <= 0:
        raise RuntimeError("Cannot define pooled p_match: zero denominator.")
    return num / den


def _frozen_profile(paths: Iterable[Path]) -> np.ndarray:
    profiles = [load_result(path).mean_efficacy_by_e for path in paths]
    if not profiles:
        raise ValueError("No full-dynamic theta runs available for frozen profile.")
    profile = np.mean(np.stack(profiles), axis=0)
    if np.any(~np.isfinite(profile)) or np.any(profile < 0) or np.any(profile > 1):
        raise ValueError("Invalid frozen spatial profile.")
    return profile


def run_phase8_mechanism_map(
    base_cfg: ProjectConfig,
    sweep: Phase8SweepConfig,
    *,
    output_dir: str | Path,
    workers: int = 1,
    resume: bool = True,
) -> dict:
    """Run the full Phase-8 batch in two dependency-aware waves.

    Wave 1 runs every temporal mode across the whole tau_f x tau_recovery grid at
    both 0 and 60 pA.  Pooled p_match values are then computed per
    combo/mode/regime.  Wave 2 runs every matched-static control and the
    full-dynamic frozen-spatial theta control.  Baseline is shared across the
    entire grid and is run only once per seed/regime.
    """
    base_cfg.validate()
    sweep.validate()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(s) for s in sweep.seeds)
    modes = tuple(sweep.temporal_modes)

    grid = []
    for tau_f, tau_r in sweep.combos:
        cid = combo_id(tau_f, tau_r)
        grid.append((cid, float(tau_f), float(tau_r), config_for_combo(base_cfg, tau_f, tau_r)))

    global_manifest = {
        "experiment": "phase8_stp_time_scale_mechanism_map",
        "scientific_question": (
            "Which E->I STP temporal component and time scale creates theta, and does the same temporal mechanism "
            "increase resistance to 60-pA perturbations beyond matched mean E->I gain?"
        ),
        "u_baseline_fixed": base_cfg.stf.u_baseline,
        "tau_facilitation_ms": list(sweep.tau_facilitation_ms),
        "tau_recovery_ms": list(sweep.tau_recovery_ms),
        "seeds": list(seeds),
        "temporal_modes": list(modes),
        "theta_noise_pa": sweep.theta_noise_pa,
        "attractor_noise_pa": sweep.attractor_noise_pa,
        "run_frozen_profile": sweep.run_frozen_profile,
        "combo_count": len(grid),
        "expected_run_count": sweep.expected_run_count(),
        "resume": bool(resume),
    }
    write_json(global_manifest, root / "experiment_manifest.json")

    # Shared baseline runs (tau values do not affect a non-plastic E->I synapse).
    baseline_tasks = []
    for regime, noise in (("theta", sweep.theta_noise_pa), ("attractor", sweep.attractor_noise_pa)):
        for seed in seeds:
            baseline_tasks.append((
                base_cfg, "baseline", seed, float(noise), None, None,
                str(_baseline_path(root, regime, seed)), resume,
            ))
    _run_parallel(baseline_tasks, workers, "baseline")

    # Wave 1: all temporal modes, all parameter combinations, both regimes.
    temporal_tasks = []
    for cid, tau_f, tau_r, cfg in grid:
        for regime, noise in (("theta", sweep.theta_noise_pa), ("attractor", sweep.attractor_noise_pa)):
            for mode in modes:
                for seed in seeds:
                    temporal_tasks.append((
                        cfg, mode, seed, float(noise), None, None,
                        str(_result_path(root, cid, regime, mode, seed)), resume,
                    ))
    contributions = _run_parallel(temporal_tasks, workers, "temporal")

    # Per-combination matching and frozen profiles.
    combo_meta: dict[str, dict] = {}
    frozen_profiles: dict[str, np.ndarray] = {}
    for cid, tau_f, tau_r, cfg in grid:
        matches: dict[str, dict[str, float]] = {"theta": {}, "attractor": {}}
        for regime in ("theta", "attractor"):
            for mode in modes:
                paths = [_result_path(root, cid, regime, mode, seed) for seed in seeds]
                matches[regime][mode] = _pooled_match(paths, contributions)
        meta = {
            "combo_id": cid,
            "tau_facilitation_ms": tau_f,
            "tau_recovery_ms": tau_r,
            "p_match": matches,
            "seeds": list(seeds),
            "temporal_modes": list(modes),
        }
        if sweep.run_frozen_profile:
            paths = [_result_path(root, cid, "theta", "dynamic", seed) for seed in seeds]
            profile = _frozen_profile(paths)
            frozen_profiles[cid] = profile
            profile_path = root / "combos" / cid / "theta" / "frozen_profile.npz"
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(profile_path, efficacy_by_e=profile)
            meta["frozen_profile"] = {
                "min": float(np.min(profile)),
                "max": float(np.max(profile)),
                "mean": float(np.mean(profile)),
                "sd_across_e": float(np.std(profile, ddof=1)),
            }
        combo_meta[cid] = meta
        write_json(meta, root / "combos" / cid / "manifest.json")

    # Wave 2: mode-specific mean-gain matched statics + full-mode frozen spatial profile.
    control_tasks = []
    for cid, tau_f, tau_r, cfg in grid:
        meta = combo_meta[cid]
        for regime, noise in (("theta", sweep.theta_noise_pa), ("attractor", sweep.attractor_noise_pa)):
            for mode in modes:
                p_match = float(meta["p_match"][regime][mode])
                label = f"{mode}_static"
                for seed in seeds:
                    control_tasks.append((
                        cfg, "static_gain", seed, float(noise), p_match, None,
                        str(_result_path(root, cid, regime, label, seed)), resume,
                    ))
        if sweep.run_frozen_profile:
            profile = frozen_profiles[cid]
            for seed in seeds:
                control_tasks.append((
                    cfg, "frozen_profile", seed, float(sweep.theta_noise_pa), None, profile,
                    str(_result_path(root, cid, "theta", "dynamic_frozen", seed)), resume,
                ))
    _run_parallel(control_tasks, workers, "matched-controls")

    global_manifest["completed"] = True
    global_manifest["combo_manifests"] = [f"combos/{cid}/manifest.json" for cid, *_ in grid]
    write_json(global_manifest, root / "experiment_manifest.json")
    return global_manifest
