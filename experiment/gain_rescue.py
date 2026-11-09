from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from config.phase8 import Phase8SweepConfig
from config.schema import ProjectConfig
from experiment.mechanism_map import combo_id, config_for_combo
from io_utils import load_result, read_json, save_result, write_json
from model import simulate


def initial_rescue_scale(target_full_gain: float, depletion_gain: float) -> float:
    if target_full_gain <= 0 or depletion_gain <= 0:
        raise ValueError("Gain targets must be positive.")
    return float(target_full_gain / depletion_gain)


def corrected_rescue_scale(initial_scale: float, target_gain: float, observed_gain: float) -> float:
    if initial_scale <= 0 or target_gain <= 0 or observed_gain <= 0:
        raise ValueError("Calibration gains/scales must be positive.")
    return float(initial_scale * target_gain / observed_gain)


def _path(root: Path, cid: str, regime: str, label: str, seed: int) -> Path:
    return root / "combos" / cid / regime / label / f"seed_{seed:04d}.npz"


def _simulate_task(
    cfg: ProjectConfig,
    seed: int,
    noise_pa: float,
    temporal_gain_scale: float,
    path: str,
    resume: bool,
) -> tuple[str, float, float]:
    out = Path(path)
    if resume and out.exists():
        old = load_result(out)
        return path, old.gain_match_numerator, old.gain_match_denominator
    result = simulate(
        cfg,
        condition="depletion_only",
        seed=int(seed),
        noise_pa=float(noise_pa),
        temporal_gain_scale=float(temporal_gain_scale),
    )
    save_result(result, out)
    return path, result.gain_match_numerator, result.gain_match_denominator


def _run_parallel(tasks: list[tuple], workers: int, label: str) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    total = len(tasks)
    if workers <= 1:
        for i, task in enumerate(tasks, 1):
            p, n, d = _simulate_task(*task)
            out[p] = (n, d)
            if i % 25 == 0 or i == total:
                print(f"[{label}] {i}/{total}")
        return out
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_simulate_task, *task) for task in tasks]
        for i, f in enumerate(as_completed(futures), 1):
            p, n, d = f.result()
            out[p] = (n, d)
            if i % 50 == 0 or i == total:
                print(f"[{label}] {i}/{total}")
    return out


def _pooled_match(paths: Iterable[Path], contributions: dict[str, tuple[float, float]] | None = None) -> float:
    num = 0.0
    den = 0.0
    contributions = contributions or {}
    for path in paths:
        key = str(path)
        if key in contributions:
            n, d = contributions[key]
        else:
            r = load_result(path)
            n, d = r.gain_match_numerator, r.gain_match_denominator
        num += float(n)
        den += float(d)
    if den <= 0:
        raise RuntimeError("Cannot compute pooled gain: denominator <= 0")
    return num / den


def run_phase9_gain_rescued_depletion(
    base_cfg: ProjectConfig,
    sweep: Phase8SweepConfig,
    *,
    source_phase8_dir: str | Path,
    output_dir: str | Path,
    workers: int = 1,
    resume: bool = True,
) -> dict:
    """Test whether resource dynamics plus rescued mean gain reconstruct Full STP.

    For every Phase-8 (tau_f, tau_recovery) combination, use the Phase-8 Full-STP
    0-pA pooled p_match as the target.  Start from a scale equal to target/full
    depletion-only p_match ratio, run one pooled 0-pA calibration wave, correct
    the scale once, then use that same final scale at both 0 and 60 pA.

    Scaling affects the postsynaptic E->I efficacy U*R(t), but *not* the resource
    state update.  This is deliberately a computational sufficiency test:
    can the R(t) waveform reproduce Full STP once its mean synaptic drive is
    restored to the Full-STP operating point?
    """
    base_cfg.validate()
    sweep.validate()
    source = Path(source_phase8_dir)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(x) for x in sweep.seeds)

    combos: list[tuple[str, float, float, ProjectConfig, dict]] = []
    for tau_f, tau_r in sweep.combos:
        cid = combo_id(tau_f, tau_r)
        src_manifest = read_json(source / "combos" / cid / "manifest.json")
        cfg = config_for_combo(base_cfg, tau_f, tau_r)
        combos.append((cid, float(tau_f), float(tau_r), cfg, src_manifest))

    manifest = {
        "experiment": "phase9_gain_rescued_depletion",
        "scientific_question": (
            "Can depletion/recovery dynamics reconstruct the Full-STP theta and noise response when their mean E->I drive "
            "is rescued to the Full-STP 0-pA operating point?"
        ),
        "source_phase8_dir": str(source.resolve()),
        "seeds": list(seeds),
        "combo_count": len(combos),
        "calibration_runs": len(combos) * len(seeds),
        "final_runs": len(combos) * len(seeds) * 2,
        "expected_run_count": len(combos) * len(seeds) * 3,
        "theta_noise_pa": sweep.theta_noise_pa,
        "attractor_noise_pa": sweep.attractor_noise_pa,
        "scale_definition": "alpha multiplies depletion-only efficacy U*R(t) but does not alter resource-depletion state updates",
        "calibration": "one pooled 0-pA multiplicative correction after the Phase-8 p_match-ratio initial scale",
    }
    write_json(manifest, root / "experiment_manifest.json")

    # Wave 1: one 0-pA calibration run set per combo.
    calibration_tasks: list[tuple] = []
    calibration_meta: dict[str, dict] = {}
    for cid, tau_f, tau_r, cfg, src in combos:
        target = float(src["p_match"]["theta"]["dynamic"])
        dep = float(src["p_match"]["theta"]["depletion_only"])
        alpha0 = initial_rescue_scale(target, dep)
        calibration_meta[cid] = {
            "target_full_p_match_theta": target,
            "source_depletion_p_match_theta": dep,
            "initial_scale": alpha0,
        }
        for seed in seeds:
            calibration_tasks.append((
                cfg, seed, float(sweep.theta_noise_pa), alpha0,
                str(_path(root, cid, "calibration", "gain_rescued_depletion", seed)), resume,
            ))
    cal_contrib = _run_parallel(calibration_tasks, workers, "phase9-calibration")

    # One pooled multiplicative correction per combo.
    final_scales: dict[str, float] = {}
    for cid, tau_f, tau_r, cfg, src in combos:
        paths = [_path(root, cid, "calibration", "gain_rescued_depletion", s) for s in seeds]
        observed = _pooled_match(paths, cal_contrib)
        m = calibration_meta[cid]
        alpha1 = corrected_rescue_scale(m["initial_scale"], m["target_full_p_match_theta"], observed)
        m["calibration_observed_p_match"] = observed
        m["final_scale"] = alpha1
        final_scales[cid] = alpha1
        write_json(m, root / "combos" / cid / "calibration_manifest.json")

    # Wave 2: use the same final scale at 0 and 60 pA so q60-q0 remains a true noise perturbation.
    final_tasks: list[tuple] = []
    for cid, tau_f, tau_r, cfg, src in combos:
        alpha = final_scales[cid]
        for regime, noise in (("theta", sweep.theta_noise_pa), ("attractor", sweep.attractor_noise_pa)):
            for seed in seeds:
                final_tasks.append((
                    cfg, seed, float(noise), alpha,
                    str(_path(root, cid, regime, "gain_rescued_depletion", seed)), resume,
                ))
    _run_parallel(final_tasks, workers, "phase9-final")

    manifest["completed"] = True
    manifest["combo_calibration_manifests"] = [f"combos/{cid}/calibration_manifest.json" for cid, *_ in combos]
    write_json(manifest, root / "experiment_manifest.json")
    return manifest
