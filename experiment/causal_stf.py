from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

from analysis.pav import summarize_attractor
from analysis.theta import analyze_theta
from config.schema import ProjectConfig
from io_utils import save_result, write_json
from model import build_connectivity, simulate


def _condition_path(regime_dir: Path, condition: str, seed: int) -> Path:
    return regime_dir / condition / f"seed_{seed:04d}.npz"


def _simulate_save_task(
    cfg: ProjectConfig,
    condition: str,
    seed: int,
    noise_pa: float,
    static_gain: float | None,
    path: str,
) -> tuple[float, float]:
    # Worker-local connectivity avoids shipping FFT arrays between processes.
    result = simulate(
        cfg,
        condition=condition,
        seed=seed,
        noise_pa=noise_pa,
        static_gain=static_gain,
    )
    save_result(result, path)
    return result.gain_match_numerator, result.gain_match_denominator


def _run_tasks(tasks: list[tuple], workers: int) -> list[tuple[float, float]]:
    if workers <= 1:
        return [_simulate_save_task(*task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_simulate_save_task, *task) for task in tasks]
        return [f.result() for f in futures]


def run_regime(
    cfg: ProjectConfig,
    *,
    name: str,
    noise_pa: float,
    seeds: Iterable[int],
    output_dir: str | Path,
    workers: int = 1,
) -> float:
    """Run dynamic first, estimate pooled p_match, then paired baseline/static-gain controls."""
    output_dir = Path(output_dir)
    regime_dir = output_dir / name
    regime_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(s) for s in seeds)

    dynamic_tasks = [
        (
            cfg,
            "dynamic",
            seed,
            float(noise_pa),
            None,
            str(_condition_path(regime_dir, "dynamic", seed)),
        )
        for seed in seeds
    ]
    contributions = _run_tasks(dynamic_tasks, workers)
    numerator = sum(x[0] for x in contributions)
    denominator = sum(x[1] for x in contributions)
    if denominator <= 0:
        raise RuntimeError("Dynamic runs produced zero gain-match denominator; cannot define p_match.")
    p_match = numerator / denominator

    control_tasks = []
    for condition in ("baseline", "static_gain"):
        for seed in seeds:
            control_tasks.append(
                (
                    cfg,
                    condition,
                    seed,
                    float(noise_pa),
                    p_match if condition == "static_gain" else None,
                    str(_condition_path(regime_dir, condition, seed)),
                )
            )
    _run_tasks(control_tasks, workers)

    write_json(
        {
            "name": name,
            "noise_pa": noise_pa,
            "seeds": list(seeds),
            "p_match": p_match,
            "conditions": ["dynamic", "baseline", "static_gain"],
            "workers": workers,
        },
        regime_dir / "manifest.json",
    )
    return p_match


def run_preflight(cfg: ProjectConfig, output_dir: str | Path, seed: int = 0) -> dict:
    """Two dynamic-STF runs only: sigma=0 for theta and sigma=60 for localized attractor."""
    output_dir = Path(output_dir) / "preflight"
    conn = build_connectivity(cfg.network)
    results: dict[str, dict] = {}

    r0 = simulate(
        cfg,
        condition="dynamic",
        seed=seed,
        noise_pa=cfg.experiment.theta_noise_pa,
        connectivity=conn,
    )
    save_result(r0, output_dir / "theta_dynamic.npz")
    theta, _ = analyze_theta(r0, cfg.simulation, cfg.analysis)
    results["theta"] = {
        "dominant_frequency_hz": theta.dominant_frequency_hz,
        "normalized_peak_power": theta.normalized_peak_power,
        "dominant_in_theta_band": theta.dominant_in_theta_band,
    }

    r60 = simulate(
        cfg,
        condition="dynamic",
        seed=seed,
        noise_pa=cfg.experiment.attractor_noise_pa,
        connectivity=conn,
    )
    save_result(r60, output_dir / "attractor_dynamic.npz")
    attr, _ = summarize_attractor(r60, cfg.simulation, cfg.analysis)
    results["attractor"] = {
        "pav_deg_at_target": attr.pav_deg_at_target,
        "displacement_deg": attr.displacement_deg,
        "localization_ratio": attr.localization_ratio,
        "localization_failure": attr.localization_failure,
    }
    results["passed"] = bool(theta.dominant_in_theta_band and not attr.localization_failure)
    write_json(results, output_dir / "preflight_summary.json")
    return results


def run_phase4(cfg: ProjectConfig, output_dir: str | Path, *, workers: int = 1) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p_attr = run_regime(
        cfg,
        name="attractor",
        noise_pa=cfg.experiment.attractor_noise_pa,
        seeds=cfg.experiment.attractor_seeds,
        output_dir=output_dir,
        workers=workers,
    )
    p_theta = run_regime(
        cfg,
        name="theta",
        noise_pa=cfg.experiment.theta_noise_pa,
        seeds=cfg.experiment.theta_seeds,
        output_dir=output_dir,
        workers=workers,
    )
    manifest = {
        "p_match_attractor": p_attr,
        "p_match_theta": p_theta,
        "workers": workers,
        "config": cfg.to_dict(),
    }
    write_json(manifest, output_dir / "experiment_manifest.json")
    return manifest
