from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np

from config.schema import ProjectConfig
from io_utils import load_result, save_result, write_json, read_json
from model import simulate


def build_frozen_profile(
    source_dir: str | Path,
    *,
    seeds: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build the phase-7 frozen spatial E->I efficacy profile.

    Definition fixed by the phase-7 specification:
        p_frozen[i] = mean_seed( mean_{t=1..7s} u_i(t) R_i(t) )

    The inner time average is already stored in each dynamic phase-4 result as
    ``mean_efficacy_by_e``.  One pooled profile is then used for every frozen
    control seed; there is no per-seed fitting.
    """
    source_dir = Path(source_dir)
    seeds = tuple(int(s) for s in seeds)
    dynamic = []
    for seed in seeds:
        path = source_dir / "theta" / "dynamic" / f"seed_{seed:04d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing corrected dynamic source run: {path}")
        dynamic.append(load_result(path))

    if not dynamic:
        raise ValueError("No dynamic source runs were provided.")
    theta = dynamic[0].theta_e_deg
    for result in dynamic[1:]:
        if not np.array_equal(theta, result.theta_e_deg):
            raise ValueError("Dynamic source runs do not share the same E-cell angle grid.")

    profiles = np.stack([r.mean_efficacy_by_e for r in dynamic])
    profile = profiles.mean(axis=0)
    if np.any(~np.isfinite(profile)) or np.any(profile < 0.0) or np.any(profile > 1.0):
        raise ValueError("Derived frozen profile contains invalid efficacy values.")

    source_manifest_path = source_dir / "experiment_manifest.json"
    source_manifest = read_json(source_manifest_path) if source_manifest_path.exists() else {}
    diagnostics = {
        "definition": "mean across corrected theta/dynamic seeds of each run's 1-7 s mean_efficacy_by_e",
        "source_seed_count": len(dynamic),
        "source_seeds": list(seeds),
        "profile_min": float(np.min(profile)),
        "profile_max": float(np.max(profile)),
        "profile_mean": float(np.mean(profile)),
        "profile_sd_across_e": float(np.std(profile, ddof=1)),
        "source_p_match_theta": source_manifest.get("p_match_theta"),
    }
    return theta.copy(), profile, diagnostics


def _run_one(
    cfg: ProjectConfig,
    seed: int,
    noise_pa: float,
    profile: np.ndarray,
    path: str,
) -> None:
    result = simulate(
        cfg,
        condition="frozen_profile",
        seed=int(seed),
        noise_pa=float(noise_pa),
        frozen_profile=profile,
    )
    save_result(result, path)


def run_frozen_spatial(
    cfg: ProjectConfig,
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    workers: int = 1,
) -> dict:
    """Run the frozen-spatial-profile theta control using existing corrected dynamic runs."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = tuple(int(s) for s in cfg.experiment.theta_seeds)
    theta_e, profile, diagnostics = build_frozen_profile(source_dir, seeds=seeds)
    np.savez_compressed(
        output_dir / "frozen_profile.npz",
        theta_e_deg=theta_e,
        efficacy_by_e=profile,
    )

    condition_dir = output_dir / "theta" / "frozen_profile"
    tasks = [
        (
            cfg,
            seed,
            float(cfg.experiment.theta_noise_pa),
            profile,
            str(condition_dir / f"seed_{seed:04d}.npz"),
        )
        for seed in seeds
    ]
    if workers <= 1:
        for task in tasks:
            _run_one(*task)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_one, *task) for task in tasks]
            for future in futures:
                future.result()

    manifest = {
        "experiment": "phase7_frozen_spatial_profile",
        "source_corrected_results": str(source_dir.resolve()),
        "noise_pa": float(cfg.experiment.theta_noise_pa),
        "seeds": list(seeds),
        "condition": "frozen_profile",
        "workers": int(workers),
        "profile": diagnostics,
        "scientific_question": (
            "Can the static spatial E->I efficacy pattern induced by Dynamic STF reproduce theta "
            "after temporal/history dependence is removed?"
        ),
    }
    write_json(manifest, output_dir / "experiment_manifest.json")
    return manifest
