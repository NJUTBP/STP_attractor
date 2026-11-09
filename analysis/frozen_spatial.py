from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import csv

import numpy as np

from analysis.flow import summarize_flow, unwrap_axial_trace
from analysis.pav import summarize_attractor
from analysis.statistics import paired_bootstrap_mean_difference
from analysis.theta import analyze_theta, PSDResult
from config.schema import ProjectConfig
from io_utils import load_result, read_json, write_json

SOURCE_CONDITIONS = ("dynamic", "baseline", "static_gain")


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _source_result(source_dir: Path, condition: str, seed: int):
    path = source_dir / "theta" / condition / f"seed_{seed:04d}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return load_result(path)


def _frozen_result(output_dir: Path, seed: int):
    path = output_dir / "theta" / "frozen_profile" / f"seed_{seed:04d}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return load_result(path)


def analyze_frozen_spatial(
    cfg: ProjectConfig,
    *,
    source_dir: str | Path,
    output_dir: str | Path,
) -> tuple[dict, dict[str, list[PSDResult]]]:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(s) for s in cfg.experiment.theta_seeds)

    condition_results: dict[str, list] = {c: [] for c in (*SOURCE_CONDITIONS, "frozen_profile")}
    for seed in seeds:
        for condition in SOURCE_CONDITIONS:
            condition_results[condition].append(_source_result(source_dir, condition, seed))
        condition_results["frozen_profile"].append(_frozen_result(output_dir, seed))

    per_seed_rows: list[dict] = []
    theta_by_condition: dict[str, dict[int, dict]] = {c: {} for c in condition_results}
    psd_by_condition: dict[str, list[PSDResult]] = {c: [] for c in condition_results}
    frozen_stationarity = []

    for condition, results in condition_results.items():
        for result in results:
            theta, psd = analyze_theta(result, cfg.simulation, cfg.analysis)
            row = asdict(theta)
            theta_by_condition[condition][theta.seed] = row
            psd_by_condition[condition].append(psd)
            per_seed_rows.append(row)

            if condition == "frozen_profile":
                attr, pav = summarize_attractor(result, cfg.simulation, cfg.analysis)
                flow = summarize_flow(
                    unwrap_axial_trace(pav, target_axis_deg=cfg.analysis.target_axis_deg, start_ms=1000.0)
                )
                frozen_stationarity.append(
                    {
                        "seed": result.seed,
                        "localization_failure": attr.localization_failure,
                        "localization_ratio": attr.localization_ratio,
                        "shift_1_8_deg": flow.shift_1_8_deg,
                        "velocity_6_8_deg_s": flow.velocity_6_8_deg_s,
                    }
                )

    _write_csv(per_seed_rows, analysis_dir / "theta_per_seed.csv")
    _write_csv(frozen_stationarity, analysis_dir / "frozen_stationarity.csv")

    summary: dict = {
        "profile": read_json(output_dir / "experiment_manifest.json").get("profile", {}),
        "theta": {},
        "paired_comparisons": {},
        "frozen_stationarity": {},
        "interpretation": {},
    }
    for condition, rows_by_seed in theta_by_condition.items():
        rows = [rows_by_seed[s] for s in seeds]
        freq = np.asarray([r["dominant_frequency_hz"] for r in rows], float)
        power = np.asarray([r["normalized_peak_power"] for r in rows], float)
        summary["theta"][condition] = {
            "n": len(rows),
            "mean_dominant_frequency_hz": float(np.mean(freq)),
            "sd_dominant_frequency_hz": float(np.std(freq, ddof=1)) if len(freq) > 1 else 0.0,
            "mean_normalized_peak_power": float(np.mean(power)),
            "theta_band_fraction": float(np.mean([r["dominant_in_theta_band"] for r in rows])),
        }

    frozen_freq = np.asarray([theta_by_condition["frozen_profile"][s]["dominant_frequency_hz"] for s in seeds], float)
    frozen_power = np.asarray([theta_by_condition["frozen_profile"][s]["normalized_peak_power"] for s in seeds], float)
    for other in ("dynamic", "static_gain", "baseline"):
        other_freq = np.asarray([theta_by_condition[other][s]["dominant_frequency_hz"] for s in seeds], float)
        other_power = np.asarray([theta_by_condition[other][s]["normalized_peak_power"] for s in seeds], float)
        summary["paired_comparisons"][f"frozen_minus_{other}_frequency"] = asdict(
            paired_bootstrap_mean_difference(
                frozen_freq,
                other_freq,
                repeats=cfg.analysis.bootstrap_repeats,
                seed=7001 + len(other),
            )
        )
        summary["paired_comparisons"][f"frozen_minus_{other}_peak_power"] = asdict(
            paired_bootstrap_mean_difference(
                frozen_power,
                other_power,
                repeats=cfg.analysis.bootstrap_repeats,
                seed=7101 + len(other),
            )
        )

    failures = sum(bool(r["localization_failure"]) for r in frozen_stationarity)
    abs_shift = np.asarray([abs(r["shift_1_8_deg"]) for r in frozen_stationarity], float)
    abs_vel = np.asarray([abs(r["velocity_6_8_deg_s"]) for r in frozen_stationarity], float)
    summary["frozen_stationarity"] = {
        "n": len(frozen_stationarity),
        "localization_failures": failures,
        "mean_absolute_shift_1_8_deg": float(np.mean(abs_shift)),
        "max_absolute_shift_1_8_deg": float(np.max(abs_shift)),
        "mean_absolute_velocity_6_8_deg_s": float(np.mean(abs_vel)),
    }

    frac_frozen = summary["theta"]["frozen_profile"]["theta_band_fraction"]
    mean_frozen = summary["theta"]["frozen_profile"]["mean_dominant_frequency_hz"]
    mean_dynamic = summary["theta"]["dynamic"]["mean_dominant_frequency_hz"]
    mean_static = summary["theta"]["static_gain"]["mean_dominant_frequency_hz"]
    summary["interpretation"] = {
        "primary_question": (
            "Does a time-independent spatial E->I efficacy profile, copied from the Dynamic-STF mean profile, "
            "reproduce the Dynamic-STF theta timescale?"
        ),
        "frozen_theta_fraction": frac_frozen,
        "frequency_context_hz": {
            "dynamic": mean_dynamic,
            "frozen_profile": mean_frozen,
            "static_gain": mean_static,
        },
        "guard": (
            "The frozen profile is the raw across-seed mean of Dynamic-STF mean_efficacy_by_e; it is not tuned "
            "to reproduce current, firing rate, or oscillation frequency. Interpret intermediate frequencies as "
            "evidence that spatial heterogeneity explains part, but not necessarily all, of the Dynamic-STF effect."
        ),
    }
    write_json(summary, analysis_dir / "frozen_spatial_summary.json")
    return summary, psd_by_condition
