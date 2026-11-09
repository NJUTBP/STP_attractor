from __future__ import annotations

from pathlib import Path
import csv

import numpy as np

from analysis.flow import summarize_flow, unwrap_axial_trace
from analysis.pav import axial_displacement_deg, circular_axis_sd_deg, summarize_attractor
from analysis.statistics import paired_bootstrap_mean_difference, paired_bootstrap_sd_difference
from config.phase13 import Phase13ConfirmatoryConfig
from config.schema import ProjectConfig
from experiment.mechanism_map import combo_id, config_for_combo
from experiment.phase13_confirmation import result_path
from io_utils import load_result, read_json, write_json


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _normalized_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if y.size != x.size or y.size < 2 or not np.all(np.isfinite(y)):
        return float("nan")
    width = float(x[-1] - x[0])
    if width <= 0:
        return float("nan")
    area = float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))
    return area / width


def analyze_phase13_attractor_confirmation(
    base_cfg: ProjectConfig,
    confirm: Phase13ConfirmatoryConfig,
    *,
    results_dir: str | Path,
) -> dict:
    base_cfg.validate()
    confirm.validate()
    root = Path(results_dir)
    manifest = read_json(root / "experiment_manifest.json")
    if manifest.get("experiment") != "phase13_attractor_noise_confirmation":
        raise ValueError("Results directory is not a Phase-13 confirmation experiment.")
    if not manifest.get("completed", False):
        raise ValueError("Phase-13 results are not marked completed.")

    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    noises = np.asarray(confirm.noise_levels_pa, dtype=float)
    zero_idx = int(np.where(noises == 0.0)[0][0])
    p_match_table: dict[str, dict[str, float]] = manifest["fixed_zero_noise_p_match"]

    seed_noise_rows: list[dict] = []
    combo_mode_rows: list[dict] = []
    noise_rows: list[dict] = []

    for tau_f, tau_r in confirm.combos:
        cid = combo_id(tau_f, tau_r)
        cfg = config_for_combo(base_cfg, tau_f, tau_r)
        for mode in confirm.temporal_modes:
            seed_integrated: list[dict] = []
            per_noise_collect = {
                float(n): {
                    "temporal_delta": [], "static_delta": [],
                    "temporal_abs": [], "static_abs": [],
                    "temporal_fail": [], "static_fail": [], "paired_valid": [],
                }
                for n in noises
            }
            zero_stationary_temporal: list[bool] = []
            zero_stationary_static: list[bool] = []
            max_t_fail = 0.0
            max_s_fail = 0.0

            for seed in confirm.seeds:
                temporal_results = [
                    load_result(result_path(root, cid, mode, "temporal", float(n), int(seed)))
                    for n in noises
                ]
                static_results = [
                    load_result(result_path(root, cid, mode, "static", float(n), int(seed)))
                    for n in noises
                ]
                t0 = temporal_results[zero_idx]
                s0 = static_results[zero_idx]
                at0, pt0 = summarize_attractor(t0, cfg.simulation, cfg.analysis)
                as0, ps0 = summarize_attractor(s0, cfg.simulation, cfg.analysis)
                ft0 = summarize_flow(unwrap_axial_trace(pt0, target_axis_deg=cfg.analysis.target_axis_deg, start_ms=1000.0))
                fs0 = summarize_flow(unwrap_axial_trace(ps0, target_axis_deg=cfg.analysis.target_axis_deg, start_ms=1000.0))
                t_stationary = (not at0.localization_failure) and abs(ft0.shift_1_8_deg) < confirm.stationary_abs_shift_cutoff_deg
                s_stationary = (not as0.localization_failure) and abs(fs0.shift_1_8_deg) < confirm.stationary_abs_shift_cutoff_deg
                zero_stationary_temporal.append(t_stationary)
                zero_stationary_static.append(s_stationary)

                abs_t_curve: list[float] = []
                abs_s_curve: list[float] = []
                all_pair_valid = t_stationary and s_stationary
                for noise, tr, sr in zip(noises, temporal_results, static_results):
                    at, _ = summarize_attractor(tr, cfg.simulation, cfg.analysis)
                    ast, _ = summarize_attractor(sr, cfg.simulation, cfg.analysis)
                    dt = float(axial_displacement_deg(at.pav_deg_at_target, at0.pav_deg_at_target))
                    ds = float(axial_displacement_deg(ast.pav_deg_at_target, as0.pav_deg_at_target))
                    pair_valid = t_stationary and s_stationary and (not at.localization_failure) and (not ast.localization_failure)
                    all_pair_valid = all_pair_valid and pair_valid
                    abs_t_curve.append(abs(dt) if pair_valid else float("nan"))
                    abs_s_curve.append(abs(ds) if pair_valid else float("nan"))
                    c = per_noise_collect[float(noise)]
                    c["temporal_delta"].append(dt)
                    c["static_delta"].append(ds)
                    c["temporal_abs"].append(abs(dt))
                    c["static_abs"].append(abs(ds))
                    c["temporal_fail"].append(int(at.localization_failure))
                    c["static_fail"].append(int(ast.localization_failure))
                    c["paired_valid"].append(bool(pair_valid))
                    seed_noise_rows.append({
                        "combo_id": cid,
                        "mode": mode,
                        "seed": int(seed),
                        "noise_pa": float(noise),
                        "frozen_p_match": float(p_match_table[cid][mode]),
                        "temporal_noise_abs_delta_deg": abs(dt),
                        "static_noise_abs_delta_deg": abs(ds),
                        "noise_resistance_gain_deg": abs(ds) - abs(dt),
                        "paired_valid": int(pair_valid),
                    })

                burden_t = _normalized_trapezoid(np.asarray(abs_t_curve, float), noises) if all_pair_valid else float("nan")
                burden_s = _normalized_trapezoid(np.asarray(abs_s_curve, float), noises) if all_pair_valid else float("nan")
                seed_integrated.append({
                    "temporal_burden_deg": burden_t,
                    "static_burden_deg": burden_s,
                    "resistance_gain_deg": burden_s - burden_t if np.isfinite(burden_t) and np.isfinite(burden_s) else float("nan"),
                })

            for noise in noises:
                c = per_noise_collect[float(noise)]
                valid = np.asarray(c["paired_valid"], bool)
                ta = np.asarray(c["temporal_abs"], float)[valid]
                sa = np.asarray(c["static_abs"], float)[valid]
                td = np.asarray(c["temporal_delta"], float)[valid]
                sd = np.asarray(c["static_delta"], float)[valid]
                t_fail = float(np.mean(c["temporal_fail"]))
                s_fail = float(np.mean(c["static_fail"]))
                max_t_fail = max(max_t_fail, t_fail)
                max_s_fail = max(max_s_fail, s_fail)
                mean_diff = paired_bootstrap_mean_difference(sa, ta, repeats=confirm.bootstrap_repeats, seed=91001 + int(noise))
                sd_diff = paired_bootstrap_sd_difference(sd, td, repeats=confirm.bootstrap_repeats, seed=92001 + int(noise))
                noise_rows.append({
                    "combo_id": cid,
                    "mode": mode,
                    "noise_pa": float(noise),
                    "paired_valid_n": int(np.sum(valid)),
                    "temporal_mean_abs_delta_deg": float(np.mean(ta)) if ta.size else float("nan"),
                    "static_mean_abs_delta_deg": float(np.mean(sa)) if sa.size else float("nan"),
                    "mean_abs_resistance_gain_deg": mean_diff.estimate,
                    "mean_abs_resistance_gain_ci_low": mean_diff.ci_low,
                    "mean_abs_resistance_gain_ci_high": mean_diff.ci_high,
                    "sd_resistance_gain_deg": sd_diff.estimate,
                    "temporal_localization_failure_fraction": t_fail,
                    "static_localization_failure_fraction": s_fail,
                })

            bt = np.asarray([x["temporal_burden_deg"] for x in seed_integrated], float)
            bs = np.asarray([x["static_burden_deg"] for x in seed_integrated], float)
            valid_burden = np.isfinite(bt) & np.isfinite(bs)
            burden_diff = paired_bootstrap_mean_difference(
                bs[valid_burden], bt[valid_burden],
                repeats=confirm.bootstrap_repeats,
                seed=93001,
            )
            stationary_t = float(np.mean(zero_stationary_temporal))
            stationary_s = float(np.mean(zero_stationary_static))
            guards_ok = (
                stationary_t >= 1.0 - confirm.localization_failure_fraction_cutoff
                and stationary_s >= 1.0 - confirm.localization_failure_fraction_cutoff
                and max_t_fail <= confirm.localization_failure_fraction_cutoff
                and max_s_fail <= confirm.localization_failure_fraction_cutoff
            )
            if guards_ok and np.isfinite(burden_diff.ci_low) and burden_diff.ci_low > 0:
                decision = "temporal_more_resistant"
            elif guards_ok and np.isfinite(burden_diff.ci_high) and burden_diff.ci_high < 0:
                decision = "temporal_less_resistant"
            elif not guards_ok:
                decision = "localization_or_stationarity_guard_failed"
            else:
                decision = "inconclusive"
            combo_mode_rows.append({
                "combo_id": cid,
                "mode": mode,
                "frozen_zero_noise_p_match": float(p_match_table[cid][mode]),
                "zero_noise_stationary_fraction_temporal": stationary_t,
                "zero_noise_stationary_fraction_static": stationary_s,
                "max_localization_failure_fraction_temporal": max_t_fail,
                "max_localization_failure_fraction_static": max_s_fail,
                "integrated_valid_seed_count": int(np.sum(valid_burden)),
                "temporal_integrated_noise_burden_deg": float(np.mean(bt[valid_burden])) if np.any(valid_burden) else float("nan"),
                "static_integrated_noise_burden_deg": float(np.mean(bs[valid_burden])) if np.any(valid_burden) else float("nan"),
                "integrated_resistance_gain_deg": burden_diff.estimate,
                "integrated_resistance_gain_ci_low": burden_diff.ci_low,
                "integrated_resistance_gain_ci_high": burden_diff.ci_high,
                "confirmatory_decision": decision,
            })

    _write_csv(seed_noise_rows, analysis_dir / "per_seed_noise_response.csv")
    _write_csv(noise_rows, analysis_dir / "noise_level_summary.csv")
    _write_csv(combo_mode_rows, analysis_dir / "combo_mode_summary.csv")

    primary_cid = combo_id(confirm.primary_tau_facilitation_ms, confirm.primary_tau_recovery_ms)
    anchor = next((r for r in combo_mode_rows if r["combo_id"] == primary_cid and r["mode"] == confirm.primary_mode), None)
    summary = {
        "experiment": "phase13_attractor_noise_confirmation",
        "run_count": confirm.expected_run_count(),
        "primary_confirmatory_anchor": anchor,
        "combo_mode_summary": combo_mode_rows,
        "decision_rule": (
            "temporal_more_resistant only if stationarity/localization guards pass "
            "and the paired bootstrap 95% CI for resistance_gain = burden_static - burden_temporal lies above 0."
        ),
    }
    write_json(summary, analysis_dir / "phase13_summary.json")
    return summary
