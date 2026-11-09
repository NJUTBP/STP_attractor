from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
from scipy.stats import spearmanr

from analysis.flow import summarize_flow, unwrap_axial_trace
from analysis.pav import axial_displacement_deg, summarize_attractor
from analysis.theta import analyze_theta
from config.phase8 import Phase8SweepConfig
from config.schema import ProjectConfig
from experiment.mechanism_map import combo_id, config_for_combo
from io_utils import load_result, read_json, write_json


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _res(root: Path, cid: str, regime: str, seed: int):
    return load_result(root / "combos" / cid / regime / "gain_rescued_depletion" / f"seed_{seed:04d}.npz")


def _pooled_gain(results) -> float:
    num = sum(float(r.gain_match_numerator) for r in results)
    den = sum(float(r.gain_match_denominator) for r in results)
    return num / den if den > 0 else float("nan")


def _rho(x, y) -> dict:
    x = np.asarray(x, float); y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    if np.sum(good) < 3:
        return {"rho": float("nan"), "pvalue": float("nan"), "n": int(np.sum(good))}
    r, p = spearmanr(x[good], y[good])
    return {"rho": float(r), "pvalue": float(p), "n": int(np.sum(good))}


def analyze_phase9_gain_rescue(
    base_cfg: ProjectConfig,
    sweep: Phase8SweepConfig,
    *,
    source_phase8_dir: str | Path,
    results_dir: str | Path,
) -> dict:
    source = Path(source_phase8_dir)
    root = Path(results_dir)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(x) for x in sweep.seeds)

    phase8_rows = _read_csv(source / "analysis" / "combo_summary.csv")
    src_lookup = {r["combo_id"]: r for r in phase8_rows}
    rows: list[dict] = []
    seed_rows: list[dict] = []

    for tau_f, tau_r in sweep.combos:
        cid = combo_id(tau_f, tau_r)
        src = src_lookup[cid]
        cfg = config_for_combo(base_cfg, tau_f, tau_r)
        cal = read_json(root / "combos" / cid / "calibration_manifest.json")

        r0s = [_res(root, cid, "theta", s) for s in seeds]
        r60s = [_res(root, cid, "attractor", s) for s in seeds]
        rescue_freq = []
        rescue_theta = []
        rescue_power = []
        rescue_noise_abs = []
        rescue_zero_shift = []
        fail60 = 0

        for seed, r0, r60 in zip(seeds, r0s, r60s):
            th, _ = analyze_theta(r0, cfg.simulation, cfg.analysis)
            a0, p0 = summarize_attractor(r0, cfg.simulation, cfg.analysis)
            a60, _ = summarize_attractor(r60, cfg.simulation, cfg.analysis)
            flow = summarize_flow(unwrap_axial_trace(p0, target_axis_deg=cfg.analysis.target_axis_deg, start_ms=1000.0))
            delta = float(axial_displacement_deg(a60.pav_deg_at_target, a0.pav_deg_at_target))
            rescue_freq.append(th.dominant_frequency_hz)
            rescue_theta.append(float(th.dominant_in_theta_band))
            rescue_power.append(th.normalized_peak_power)
            rescue_noise_abs.append(abs(delta))
            rescue_zero_shift.append(abs(flow.shift_1_8_deg))
            fail60 += int(a60.localization_failure)
            seed_rows.append({
                "combo_id": cid,
                "tau_facilitation_ms": float(tau_f),
                "tau_recovery_ms": float(tau_r),
                "seed": seed,
                "rescue_frequency_hz": th.dominant_frequency_hz,
                "rescue_theta": int(th.dominant_in_theta_band),
                "rescue_zero_noise_abs_shift_deg": abs(flow.shift_1_8_deg),
                "rescue_noise_abs_delta_deg": abs(delta),
            })

        actual_match = _pooled_gain(r0s)
        target = float(src["dynamic_p_match_theta"])
        full_freq = float(src["dynamic_frequency_hz"])
        full_noise = float(src["dynamic_noise_mean_abs_delta_deg"])
        full_theta_frac = float(src["dynamic_theta_fraction"])
        rescue_freq_mean = float(np.mean(rescue_freq))
        rescue_noise_mean = float(np.mean(rescue_noise_abs))
        rel_match_err = (actual_match - target) / target if target else float("nan")

        rows.append({
            "combo_id": cid,
            "tau_facilitation_ms": float(tau_f),
            "tau_recovery_ms": float(tau_r),
            "initial_scale": float(cal["initial_scale"]),
            "final_scale": float(cal["final_scale"]),
            "target_full_p_match_theta": target,
            "rescue_actual_p_match_theta": actual_match,
            "p_match_relative_error": rel_match_err,
            "full_frequency_hz": full_freq,
            "depletion_only_frequency_hz": float(src["depletion_only_frequency_hz"]),
            "rescue_frequency_hz": rescue_freq_mean,
            "rescue_frequency_sd_hz": float(np.std(rescue_freq, ddof=1)) if len(rescue_freq) > 1 else 0.0,
            "rescue_minus_full_frequency_hz": rescue_freq_mean - full_freq,
            "abs_frequency_reconstruction_error_hz": abs(rescue_freq_mean - full_freq),
            "full_theta_fraction": full_theta_frac,
            "rescue_theta_fraction": float(np.mean(rescue_theta)),
            "theta_classification_agreement": float((full_theta_frac > 0.5) == (np.mean(rescue_theta) > 0.5)),
            "rescue_peak_power": float(np.mean(rescue_power)),
            "rescue_zero_noise_mean_abs_shift_deg": float(np.mean(rescue_zero_shift)),
            "rescue_stationary_fraction": float(np.mean(np.asarray(rescue_zero_shift) < sweep.stationary_abs_shift_cutoff_deg)),
            "full_noise_mean_abs_delta_deg": full_noise,
            "rescue_noise_mean_abs_delta_deg": rescue_noise_mean,
            "rescue_minus_full_noise_abs_delta_deg": rescue_noise_mean - full_noise,
            "full_matched_static_noise_mean_abs_delta_deg": float(src["dynamic_matched_static_noise_mean_abs_delta_deg"]),
            "rescue_localization_failure_fraction_60pa": fail60 / len(seeds),
        })

    _write_csv(rows, out / "combo_summary.csv")
    _write_csv(seed_rows, out / "per_seed_summary.csv")

    freq_err = np.asarray([r["abs_frequency_reconstruction_error_hz"] for r in rows], float)
    match_err = np.abs(np.asarray([r["p_match_relative_error"] for r in rows], float))
    noise_err = np.abs(np.asarray([r["rescue_minus_full_noise_abs_delta_deg"] for r in rows], float))
    full_f = [r["full_frequency_hz"] for r in rows]
    rescue_f = [r["rescue_frequency_hz"] for r in rows]
    full_n = [r["full_noise_mean_abs_delta_deg"] for r in rows]
    rescue_n = [r["rescue_noise_mean_abs_delta_deg"] for r in rows]

    current = next((r for r in rows if abs(r["tau_facilitation_ms"] - base_cfg.stf.tau_facilitation_ms) < 1e-9 and abs(r["tau_recovery_ms"] - base_cfg.stf.tau_recovery_ms) < 1e-9), None)
    summary = {
        "combo_count": len(rows),
        "seed_count_per_combo": len(seeds),
        "current_parameter_anchor": current,
        "gain_match_quality": {
            "median_absolute_relative_error": float(np.median(match_err)),
            "max_absolute_relative_error": float(np.max(match_err)),
            "fraction_within_2_percent": float(np.mean(match_err <= 0.02)),
            "fraction_within_5_percent": float(np.mean(match_err <= 0.05)),
        },
        "theta_reconstruction": {
            "median_absolute_frequency_error_hz": float(np.median(freq_err)),
            "mean_absolute_frequency_error_hz": float(np.mean(freq_err)),
            "fraction_within_0p5_hz": float(np.mean(freq_err <= 0.5)),
            "fraction_within_1_hz": float(np.mean(freq_err <= 1.0)),
            "fraction_within_2_hz": float(np.mean(freq_err <= 2.0)),
            "theta_classification_agreement_fraction": float(np.mean([r["theta_classification_agreement"] for r in rows])),
            "spearman_full_vs_rescue_frequency": _rho(full_f, rescue_f),
        },
        "noise_reconstruction": {
            "median_absolute_error_deg": float(np.median(noise_err)),
            "mean_absolute_error_deg": float(np.mean(noise_err)),
            "spearman_full_vs_rescue_noise": _rho(full_n, rescue_n),
        },
        "interpretation": {
            "positive_result": (
                "If gain-rescued depletion closely reproduces the Full-STP frequency surface, the parsimonious mechanism is: "
                "facilitation primarily elevates the operating E->I gain, while resource depletion/recovery supplies the slow temporal memory."
            ),
            "negative_result": (
                "If reconstruction remains poor despite close mean-gain matching, dynamic facilitation or its coupling to resource depletion contributes beyond mean gain."
            ),
            "scope": (
                "The rescue scale multiplies postsynaptic depletion-only efficacy but does not change the resource release update; success is a sufficiency result, failure is not by itself proof that dynamic u(t) is necessary."
            ),
        },
    }
    write_json(summary, out / "gain_rescue_summary.json")
    return summary
