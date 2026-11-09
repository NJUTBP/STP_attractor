from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import csv
import math

import numpy as np
from scipy.stats import spearmanr

from analysis.flow import summarize_flow, unwrap_axial_trace
from analysis.pav import axial_displacement_deg, summarize_attractor
from analysis.theta import analyze_theta
from config.phase8 import Phase8SweepConfig
from config.schema import ProjectConfig
from experiment.mechanism_map import TEMPORAL_MODES, combo_id, config_for_combo
from io_utils import load_result, read_json, write_json


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _result(root: Path, cid: str, regime: str, label: str, seed: int):
    path = root / "combos" / cid / regime / label / f"seed_{seed:04d}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return load_result(path)


def _baseline(root: Path, regime: str, seed: int):
    path = root / "baseline" / regime / f"seed_{seed:04d}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return load_result(path)


def _mean_sd(x: list[float]) -> tuple[float, float]:
    a = np.asarray(x, dtype=float)
    return float(np.mean(a)), float(np.std(a, ddof=1)) if a.size > 1 else 0.0


def _spearman(rows: list[dict], x_key: str, y_key: str) -> dict:
    x = np.asarray([float(r[x_key]) for r in rows], float)
    y = np.asarray([float(r[y_key]) for r in rows], float)
    good = np.isfinite(x) & np.isfinite(y)
    if np.sum(good) < 3:
        return {"rho": float("nan"), "pvalue": float("nan"), "n": int(np.sum(good))}
    rho, p = spearmanr(x[good], y[good])
    return {"rho": float(rho), "pvalue": float(p), "n": int(np.sum(good))}


def analyze_phase8_mechanism_map(
    base_cfg: ProjectConfig,
    sweep: Phase8SweepConfig,
    *,
    results_dir: str | Path,
) -> dict:
    root = Path(results_dir)
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(s) for s in sweep.seeds)
    modes = tuple(sweep.temporal_modes)

    baseline_theta_freq = []
    baseline_theta_frac = []
    baseline_delta = []
    for seed in seeds:
        r0 = _baseline(root, "theta", seed)
        r60 = _baseline(root, "attractor", seed)
        t, _ = analyze_theta(r0, base_cfg.simulation, base_cfg.analysis)
        baseline_theta_freq.append(t.dominant_frequency_hz)
        baseline_theta_frac.append(float(t.dominant_in_theta_band))
        a0, _ = summarize_attractor(r0, base_cfg.simulation, base_cfg.analysis)
        a60, _ = summarize_attractor(r60, base_cfg.simulation, base_cfg.analysis)
        baseline_delta.append(float(axial_displacement_deg(a60.pav_deg_at_target, a0.pav_deg_at_target)))

    baseline_summary = {
        "theta_mean_frequency_hz": float(np.mean(baseline_theta_freq)),
        "theta_band_fraction": float(np.mean(baseline_theta_frac)),
        "noise_mean_absolute_endpoint_delta_deg": float(np.mean(np.abs(baseline_delta))),
    }

    combo_rows: list[dict] = []
    seed_rows: list[dict] = []

    for tau_f, tau_r in sweep.combos:
        cid = combo_id(tau_f, tau_r)
        cfg = config_for_combo(base_cfg, tau_f, tau_r)
        manifest = read_json(root / "combos" / cid / "manifest.json")
        row: dict = {
            "combo_id": cid,
            "tau_facilitation_ms": float(tau_f),
            "tau_recovery_ms": float(tau_r),
            "log10_tau_facilitation": math.log10(float(tau_f)),
            "log10_tau_recovery": math.log10(float(tau_r)),
        }

        for mode in modes:
            temporal_freq: list[float] = []
            static_freq: list[float] = []
            temporal_power: list[float] = []
            static_power: list[float] = []
            temporal_theta: list[float] = []
            static_theta: list[float] = []
            zero_abs_shift: list[float] = []
            noise_abs_temporal: list[float] = []
            noise_abs_static: list[float] = []
            noise_delta_temporal: list[float] = []
            noise_delta_static: list[float] = []
            fail_temporal = 0
            fail_static = 0

            for seed in seeds:
                r0 = _result(root, cid, "theta", mode, seed)
                rs0 = _result(root, cid, "theta", f"{mode}_static", seed)
                r60 = _result(root, cid, "attractor", mode, seed)
                rs60 = _result(root, cid, "attractor", f"{mode}_static", seed)

                tt, _ = analyze_theta(r0, cfg.simulation, cfg.analysis)
                ts, _ = analyze_theta(rs0, cfg.simulation, cfg.analysis)
                temporal_freq.append(tt.dominant_frequency_hz)
                static_freq.append(ts.dominant_frequency_hz)
                temporal_power.append(tt.normalized_peak_power)
                static_power.append(ts.normalized_peak_power)
                temporal_theta.append(float(tt.dominant_in_theta_band))
                static_theta.append(float(ts.dominant_in_theta_band))

                a0, p0 = summarize_attractor(r0, cfg.simulation, cfg.analysis)
                as0, _ = summarize_attractor(rs0, cfg.simulation, cfg.analysis)
                a60, _ = summarize_attractor(r60, cfg.simulation, cfg.analysis)
                as60, _ = summarize_attractor(rs60, cfg.simulation, cfg.analysis)
                flow = summarize_flow(unwrap_axial_trace(p0, target_axis_deg=cfg.analysis.target_axis_deg, start_ms=1000.0))
                zero_abs_shift.append(abs(flow.shift_1_8_deg))
                dt = float(axial_displacement_deg(a60.pav_deg_at_target, a0.pav_deg_at_target))
                ds = float(axial_displacement_deg(as60.pav_deg_at_target, as0.pav_deg_at_target))
                noise_delta_temporal.append(dt)
                noise_delta_static.append(ds)
                noise_abs_temporal.append(abs(dt))
                noise_abs_static.append(abs(ds))
                fail_temporal += int(a60.localization_failure)
                fail_static += int(as60.localization_failure)

                seed_rows.append({
                    "combo_id": cid,
                    "tau_facilitation_ms": float(tau_f),
                    "tau_recovery_ms": float(tau_r),
                    "mode": mode,
                    "seed": seed,
                    "temporal_frequency_hz": tt.dominant_frequency_hz,
                    "matched_static_frequency_hz": ts.dominant_frequency_hz,
                    "temporal_minus_static_frequency_hz": tt.dominant_frequency_hz - ts.dominant_frequency_hz,
                    "zero_noise_abs_shift_1_8_deg": abs(flow.shift_1_8_deg),
                    "noise_endpoint_delta_temporal_deg": dt,
                    "noise_endpoint_delta_static_deg": ds,
                    "noise_abs_delta_temporal_deg": abs(dt),
                    "noise_abs_delta_static_deg": abs(ds),
                    "noise_resistance_gain_deg": abs(ds) - abs(dt),
                    "localization_failure_temporal_60pa": int(a60.localization_failure),
                    "localization_failure_static_60pa": int(as60.localization_failure),
                })

            tf_mean, tf_sd = _mean_sd(temporal_freq)
            sf_mean, sf_sd = _mean_sd(static_freq)
            row[f"{mode}_frequency_hz"] = tf_mean
            row[f"{mode}_frequency_sd_hz"] = tf_sd
            row[f"{mode}_matched_static_frequency_hz"] = sf_mean
            row[f"{mode}_matched_static_frequency_sd_hz"] = sf_sd
            row[f"{mode}_temporal_slowing_hz"] = sf_mean - tf_mean
            row[f"{mode}_theta_fraction"] = float(np.mean(temporal_theta))
            row[f"{mode}_matched_static_theta_fraction"] = float(np.mean(static_theta))
            row[f"{mode}_peak_power"] = float(np.mean(temporal_power))
            row[f"{mode}_matched_static_peak_power"] = float(np.mean(static_power))
            row[f"{mode}_p_match_theta"] = float(manifest["p_match"]["theta"][mode])
            row[f"{mode}_p_match_attractor"] = float(manifest["p_match"]["attractor"][mode])
            row[f"{mode}_zero_noise_mean_abs_shift_deg"] = float(np.mean(zero_abs_shift))
            row[f"{mode}_stationary_fraction"] = float(np.mean(np.asarray(zero_abs_shift) < sweep.stationary_abs_shift_cutoff_deg))
            row[f"{mode}_noise_mean_abs_delta_deg"] = float(np.mean(noise_abs_temporal))
            row[f"{mode}_matched_static_noise_mean_abs_delta_deg"] = float(np.mean(noise_abs_static))
            row[f"{mode}_noise_resistance_gain_deg"] = float(np.mean(noise_abs_static) - np.mean(noise_abs_temporal))
            row[f"{mode}_noise_delta_sd_deg"] = float(np.std(noise_delta_temporal, ddof=1)) if len(seeds) > 1 else 0.0
            row[f"{mode}_matched_static_noise_delta_sd_deg"] = float(np.std(noise_delta_static, ddof=1)) if len(seeds) > 1 else 0.0
            row[f"{mode}_localization_failure_fraction_60pa"] = fail_temporal / len(seeds)
            row[f"{mode}_matched_static_localization_failure_fraction_60pa"] = fail_static / len(seeds)

        if sweep.run_frozen_profile:
            frozen_freq = []
            frozen_theta = []
            frozen_shift = []
            for seed in seeds:
                rf = _result(root, cid, "theta", "dynamic_frozen", seed)
                ft, _ = analyze_theta(rf, cfg.simulation, cfg.analysis)
                fa, fp = summarize_attractor(rf, cfg.simulation, cfg.analysis)
                flow = summarize_flow(unwrap_axial_trace(fp, target_axis_deg=cfg.analysis.target_axis_deg, start_ms=1000.0))
                frozen_freq.append(ft.dominant_frequency_hz)
                frozen_theta.append(float(ft.dominant_in_theta_band))
                frozen_shift.append(abs(flow.shift_1_8_deg))
            ff, ffsd = _mean_sd(frozen_freq)
            row["dynamic_frozen_frequency_hz"] = ff
            row["dynamic_frozen_frequency_sd_hz"] = ffsd
            row["dynamic_frozen_theta_fraction"] = float(np.mean(frozen_theta))
            row["dynamic_frozen_zero_noise_mean_abs_shift_deg"] = float(np.mean(frozen_shift))
            row["dynamic_temporal_vs_frozen_slowing_hz"] = ff - row["dynamic_frequency_hz"]
            row["dynamic_spatial_vs_scalar_slowing_hz"] = row["dynamic_matched_static_frequency_hz"] - ff
            profile = manifest.get("frozen_profile", {})
            row["dynamic_frozen_profile_mean"] = profile.get("mean", float("nan"))
            row["dynamic_frozen_profile_sd_across_e"] = profile.get("sd_across_e", float("nan"))

        combo_rows.append(row)

    _write_csv(combo_rows, analysis_dir / "combo_summary.csv")
    _write_csv(seed_rows, analysis_dir / "per_seed_summary.csv")

    current = next(
        (r for r in combo_rows if abs(r["tau_facilitation_ms"] - base_cfg.stf.tau_facilitation_ms) < 1e-9
         and abs(r["tau_recovery_ms"] - base_cfg.stf.tau_recovery_ms) < 1e-9),
        None,
    )
    global_summary = {
        "combo_count": len(combo_rows),
        "seed_count_per_combo": len(seeds),
        "baseline": baseline_summary,
        "current_parameter_anchor": current,
        "theta_region_fraction": {
            mode: float(np.mean([r[f"{mode}_theta_fraction"] > 0 for r in combo_rows])) for mode in modes
        },
        "stationary_region_fraction": {
            mode: float(np.mean([r[f"{mode}_stationary_fraction"] == 1.0 for r in combo_rows])) for mode in modes
        },
        "parameter_associations": {},
        "interpretation_keys": {
            "temporal_slowing_hz": "matched-static dominant frequency minus temporal-mode frequency; positive values quantify slowing not explained by mean E->I gain.",
            "noise_resistance_gain_deg": "matched-static mean |q60-q0| minus temporal-mode mean |q60-q0|; positive values mean the temporal dynamics reduce the incremental 60-pA endpoint perturbation.",
            "dynamic_temporal_vs_frozen_slowing_hz": "frozen-spatial frequency minus full-dynamic frequency; positive values isolate the part not reproduced by the time-averaged spatial profile.",
        },
    }
    for mode in modes:
        for y in (f"{mode}_frequency_hz", f"{mode}_temporal_slowing_hz", f"{mode}_noise_resistance_gain_deg"):
            global_summary["parameter_associations"][f"log_tau_f_vs_{y}"] = _spearman(combo_rows, "log10_tau_facilitation", y)
            global_summary["parameter_associations"][f"log_tau_r_vs_{y}"] = _spearman(combo_rows, "log10_tau_recovery", y)

    write_json(global_summary, analysis_dir / "mechanism_map_summary.json")
    return global_summary
