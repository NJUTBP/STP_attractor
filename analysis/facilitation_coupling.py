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
from experiment.facilitation_coupling import HYBRID_MODES
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
                fields.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _rho(x, y) -> dict:
    x = np.asarray(x, float); y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    if np.sum(good) < 3:
        return {"rho": float("nan"), "pvalue": float("nan"), "n": int(np.sum(good))}
    r, p = spearmanr(x[good], y[good])
    return {"rho": float(r), "pvalue": float(p), "n": int(np.sum(good))}


def _pooled_gain(results) -> float:
    num = sum(float(r.gain_match_numerator) for r in results)
    den = sum(float(r.gain_match_denominator) for r in results)
    return num / den if den > 0 else float("nan")


def _res(root: Path, cid: str, regime: str, mode: str, seed: int):
    return load_result(root / "combos" / cid / regime / mode / f"seed_{seed:04d}.npz")


def _mode_summary(cfg, r0s, r60s, target_gain: float, stationary_cutoff: float) -> dict:
    freq=[]; theta=[]; power=[]; zero_shift=[]; noise_abs=[]; noise_signed=[]
    fail60=0
    for r0, r60 in zip(r0s, r60s):
        th, _ = analyze_theta(r0, cfg.simulation, cfg.analysis)
        a0, p0 = summarize_attractor(r0, cfg.simulation, cfg.analysis)
        a60, _ = summarize_attractor(r60, cfg.simulation, cfg.analysis)
        flow = summarize_flow(unwrap_axial_trace(
            p0,
            target_axis_deg=cfg.analysis.target_axis_deg,
            start_ms=1000.0,
        ))
        delta = float(axial_displacement_deg(a60.pav_deg_at_target, a0.pav_deg_at_target))
        freq.append(th.dominant_frequency_hz)
        theta.append(float(th.dominant_in_theta_band))
        power.append(th.normalized_peak_power)
        zero_shift.append(abs(flow.shift_1_8_deg))
        noise_abs.append(abs(delta))
        noise_signed.append(delta)
        fail60 += int(a60.localization_failure)
    actual = _pooled_gain(r0s)
    return {
        "frequency_hz": float(np.mean(freq)),
        "frequency_sd_hz": float(np.std(freq, ddof=1)) if len(freq)>1 else 0.0,
        "theta_fraction": float(np.mean(theta)),
        "peak_power": float(np.mean(power)),
        "actual_p_match_theta": actual,
        "p_match_relative_error": (actual-target_gain)/target_gain if target_gain else float("nan"),
        "zero_noise_mean_abs_shift_deg": float(np.mean(zero_shift)),
        "stationary_fraction": float(np.mean(np.asarray(zero_shift)<stationary_cutoff)),
        "noise_mean_abs_delta_deg": float(np.mean(noise_abs)),
        "noise_delta_sd_deg": float(np.std(noise_signed, ddof=1)) if len(noise_signed)>1 else 0.0,
        "localization_failure_fraction_60pa": fail60/len(r60s),
    }


def analyze_phase10_facilitation_coupling(
    base_cfg: ProjectConfig,
    sweep: Phase8SweepConfig,
    *,
    source_phase8_dir: str | Path,
    source_phase9_dir: str | Path,
    results_dir: str | Path,
) -> dict:
    source8=Path(source_phase8_dir); source9=Path(source_phase9_dir); root=Path(results_dir)
    out=root/"analysis"; out.mkdir(parents=True,exist_ok=True)
    seeds=tuple(int(x) for x in sweep.seeds)
    p8={r["combo_id"]:r for r in _read_csv(source8/"analysis"/"combo_summary.csv")}
    p9={r["combo_id"]:r for r in _read_csv(source9/"analysis"/"combo_summary.csv")}

    rows=[]; seed_rows=[]
    for tau_f,tau_r in sweep.combos:
        cid=combo_id(tau_f,tau_r)
        src8=p8[cid]; src9=p9[cid]
        cfg=config_for_combo(base_cfg,tau_f,tau_r)
        meta=read_json(root/"combos"/cid/"manifest.json")
        full_freq=float(src8["dynamic_frequency_hz"])
        full_theta=float(src8["dynamic_theta_fraction"])
        full_noise=float(src8["dynamic_noise_mean_abs_delta_deg"])
        target=float(src8["dynamic_p_match_theta"])
        row={
            "combo_id":cid,
            "tau_facilitation_ms":float(tau_f),
            "tau_recovery_ms":float(tau_r),
            "full_frequency_hz":full_freq,
            "full_theta_fraction":full_theta,
            "full_noise_mean_abs_delta_deg":full_noise,
            "full_p_match_theta":target,
            "full_mean_spike_release_u":float(meta["full_mean_spike_release_u"]),
            "neither_frequency_hz":float(src9["rescue_frequency_hz"]),
            "neither_frequency_error_hz":float(src9["rescue_frequency_hz"])-full_freq,
            "neither_abs_frequency_error_hz":abs(float(src9["rescue_frequency_hz"])-full_freq),
            "neither_noise_mean_abs_delta_deg":float(src9["rescue_noise_mean_abs_delta_deg"]),
            "neither_noise_error_deg":float(src9["rescue_noise_mean_abs_delta_deg"])-full_noise,
        }
        mode_errors={"neither":row["neither_abs_frequency_error_hz"]}

        for mode in HYBRID_MODES:
            r0s=[_res(root,cid,"theta",mode,s) for s in seeds]
            r60s=[_res(root,cid,"attractor",mode,s) for s in seeds]
            ms=_mode_summary(cfg,r0s,r60s,target,sweep.stationary_abs_shift_cutoff_deg)
            final_scale=float(meta["modes"][mode]["final_scale"])
            row[f"{mode}_final_scale"]=final_scale
            for k,v in ms.items():
                row[f"{mode}_{k}"]=v
            row[f"{mode}_frequency_error_hz"]=ms["frequency_hz"]-full_freq
            row[f"{mode}_abs_frequency_error_hz"]=abs(ms["frequency_hz"]-full_freq)
            row[f"{mode}_noise_error_deg"]=ms["noise_mean_abs_delta_deg"]-full_noise
            row[f"{mode}_abs_noise_error_deg"]=abs(ms["noise_mean_abs_delta_deg"]-full_noise)
            row[f"{mode}_theta_classification_agreement"]=float(
                (ms["theta_fraction"]>0.5)==(full_theta>0.5)
            )
            mode_errors[mode]=row[f"{mode}_abs_frequency_error_hz"]

            for seed,r0,r60 in zip(seeds,r0s,r60s):
                th,_=analyze_theta(r0,cfg.simulation,cfg.analysis)
                a0,p0=summarize_attractor(r0,cfg.simulation,cfg.analysis)
                a60,_=summarize_attractor(r60,cfg.simulation,cfg.analysis)
                flow=summarize_flow(unwrap_axial_trace(p0,target_axis_deg=cfg.analysis.target_axis_deg,start_ms=1000.0))
                delta=float(axial_displacement_deg(a60.pav_deg_at_target,a0.pav_deg_at_target))
                seed_rows.append({
                    "combo_id":cid,"tau_facilitation_ms":float(tau_f),"tau_recovery_ms":float(tau_r),
                    "mode":mode,"seed":seed,"frequency_hz":th.dominant_frequency_hz,
                    "theta":int(th.dominant_in_theta_band),"zero_noise_abs_shift_deg":abs(flow.shift_1_8_deg),
                    "noise_abs_delta_deg":abs(delta),"noise_signed_delta_deg":delta,
                })

        row["best_frequency_reconstruction_mode"]=min(mode_errors,key=mode_errors.get)
        row["best_frequency_reconstruction_error_hz"]=min(mode_errors.values())
        rows.append(row)

    _write_csv(rows,out/"combo_summary.csv")
    _write_csv(seed_rows,out/"per_seed_summary.csv")

    def mode_aggregate(mode:str)->dict:
        if mode=="neither":
            freq=np.asarray([r["neither_frequency_hz"] for r in rows],float)
            ferr=np.asarray([r["neither_abs_frequency_error_hz"] for r in rows],float)
            noise=np.asarray([r["neither_noise_mean_abs_delta_deg"] for r in rows],float)
            nerr=np.abs(np.asarray([r["neither_noise_error_deg"] for r in rows],float))
            match=None
            agree=np.asarray([
                float((float(p9[r["combo_id"]]["rescue_theta_fraction"])>0.5)==(r["full_theta_fraction"]>0.5))
                for r in rows
            ])
        else:
            freq=np.asarray([r[f"{mode}_frequency_hz"] for r in rows],float)
            ferr=np.asarray([r[f"{mode}_abs_frequency_error_hz"] for r in rows],float)
            noise=np.asarray([r[f"{mode}_noise_mean_abs_delta_deg"] for r in rows],float)
            nerr=np.asarray([r[f"{mode}_abs_noise_error_deg"] for r in rows],float)
            match=np.abs(np.asarray([r[f"{mode}_p_match_relative_error"] for r in rows],float))
            agree=np.asarray([r[f"{mode}_theta_classification_agreement"] for r in rows],float)
        fullf=np.asarray([r["full_frequency_hz"] for r in rows],float)
        fulln=np.asarray([r["full_noise_mean_abs_delta_deg"] for r in rows],float)
        outd={
            "median_absolute_frequency_error_hz":float(np.median(ferr)),
            "mean_absolute_frequency_error_hz":float(np.mean(ferr)),
            "fraction_within_0p5_hz":float(np.mean(ferr<=0.5)),
            "fraction_within_1_hz":float(np.mean(ferr<=1.0)),
            "fraction_within_2_hz":float(np.mean(ferr<=2.0)),
            "theta_classification_agreement_fraction":float(np.mean(agree)),
            "spearman_full_vs_mode_frequency":_rho(fullf,freq),
            "median_absolute_noise_error_deg":float(np.median(nerr)),
            "mean_absolute_noise_error_deg":float(np.mean(nerr)),
            "spearman_full_vs_mode_noise":_rho(fulln,noise),
        }
        if match is not None:
            outd["gain_match_quality"]={
                "median_absolute_relative_error":float(np.median(match)),
                "max_absolute_relative_error":float(np.max(match)),
                "fraction_within_2_percent":float(np.mean(match<=0.02)),
                "fraction_within_5_percent":float(np.mean(match<=0.05)),
            }
        return outd

    best_counts={}
    for r in rows:
        b=r["best_frequency_reconstruction_mode"]
        best_counts[b]=best_counts.get(b,0)+1

    current=next((
        r for r in rows
        if abs(r["tau_facilitation_ms"]-base_cfg.stf.tau_facilitation_ms)<1e-9
        and abs(r["tau_recovery_ms"]-base_cfg.stf.tau_recovery_ms)<1e-9
    ),None)

    summary={
        "combo_count":len(rows),
        "seed_count_per_combo":len(seeds),
        "current_parameter_anchor":current,
        "mechanism_modes":{
            "both_dynamic":"Phase-8 Full STP (u in output and dynamic-u resource release)",
            "neither_dynamic":"Phase-9 gain-rescued depletion (U in output and fixed-U resource release)",
            "u_output_only":"dynamic u in output; fixed-U resource release",
            "u_release_only":"U*R output; dynamic u controls resource release",
            "mean_u_release":"U*R output; resource release uses Full-STP pooled mean spike-event u",
        },
        "reconstruction":{
            "neither":mode_aggregate("neither"),
            "u_output_only":mode_aggregate("u_output_only"),
            "u_release_only":mode_aggregate("u_release_only"),
            "mean_u_release":mode_aggregate("mean_u_release"),
        },
        "best_frequency_reconstruction_mode_counts":best_counts,
        "interpretation":{
            "u_output_only_success":"Direct temporal modulation by u(t) is sufficient once mean gain is matched.",
            "u_release_only_success":"The key role of facilitation is its dynamic control of resource depletion; direct u(t) output modulation is dispensable.",
            "mean_u_release_success":"A higher mean release probability is sufficient to reshape R(t); temporal u->R coupling is not required.",
            "only_full_succeeds":"Direct u(t) output modulation and u(t)->R(t) coupling act synergistically; neither route alone reconstructs Full STP.",
            "scope":"All new hybrid outputs are independently matched to Full-STP mean E->I gain. Mean-u-release uses a spike-event-weighted Full-STP release probability, not a hand-tuned value.",
        },
    }
    write_json(summary,out/"facilitation_coupling_summary.json")
    return summary
