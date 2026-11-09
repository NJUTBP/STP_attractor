from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
from statistics import median

import numpy as np

from analysis.pav import axial_displacement_deg, compute_axial_pav
from analysis.phase11_pilot import directed_displacement, recovery_fraction
from config.schema import ProjectConfig
from experiment.phase12_attractor import (
    DIFFUSION_HORIZON_MS,
    SMALL_DELTA_DEG,
    diffusion_path,
    iter_module_a_jobs,
    iter_module_b_jobs,
    shift_path,
)
from io_utils import load_result, write_json


D_WINDOW = (200.0, 800.0)


def _interp(times: np.ndarray, values: np.ndarray, t: float) -> float:
    return float(np.interp(t, times, values))


def _pav(result, cfg: ProjectConfig):
    return compute_axial_pav(
        result.spike_times_e_ms,
        result.spike_angles_e_deg,
        cfg.simulation,
        cfg.analysis,
    )


def _i_pav(result, cfg: ProjectConfig):
    return compute_axial_pav(
        result.spike_times_i_ms,
        result.spike_angles_i_deg,
        cfg.simulation,
        cfg.analysis,
    )


def _d_short(times: np.ndarray, dq2: np.ndarray, t0: float) -> float:
    t_rel = times - t0
    mask = (t_rel >= D_WINDOW[0]) & (t_rel <= D_WINDOW[1]) & np.isfinite(dq2)
    if np.count_nonzero(mask) < 4:
        return float("nan")
    # ⟨δq²⟩ ≈ 2 D t  on a ring axis measured in degrees.
    slope = np.polyfit(2.0 * t_rel[mask], dq2[mask], 1)[0]
    return float(slope)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def analyze_module_a(cfg: ProjectConfig, results_dir: str | Path) -> dict:
    root = Path(results_dir)
    jobs = iter_module_a_jobs()
    traces = {}
    for job in jobs:
        path = diffusion_path(root, job)
        if not path.exists():
            raise FileNotFoundError(path)
        result = load_result(path)
        e_pav = _pav(result, cfg)
        i_pav = _i_pav(result, cfg)
        traces[(job.condition, job.seed, job.noise_pa, job.noise_draw)] = (job, e_pav, i_pav)

    rows = []
    for job in jobs:
        if job.noise_pa == 0.0:
            continue
        _j, e_pav, i_pav = traces[(job.condition, job.seed, job.noise_pa, job.noise_draw)]
        _z, e_ref, _i_ref = traces[(job.condition, job.seed, 0.0, 0)]
        q_ref = np.interp(e_pav.times_ms, e_ref.times_ms, e_ref.axis_deg)
        dq = np.asarray(axial_displacement_deg(e_pav.axis_deg, q_ref), float)
        t0 = job.checkpoint_ms
        dq2 = dq * dq
        ie_lag = np.asarray(axial_displacement_deg(i_pav.axis_deg, np.interp(i_pav.times_ms, e_pav.times_ms, e_pav.axis_deg)), float)
        rows.append({
            "job_id": job.job_id,
            "condition": job.condition,
            "seed": job.seed,
            "noise_pa": job.noise_pa,
            "noise_draw": job.noise_draw,
            "checkpoint_ms": t0,
            "abs_dq_200ms": abs(_interp(e_pav.times_ms, dq, t0 + 200.0)),
            "abs_dq_500ms": abs(_interp(e_pav.times_ms, dq, t0 + 500.0)),
            "abs_dq_1000ms": abs(_interp(e_pav.times_ms, dq, t0 + 1000.0)),
            "abs_dq_2000ms": abs(_interp(e_pav.times_ms, dq, t0 + 2000.0)),
            "msd_200ms": _interp(e_pav.times_ms, dq2, t0 + 200.0),
            "msd_500ms": _interp(e_pav.times_ms, dq2, t0 + 500.0),
            "msd_800ms": _interp(e_pav.times_ms, dq2, t0 + 800.0),
            "D_short": _d_short(e_pav.times_ms, dq2, t0),
            "pav_len_500ms": _interp(e_pav.times_ms, e_pav.length, t0 + 500.0),
            "pav_len_ratio_500ms": _interp(e_pav.times_ms, e_pav.length, t0 + 500.0)
            / max(_interp(e_ref.times_ms, e_ref.length, t0 + 500.0), 1e-9),
            "ie_lag_500ms": _interp(i_pav.times_ms, ie_lag, t0 + 500.0),
        })

    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["condition"], float(r["noise_pa"]))].append(r)
    summary_rows = []
    for (cond, noise), items in sorted(grouped.items()):
        def med(key: str) -> float:
            vals = [float(x[key]) for x in items if np.isfinite(float(x[key]))]
            return float(median(vals)) if vals else float("nan")

        summary_rows.append({
            "condition": cond,
            "noise_pa": noise,
            "n": len(items),
            "D_short_median": med("D_short"),
            "abs_dq_500ms_median": med("abs_dq_500ms"),
            "abs_dq_2000ms_median": med("abs_dq_2000ms"),
            "msd_800ms_median": med("msd_800ms"),
            "pav_len_ratio_500ms_median": med("pav_len_ratio_500ms"),
            "ie_lag_500ms_median": med("ie_lag_500ms"),
        })
    return {"rows": rows, "summary": summary_rows}


def analyze_module_b(cfg: ProjectConfig, results_dir: str | Path) -> dict:
    root = Path(results_dir)
    jobs = iter_module_b_jobs()
    traces = {}
    for job in jobs:
        path = shift_path(root, job)
        if not path.exists():
            raise FileNotFoundError(path)
        result = load_result(path)
        traces[(job.condition, job.seed, job.shift_kind, job.delta_deg)] = (job, _pav(result, cfg))

    rows = []
    for job in jobs:
        if job.shift_kind == "no_shift":
            continue
        _j, pav = traces[(job.condition, job.seed, job.shift_kind, job.delta_deg)]
        _z, ref = traces[(job.condition, job.seed, "no_shift", 0.0)]
        q_ref = np.interp(pav.times_ms, ref.times_ms, ref.axis_deg)
        raw = np.asarray(axial_displacement_deg(pav.axis_deg, q_ref), float)
        dq_dir = directed_displacement(raw, job.delta_deg)
        t0 = job.checkpoint_ms
        rows.append({
            "job_id": job.job_id,
            "condition": job.condition,
            "seed": job.seed,
            "shift_kind": job.shift_kind,
            "delta_deg": job.delta_deg,
            "R_500ms": float(recovery_fraction(_interp(pav.times_ms, dq_dir, t0 + 500.0), job.delta_deg)),
            "R_1000ms": float(recovery_fraction(_interp(pav.times_ms, dq_dir, t0 + 1000.0), job.delta_deg)),
            "R_2000ms": float(recovery_fraction(_interp(pav.times_ms, dq_dir, t0 + 2000.0), job.delta_deg)),
            "R_4000ms": float(recovery_fraction(_interp(pav.times_ms, dq_dir, t0 + 4000.0), job.delta_deg)),
            "dq_dir_4s": _interp(pav.times_ms, dq_dir, t0 + 4000.0),
            "pav_len_ratio_4s": _interp(pav.times_ms, pav.length, t0 + 4000.0)
            / max(_interp(ref.times_ms, ref.length, t0 + 4000.0), 1e-9),
        })

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["condition"], r["shift_kind"])].append(r)
    summary_rows = []
    for (cond, kind), items in sorted(grouped.items()):
        def med(key: str) -> float:
            return float(median(float(x[key]) for x in items))

        summary_rows.append({
            "condition": cond,
            "shift_kind": kind,
            "n": len(items),
            "delta_deg": SMALL_DELTA_DEG,
            "R_500ms_median": med("R_500ms"),
            "R_4000ms_median": med("R_4000ms"),
            "dq_dir_4s_median": med("dq_dir_4s"),
        })
    return {"rows": rows, "summary": summary_rows}


def analyze_phase12_attractor(cfg: ProjectConfig, results_dir: str | Path) -> dict:
    root = Path(results_dir)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    module_a = analyze_module_a(cfg, root) if (root / "module_a").exists() else {"rows": [], "summary": []}
    module_b = analyze_module_b(cfg, root) if (root / "module_b").exists() else {"rows": [], "summary": []}
    _write_csv(out / "module_a_trials.csv", module_a["rows"])
    _write_csv(out / "module_a_summary.csv", module_a["summary"])
    _write_csv(out / "module_b_trials.csv", module_b["rows"])
    _write_csv(out / "module_b_summary.csv", module_b["summary"])
    summary = {
        "diffusion_horizon_ms": DIFFUSION_HORIZON_MS,
        "d_short_window_ms": list(D_WINDOW),
        "small_delta_deg": SMALL_DELTA_DEG,
        "module_a_n": len(module_a["rows"]),
        "module_b_n": len(module_b["rows"]),
        "module_a_summary": module_a["summary"],
        "module_b_summary": module_b["summary"],
    }
    write_json(summary, out / "phase12_summary.json")
    return summary
