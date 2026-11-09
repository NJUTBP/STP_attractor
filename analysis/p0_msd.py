from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import csv

import numpy as np

from analysis.flow import unwrap_axial_trace
from analysis.pav import PAVTrace, axial_displacement_deg, compute_axial_pav
from config.schema import ProjectConfig
from experiment.mechanism_map import combo_id, config_for_combo
from experiment.phase12_attractor import diffusion_path, iter_module_a_jobs
from experiment.phase13_confirmation import result_path
from io_utils import load_result, read_json, write_json


# P0 is analysis-only.  The windows are deliberately expressed in elapsed time
# after the diffusion origin rather than absolute simulation time.
PHASE12_EARLY_WINDOW_MS = (400.0, 900.0)
PHASE12_LATE_WINDOW_MS = (1100.0, 1800.0)
PHASE13_ANALYSIS_START_MS = 1000.0
PHASE13_ANALYSIS_END_MS = 7000.0
PHASE13_EARLY_WINDOW_MS = (500.0, 2000.0)
PHASE13_LATE_WINDOW_MS = (3500.0, 6000.0)


@dataclass(frozen=True)
class DiffusionFit:
    d_deg2_per_s: float
    slope_deg2_per_s: float
    intercept_deg2: float
    r2: float
    n_timepoints: int


@dataclass(frozen=True)
class RelativeTrace:
    elapsed_ms: np.ndarray
    relative_deg: np.ndarray
    relative_from_origin_deg: np.ndarray
    reference_sq_deg2: np.ndarray
    increment_sq_deg2: np.ndarray


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _pav(result, cfg: ProjectConfig) -> PAVTrace:
    return compute_axial_pav(
        result.spike_times_e_ms,
        result.spike_angles_e_deg,
        cfg.simulation,
        cfg.analysis,
    )


def _relative_unwrapped_raw(noisy: PAVTrace, reference: PAVTrace) -> tuple[np.ndarray, np.ndarray]:
    """Return the continuous axial displacement noisy-reference on noisy's time grid.

    The wrapped relative angle is formed first and then the doubled angle is
    unwrapped.  This is safer for long trajectories than squaring the shortest
    wrapped displacement, which would artificially cap excursions at 90 deg.
    """
    times = np.asarray(noisy.times_ms, float)
    ref_axis = np.interp(times, np.asarray(reference.times_ms, float), np.asarray(reference.axis_deg, float))
    wrapped = np.asarray(axial_displacement_deg(np.asarray(noisy.axis_deg, float), ref_axis), float)
    rel = 0.5 * np.rad2deg(np.unwrap(np.deg2rad(2.0 * wrapped)))
    return times, rel


def relative_trace_on_grid(
    noisy: PAVTrace,
    reference: PAVTrace,
    *,
    start_ms: float,
    end_ms: float,
    step_ms: float,
) -> RelativeTrace:
    """Build a common-grid time-resolved noise divergence trace.

    ``relative_deg`` is the continuous noisy-vs-zero-noise PAV displacement.
    ``relative_from_origin_deg`` subtracts the displacement already present at
    ``start_ms``.  The latter is the quantity used for diffusion fits, so Phase
    13 estimates growth *after* the 1-s attractor-establishment guard rather
    than conflating it with divergence accumulated during cue/formation.
    """
    if end_ms <= start_ms:
        raise ValueError("end_ms must be greater than start_ms")
    if step_ms <= 0:
        raise ValueError("step_ms must be positive")
    times, rel = _relative_unwrapped_raw(noisy, reference)
    if start_ms < times[0] - 1e-9 or end_ms > times[-1] + 1e-9:
        raise ValueError(
            f"Requested [{start_ms}, {end_ms}] ms outside PAV range "
            f"[{times[0]}, {times[-1]}] ms"
        )
    elapsed = np.arange(0.0, end_ms - start_ms + 0.5 * step_ms, step_ms, dtype=float)
    abs_times = start_ms + elapsed
    rel_grid = np.interp(abs_times, times, rel)
    origin = float(np.interp(start_ms, times, rel))
    inc = rel_grid - origin
    return RelativeTrace(
        elapsed_ms=elapsed,
        relative_deg=rel_grid,
        relative_from_origin_deg=inc,
        reference_sq_deg2=rel_grid * rel_grid,
        increment_sq_deg2=inc * inc,
    )


def fit_effective_diffusion(
    elapsed_ms: np.ndarray,
    msd_deg2: np.ndarray,
    window_ms: tuple[float, float],
) -> DiffusionFit:
    """Fit MSD ~= intercept + 2 D t in the requested elapsed-time window.

    D is intentionally *not* clamped at zero.  A negative local slope is a
    scientifically useful sign that the selected interval is not behaving as
    free diffusion (for example because of restoring dynamics or MSD
    saturation).  R^2 is emitted so the linear-diffusion approximation can be
    audited rather than assumed.
    """
    t = np.asarray(elapsed_ms, float) / 1000.0
    y = np.asarray(msd_deg2, float)
    lo, hi = map(float, window_ms)
    mask = (np.asarray(elapsed_ms, float) >= lo) & (np.asarray(elapsed_ms, float) <= hi) & np.isfinite(y)
    if np.count_nonzero(mask) < 4:
        return DiffusionFit(float("nan"), float("nan"), float("nan"), float("nan"), int(np.count_nonzero(mask)))
    x = t[mask]
    yy = y[mask]
    slope, intercept = np.polyfit(x, yy, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return DiffusionFit(
        d_deg2_per_s=float(slope / 2.0),
        slope_deg2_per_s=float(slope),
        intercept_deg2=float(intercept),
        r2=float(r2),
        n_timepoints=int(np.count_nonzero(mask)),
    )


def _bootstrap_curve_fit(
    seed_curves: np.ndarray,
    elapsed_ms: np.ndarray,
    window_ms: tuple[float, float],
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float, float]:
    """Bootstrap seeds, recompute the ensemble MSD, then refit D."""
    arr = np.asarray(seed_curves, float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = fit_effective_diffusion(elapsed_ms, np.nanmean(arr, axis=0), window_ms).d_deg2_per_s
    rng = np.random.default_rng(seed)
    vals = np.empty(repeats, dtype=float)
    n = arr.shape[0]
    for i in range(repeats):
        idx = rng.integers(0, n, size=n)
        vals[i] = fit_effective_diffusion(elapsed_ms, np.nanmean(arr[idx], axis=0), window_ms).d_deg2_per_s
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return estimate, float("nan"), float("nan")
    return estimate, float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def _bootstrap_paired_diffusion_difference(
    control_curves: np.ndarray,
    temporal_curves: np.ndarray,
    elapsed_ms: np.ndarray,
    window_ms: tuple[float, float],
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float, float]:
    """Paired bootstrap of D_control - D_temporal over the seed axis."""
    c = np.asarray(control_curves, float)
    t = np.asarray(temporal_curves, float)
    if c.shape != t.shape or c.ndim != 2 or c.shape[0] == 0:
        return float("nan"), float("nan"), float("nan")

    def diff(cc: np.ndarray, tt: np.ndarray) -> float:
        dc = fit_effective_diffusion(elapsed_ms, np.nanmean(cc, axis=0), window_ms).d_deg2_per_s
        dt = fit_effective_diffusion(elapsed_ms, np.nanmean(tt, axis=0), window_ms).d_deg2_per_s
        return float(dc - dt)

    estimate = diff(c, t)
    rng = np.random.default_rng(seed)
    vals = np.empty(repeats, dtype=float)
    n = c.shape[0]
    for i in range(repeats):
        idx = rng.integers(0, n, size=n)
        vals[i] = diff(c[idx], t[idx])
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return estimate, float("nan"), float("nan")
    return estimate, float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def _curve_rows(
    *,
    dataset: str,
    group_fields: dict,
    elapsed_ms: np.ndarray,
    seed_curves: np.ndarray,
    reference_seed_curves: np.ndarray | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for j, tm in enumerate(elapsed_ms):
        vals = seed_curves[:, j]
        vals = vals[np.isfinite(vals)]
        ref_vals = None
        if reference_seed_curves is not None:
            rr = reference_seed_curves[:, j]
            ref_vals = rr[np.isfinite(rr)]
        row = {
            "dataset": dataset,
            **group_fields,
            "elapsed_ms": float(tm),
            "n_seeds": int(vals.size),
            "increment_msd_mean_deg2": float(np.mean(vals)) if vals.size else float("nan"),
            "increment_msd_median_deg2": float(np.median(vals)) if vals.size else float("nan"),
            "increment_msd_sem_deg2": float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else float("nan"),
        }
        if ref_vals is not None:
            row["reference_msd_mean_deg2"] = float(np.mean(ref_vals)) if ref_vals.size else float("nan")
        rows.append(row)
    return rows


def analyze_phase12_p0_msd(
    cfg: ProjectConfig,
    results_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    bootstrap_repeats: int = 5000,
) -> dict:
    """Reanalyze Phase-12 Module A as a hierarchical full-MSD experiment.

    Noise draws are averaged *within seed first*.  Seeds, not the 64
    seed-by-draw rows, are the independent bootstrap unit.
    """
    root = Path(results_dir)
    out = Path(output_dir) if output_dir is not None else root / "analysis" / "p0_msd"
    out.mkdir(parents=True, exist_ok=True)

    jobs = iter_module_a_jobs()
    refs: dict[tuple[str, int], PAVTrace] = {}
    noisy_jobs = []
    for job in jobs:
        path = diffusion_path(root, job)
        if not path.exists():
            raise FileNotFoundError(path)
        if job.noise_pa == 0.0:
            refs[(job.condition, job.seed)] = _pav(load_result(path), cfg)
        else:
            noisy_jobs.append(job)

    # condition/noise/seed -> list of draw curves
    inc: dict[tuple[str, float, int], list[np.ndarray]] = defaultdict(list)
    refsq: dict[tuple[str, float, int], list[np.ndarray]] = defaultdict(list)
    common_elapsed: np.ndarray | None = None
    step_ms = float(cfg.analysis.pav_step_ms)

    for job in noisy_jobs:
        result = load_result(diffusion_path(root, job))
        pav = _pav(result, cfg)
        ref = refs[(job.condition, job.seed)]
        rt = relative_trace_on_grid(
            pav,
            ref,
            start_ms=float(job.checkpoint_ms),
            end_ms=float(job.end_ms),
            step_ms=step_ms,
        )
        if common_elapsed is None:
            common_elapsed = rt.elapsed_ms
        elif not np.allclose(common_elapsed, rt.elapsed_ms):
            raise RuntimeError("Phase-12 elapsed grids are inconsistent after resampling.")
        key = (job.condition, float(job.noise_pa), int(job.seed))
        inc[key].append(rt.increment_sq_deg2)
        refsq[key].append(rt.reference_sq_deg2)

    if common_elapsed is None:
        raise RuntimeError("No Phase-12 noisy jobs found.")

    grouped_seed_inc: dict[tuple[str, float], dict[int, np.ndarray]] = defaultdict(dict)
    grouped_seed_ref: dict[tuple[str, float], dict[int, np.ndarray]] = defaultdict(dict)
    seed_rows: list[dict] = []
    for key, curves in sorted(inc.items()):
        condition, noise, seed = key
        seed_curve = np.nanmean(np.asarray(curves, float), axis=0)
        seed_ref = np.nanmean(np.asarray(refsq[key], float), axis=0)
        grouped_seed_inc[(condition, noise)][seed] = seed_curve
        grouped_seed_ref[(condition, noise)][seed] = seed_ref
        early = fit_effective_diffusion(common_elapsed, seed_curve, PHASE12_EARLY_WINDOW_MS)
        late = fit_effective_diffusion(common_elapsed, seed_curve, PHASE12_LATE_WINDOW_MS)
        seed_rows.append({
            "condition": condition,
            "noise_pa": noise,
            "seed": seed,
            "noise_draw_count": len(curves),
            "D_early_deg2_per_s": early.d_deg2_per_s,
            "D_early_r2": early.r2,
            "D_late_deg2_per_s": late.d_deg2_per_s,
            "D_late_r2": late.r2,
        })

    curve_rows: list[dict] = []
    summary_rows: list[dict] = []
    for gi, ((condition, noise), seed_map) in enumerate(sorted(grouped_seed_inc.items())):
        seeds = sorted(seed_map)
        seed_curves = np.asarray([seed_map[s] for s in seeds], float)
        seed_ref_curves = np.asarray([grouped_seed_ref[(condition, noise)][s] for s in seeds], float)
        mean_curve = np.nanmean(seed_curves, axis=0)
        early = fit_effective_diffusion(common_elapsed, mean_curve, PHASE12_EARLY_WINDOW_MS)
        late = fit_effective_diffusion(common_elapsed, mean_curve, PHASE12_LATE_WINDOW_MS)
        e_est, e_lo, e_hi = _bootstrap_curve_fit(
            seed_curves, common_elapsed, PHASE12_EARLY_WINDOW_MS,
            repeats=bootstrap_repeats, seed=12000 + gi,
        )
        l_est, l_lo, l_hi = _bootstrap_curve_fit(
            seed_curves, common_elapsed, PHASE12_LATE_WINDOW_MS,
            repeats=bootstrap_repeats, seed=13000 + gi,
        )
        summary_rows.append({
            "condition": condition,
            "noise_pa": noise,
            "n_seeds": len(seeds),
            "draws_per_seed": int(len(inc[(condition, noise, seeds[0])])),
            "D_early_deg2_per_s": e_est,
            "D_early_ci_low": e_lo,
            "D_early_ci_high": e_hi,
            "D_early_r2": early.r2,
            "D_late_deg2_per_s": l_est,
            "D_late_ci_low": l_lo,
            "D_late_ci_high": l_hi,
            "D_late_r2": late.r2,
        })
        curve_rows.extend(_curve_rows(
            dataset="phase12",
            group_fields={"condition": condition, "noise_pa": noise},
            elapsed_ms=common_elapsed,
            seed_curves=seed_curves,
            reference_seed_curves=seed_ref_curves,
        ))

    # Paired, seed-level comparisons against Full dynamic.
    comparison_rows: list[dict] = []
    controls = ("static_gain", "u_release_only", "u_output_only")
    noises = sorted({noise for _, noise in grouped_seed_inc})
    for noise in noises:
        dyn_map = grouped_seed_inc.get(("dynamic", noise), {})
        for ci, control in enumerate(controls):
            ctl_map = grouped_seed_inc.get((control, noise), {})
            seeds = sorted(set(dyn_map) & set(ctl_map))
            if not seeds:
                continue
            dyn = np.asarray([dyn_map[s] for s in seeds], float)
            ctl = np.asarray([ctl_map[s] for s in seeds], float)
            ee, elo, ehi = _bootstrap_paired_diffusion_difference(
                ctl, dyn, common_elapsed, PHASE12_EARLY_WINDOW_MS,
                repeats=bootstrap_repeats, seed=14000 + int(noise) * 10 + ci,
            )
            le, llo, lhi = _bootstrap_paired_diffusion_difference(
                ctl, dyn, common_elapsed, PHASE12_LATE_WINDOW_MS,
                repeats=bootstrap_repeats, seed=15000 + int(noise) * 10 + ci,
            )
            comparison_rows.append({
                "noise_pa": noise,
                "temporal_condition": "dynamic",
                "control_condition": control,
                "paired_seed_count": len(seeds),
                "D_early_control_minus_dynamic": ee,
                "D_early_diff_ci_low": elo,
                "D_early_diff_ci_high": ehi,
                "D_late_control_minus_dynamic": le,
                "D_late_diff_ci_low": llo,
                "D_late_diff_ci_high": lhi,
            })

    _write_csv(out / "phase12_seed_diffusion.csv", seed_rows)
    _write_csv(out / "phase12_msd_curves.csv", curve_rows)
    _write_csv(out / "phase12_diffusion_summary.csv", summary_rows)
    _write_csv(out / "phase12_paired_comparisons.csv", comparison_rows)
    summary = {
        "dataset": "phase12_module_a",
        "definition": (
            "MSD is the squared continuous axial PAV divergence from the matched zero-noise continuation. "
            "For D fits, the divergence already present at the diffusion origin is subtracted before squaring. "
            "Eight noise draws are averaged within each prehistory seed before seed-level aggregation/bootstrap."
        ),
        "pav_window_ms": float(cfg.analysis.pav_window_ms),
        "pav_step_ms": step_ms,
        "early_window_elapsed_ms": list(PHASE12_EARLY_WINDOW_MS),
        "late_window_elapsed_ms": list(PHASE12_LATE_WINDOW_MS),
        "bootstrap_repeats": int(bootstrap_repeats),
        "summary": summary_rows,
        "paired_comparisons": comparison_rows,
    }
    write_json(summary, out / "phase12_p0_msd_summary.json")
    return summary


def _localization_failure(trace: PAVTrace, cfg: ProjectConfig) -> bool:
    a = cfg.analysis
    t = np.asarray(trace.times_ms, float)
    l = np.asarray(trace.length, float)
    rmask = (t >= a.localization_reference_start_ms) & (t <= a.localization_reference_end_ms)
    lmask = (t >= a.localization_target_start_ms) & (t <= a.localization_target_end_ms)
    if not np.any(rmask) or not np.any(lmask):
        return True
    ref = float(np.nanmedian(l[rmask]))
    late = float(np.nanmedian(l[lmask]))
    ratio = late / ref if np.isfinite(ref) and ref > 0 else float("nan")
    return (not np.isfinite(ratio)) or ratio < a.localization_ratio_cutoff


def _zero_noise_stationary(trace: PAVTrace, cfg: ProjectConfig, cutoff_deg: float) -> bool:
    if _localization_failure(trace, cfg):
        return False
    flow = unwrap_axial_trace(trace, target_axis_deg=cfg.analysis.target_axis_deg, start_ms=1000.0)
    d1 = float(np.interp(1000.0, flow.times_ms, flow.displacement_deg))
    d8 = float(np.interp(8000.0, flow.times_ms, flow.displacement_deg))
    return abs(d8 - d1) < float(cutoff_deg)


def _phase13_groups_from_manifest(manifest: dict, *, scope: str) -> list[tuple[float, float, str]]:
    tf = tuple(float(x) for x in manifest["tau_facilitation_ms"])
    tr = tuple(float(x) for x in manifest["tau_recovery_ms"])
    modes = tuple(str(x) for x in manifest["temporal_modes"])
    if scope == "anchor":
        p = manifest["primary_confirmatory_anchor"]
        return [(float(p["tau_facilitation_ms"]), float(p["tau_recovery_ms"]), str(p["mode"]))]
    if scope != "all":
        raise ValueError("scope must be 'anchor' or 'all'")
    return [(a, b, m) for a, b in product(tf, tr) for m in modes]


def analyze_phase13_p0_msd(
    base_cfg: ProjectConfig,
    results_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    scope: str = "anchor",
    bootstrap_repeats: int = 5000,
) -> dict:
    """Reanalyze Phase-13 trajectories as time-resolved paired MSD curves.

    Each temporal or static noisy trajectory is compared with its own same-seed
    zero-noise trajectory.  Diffusion fits use divergence growth after 1 s,
    while ``reference_msd_mean_deg2`` retains the direct time-resolved analogue
    of the original Phase-13 endpoint comparison.

    Valid seeds follow the Phase-13 pairwise guard: both zero-noise temporal and
    static runs must be stationary/localized, and both noisy runs must remain
    localized.
    """
    root = Path(results_dir)
    manifest_path = root / "experiment_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    out = Path(output_dir) if output_dir is not None else root / "analysis" / "p0_msd"
    out.mkdir(parents=True, exist_ok=True)

    seeds = tuple(int(x) for x in manifest["confirmatory_seeds"])
    noises = tuple(float(x) for x in manifest["noise_levels_pa"])
    positive_noises = tuple(x for x in noises if x > 0.0)
    cutoff = 10.0
    # Keep the exact Phase-13 guard if it is persisted in a future manifest.
    if "stationary_abs_shift_cutoff_deg" in manifest:
        cutoff = float(manifest["stationary_abs_shift_cutoff_deg"])
    groups = _phase13_groups_from_manifest(manifest, scope=scope)
    step_ms = float(base_cfg.analysis.pav_step_ms)
    elapsed = np.arange(
        0.0,
        PHASE13_ANALYSIS_END_MS - PHASE13_ANALYSIS_START_MS + 0.5 * step_ms,
        step_ms,
        dtype=float,
    )

    seed_rows: list[dict] = []
    curve_rows: list[dict] = []
    summary_rows: list[dict] = []

    for gi, (tau_f, tau_r, mode) in enumerate(groups):
        cid = combo_id(tau_f, tau_r)
        cfg = config_for_combo(base_cfg, tau_f, tau_r)
        # Cache zero-noise traces once per seed/group.
        zero: dict[int, tuple[PAVTrace, PAVTrace, bool]] = {}
        for seed in seeds:
            t0_path = result_path(root, cid, mode, "temporal", 0.0, seed)
            s0_path = result_path(root, cid, mode, "static", 0.0, seed)
            if not t0_path.exists():
                raise FileNotFoundError(t0_path)
            if not s0_path.exists():
                raise FileNotFoundError(s0_path)
            tp = _pav(load_result(t0_path), cfg)
            sp = _pav(load_result(s0_path), cfg)
            valid0 = _zero_noise_stationary(tp, cfg, cutoff) and _zero_noise_stationary(sp, cfg, cutoff)
            zero[seed] = (tp, sp, valid0)

        for ni, noise in enumerate(positive_noises):
            temporal_curves: list[np.ndarray] = []
            temporal_ref_curves: list[np.ndarray] = []
            static_curves: list[np.ndarray] = []
            static_ref_curves: list[np.ndarray] = []
            valid_seeds: list[int] = []

            for seed in seeds:
                tp0, sp0, valid0 = zero[seed]
                tp_path = result_path(root, cid, mode, "temporal", noise, seed)
                sp_path = result_path(root, cid, mode, "static", noise, seed)
                if not tp_path.exists():
                    raise FileNotFoundError(tp_path)
                if not sp_path.exists():
                    raise FileNotFoundError(sp_path)
                tp = _pav(load_result(tp_path), cfg)
                sp = _pav(load_result(sp_path), cfg)
                valid = valid0 and (not _localization_failure(tp, cfg)) and (not _localization_failure(sp, cfg))
                if not valid:
                    seed_rows.append({
                        "combo_id": cid,
                        "tau_facilitation_ms": tau_f,
                        "tau_recovery_ms": tau_r,
                        "mode": mode,
                        "noise_pa": noise,
                        "seed": seed,
                        "paired_valid": 0,
                        "D_early_temporal_deg2_per_s": float("nan"),
                        "D_early_static_deg2_per_s": float("nan"),
                        "D_late_temporal_deg2_per_s": float("nan"),
                        "D_late_static_deg2_per_s": float("nan"),
                    })
                    continue
                tr_t = relative_trace_on_grid(
                    tp, tp0,
                    start_ms=PHASE13_ANALYSIS_START_MS,
                    end_ms=PHASE13_ANALYSIS_END_MS,
                    step_ms=step_ms,
                )
                tr_s = relative_trace_on_grid(
                    sp, sp0,
                    start_ms=PHASE13_ANALYSIS_START_MS,
                    end_ms=PHASE13_ANALYSIS_END_MS,
                    step_ms=step_ms,
                )
                if not np.allclose(tr_t.elapsed_ms, elapsed) or not np.allclose(tr_s.elapsed_ms, elapsed):
                    raise RuntimeError("Unexpected Phase-13 PAV time grid.")
                valid_seeds.append(seed)
                temporal_curves.append(tr_t.increment_sq_deg2)
                temporal_ref_curves.append(tr_t.reference_sq_deg2)
                static_curves.append(tr_s.increment_sq_deg2)
                static_ref_curves.append(tr_s.reference_sq_deg2)
                et = fit_effective_diffusion(elapsed, tr_t.increment_sq_deg2, PHASE13_EARLY_WINDOW_MS)
                es = fit_effective_diffusion(elapsed, tr_s.increment_sq_deg2, PHASE13_EARLY_WINDOW_MS)
                lt = fit_effective_diffusion(elapsed, tr_t.increment_sq_deg2, PHASE13_LATE_WINDOW_MS)
                ls = fit_effective_diffusion(elapsed, tr_s.increment_sq_deg2, PHASE13_LATE_WINDOW_MS)
                seed_rows.append({
                    "combo_id": cid,
                    "tau_facilitation_ms": tau_f,
                    "tau_recovery_ms": tau_r,
                    "mode": mode,
                    "noise_pa": noise,
                    "seed": seed,
                    "paired_valid": 1,
                    "D_early_temporal_deg2_per_s": et.d_deg2_per_s,
                    "D_early_static_deg2_per_s": es.d_deg2_per_s,
                    "D_late_temporal_deg2_per_s": lt.d_deg2_per_s,
                    "D_late_static_deg2_per_s": ls.d_deg2_per_s,
                })

            if not valid_seeds:
                summary_rows.append({
                    "combo_id": cid,
                    "tau_facilitation_ms": tau_f,
                    "tau_recovery_ms": tau_r,
                    "mode": mode,
                    "noise_pa": noise,
                    "paired_valid_n": 0,
                    "D_early_temporal_deg2_per_s": float("nan"),
                    "D_early_static_deg2_per_s": float("nan"),
                    "D_early_static_minus_temporal": float("nan"),
                    "D_early_diff_ci_low": float("nan"),
                    "D_early_diff_ci_high": float("nan"),
                    "D_late_temporal_deg2_per_s": float("nan"),
                    "D_late_static_deg2_per_s": float("nan"),
                    "D_late_static_minus_temporal": float("nan"),
                    "D_late_diff_ci_low": float("nan"),
                    "D_late_diff_ci_high": float("nan"),
                })
                continue

            tc = np.asarray(temporal_curves, float)
            sc = np.asarray(static_curves, float)
            trc = np.asarray(temporal_ref_curves, float)
            src = np.asarray(static_ref_curves, float)
            mt = np.nanmean(tc, axis=0)
            ms = np.nanmean(sc, axis=0)
            et = fit_effective_diffusion(elapsed, mt, PHASE13_EARLY_WINDOW_MS)
            es = fit_effective_diffusion(elapsed, ms, PHASE13_EARLY_WINDOW_MS)
            lt = fit_effective_diffusion(elapsed, mt, PHASE13_LATE_WINDOW_MS)
            ls = fit_effective_diffusion(elapsed, ms, PHASE13_LATE_WINDOW_MS)
            ed, elo, ehi = _bootstrap_paired_diffusion_difference(
                sc, tc, elapsed, PHASE13_EARLY_WINDOW_MS,
                repeats=bootstrap_repeats, seed=16000 + gi * 100 + ni,
            )
            ld, llo, lhi = _bootstrap_paired_diffusion_difference(
                sc, tc, elapsed, PHASE13_LATE_WINDOW_MS,
                repeats=bootstrap_repeats, seed=17000 + gi * 100 + ni,
            )
            summary_rows.append({
                "combo_id": cid,
                "tau_facilitation_ms": tau_f,
                "tau_recovery_ms": tau_r,
                "mode": mode,
                "noise_pa": noise,
                "paired_valid_n": len(valid_seeds),
                "D_early_temporal_deg2_per_s": et.d_deg2_per_s,
                "D_early_temporal_r2": et.r2,
                "D_early_static_deg2_per_s": es.d_deg2_per_s,
                "D_early_static_r2": es.r2,
                "D_early_static_minus_temporal": ed,
                "D_early_diff_ci_low": elo,
                "D_early_diff_ci_high": ehi,
                "D_late_temporal_deg2_per_s": lt.d_deg2_per_s,
                "D_late_temporal_r2": lt.r2,
                "D_late_static_deg2_per_s": ls.d_deg2_per_s,
                "D_late_static_r2": ls.r2,
                "D_late_static_minus_temporal": ld,
                "D_late_diff_ci_low": llo,
                "D_late_diff_ci_high": lhi,
            })
            curve_rows.extend(_curve_rows(
                dataset="phase13",
                group_fields={
                    "combo_id": cid, "mode": mode, "noise_pa": noise, "condition_kind": "temporal",
                },
                elapsed_ms=elapsed,
                seed_curves=tc,
                reference_seed_curves=trc,
            ))
            curve_rows.extend(_curve_rows(
                dataset="phase13",
                group_fields={
                    "combo_id": cid, "mode": mode, "noise_pa": noise, "condition_kind": "static",
                },
                elapsed_ms=elapsed,
                seed_curves=sc,
                reference_seed_curves=src,
            ))

    _write_csv(out / "phase13_seed_diffusion.csv", seed_rows)
    _write_csv(out / "phase13_msd_curves.csv", curve_rows)
    _write_csv(out / "phase13_diffusion_summary.csv", summary_rows)
    summary = {
        "dataset": "phase13",
        "scope": scope,
        "definition": (
            "For each condition kind and seed, noisy PAV is compared against its own same-seed zero-noise PAV. "
            "The continuous axial relative trace is unwrapped. reference_msd is the squared direct divergence; "
            "increment_msd subtracts the divergence already present at 1 s and is used for early/late D fits. "
            "Paired bootstrap resamples seeds and reports D_static - D_temporal; positive values favor lower temporal diffusion."
        ),
        "analysis_start_ms": PHASE13_ANALYSIS_START_MS,
        "analysis_end_ms": PHASE13_ANALYSIS_END_MS,
        "early_window_elapsed_ms": list(PHASE13_EARLY_WINDOW_MS),
        "late_window_elapsed_ms": list(PHASE13_LATE_WINDOW_MS),
        "bootstrap_repeats": int(bootstrap_repeats),
        "summary": summary_rows,
    }
    write_json(summary, out / "phase13_p0_msd_summary.json")
    return summary


def analyze_p0_msd(
    base_cfg: ProjectConfig,
    *,
    phase12_results: str | Path | None,
    phase13_results: str | Path | None,
    output_dir: str | Path,
    phase13_scope: str = "anchor",
    bootstrap_repeats: int = 5000,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "analysis": "P0_full_MSD_and_early_late_diffusion",
        "network_simulations_run": False,
        "bootstrap_repeats": int(bootstrap_repeats),
    }
    if phase12_results is not None:
        payload["phase12"] = analyze_phase12_p0_msd(
            base_cfg,
            phase12_results,
            output_dir=out / "phase12",
            bootstrap_repeats=bootstrap_repeats,
        )
    if phase13_results is not None:
        payload["phase13"] = analyze_phase13_p0_msd(
            base_cfg,
            phase13_results,
            output_dir=out / "phase13",
            scope=phase13_scope,
            bootstrap_repeats=bootstrap_repeats,
        )
    write_json(payload, out / "p0_msd_summary.json")
    return payload
