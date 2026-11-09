from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv

import numpy as np

from analysis.pav import axial_displacement_deg, compute_axial_pav
from config.schema import ProjectConfig
from experiment.phase14_tether import branch_path, iter_phase14_branches
from io_utils import load_result, write_json

READOUT_TIMES_MS = (500.0, 1000.0, 2000.0)
VELOCITY_WINDOW_MS = (500.0, 1000.0)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    # Rows in several Phase-14 tables intentionally mix per-seed records
    # with appended summary records.  Summary rows can contain additional
    # fields (for example bootstrap ci_low/ci_high), so deriving the CSV
    # schema from rows[0] is unsafe.  Preserve deterministic column order by
    # taking keys in first-seen order across the complete table.
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def directed_displacement(raw_dq: float | np.ndarray, delta_deg: float) -> np.ndarray:
    sign = 1.0 if delta_deg >= 0 else -1.0
    return sign * np.asarray(raw_dq, dtype=float)


def recovery_fraction(dq_dir: float | np.ndarray, delta_deg: float) -> np.ndarray:
    return 1.0 - np.asarray(dq_dir, dtype=float) / abs(float(delta_deg))


def effective_restoring_rate(dq_start: float, dq_end: float, delta_deg: float, dt_s: float) -> tuple[float, float]:
    """Return positive restoring velocity (deg/s) and normalized rate (1/s)."""
    if dt_s <= 0 or abs(delta_deg) < 1e-12:
        return float("nan"), float("nan")
    velocity = (float(dq_start) - float(dq_end)) / dt_s
    return velocity, velocity / abs(float(delta_deg))


def _bootstrap_mean(values: np.ndarray, repeats: int, seed: int) -> tuple[float, float, float]:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = float(np.mean(x))
    rng = np.random.default_rng(seed)
    vals = np.empty(repeats, dtype=float)
    for i in range(repeats):
        vals[i] = float(np.mean(x[rng.integers(0, x.size, size=x.size)]))
    return estimate, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _trace_metric_at(result, pav_times: np.ndarray, pav_axis: np.ndarray, t_abs: float) -> dict:
    if result.stp_trace_times_ms.size == 0:
        return {
            "activity_minus_u_deg": float("nan"), "activity_minus_r_deg": float("nan"),
            "u_axis_deg": float("nan"), "r_axis_deg": float("nan"),
            "u_resultant": float("nan"), "r_resultant": float("nan"),
            "u_mass": float("nan"), "r_mass": float("nan"),
        }
    q = float(np.interp(t_abs, pav_times, pav_axis))
    tu = np.asarray(result.stp_trace_times_ms, float)
    ua = float(np.interp(t_abs, tu, result.stp_u_axis_deg))
    ra = float(np.interp(t_abs, tu, result.stp_r_axis_deg))
    return {
        "activity_minus_u_deg": float(axial_displacement_deg(q, ua)),
        "activity_minus_r_deg": float(axial_displacement_deg(q, ra)),
        "u_axis_deg": ua,
        "r_axis_deg": ra,
        "u_resultant": float(np.interp(t_abs, tu, result.stp_u_resultant)),
        "r_resultant": float(np.interp(t_abs, tu, result.stp_r_resultant)),
        "u_mass": float(np.interp(t_abs, tu, result.stp_u_mass)),
        "r_mass": float(np.interp(t_abs, tu, result.stp_r_mass)),
    }


def analyze_phase14_tether(
    cfg: ProjectConfig,
    results_dir: str | Path,
    *,
    bootstrap_repeats: int = 5000,
) -> dict:
    root = Path(results_dir)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    branches = iter_phase14_branches()
    missing = [b for b in branches if not branch_path(root, b).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} Phase-14 branch files are missing")

    pavs = {}
    results = {}
    for b in branches:
        r = load_result(branch_path(root, b))
        p = compute_axial_pav(r.spike_times_e_ms, r.spike_angles_e_deg, cfg.simulation, cfg.analysis)
        key = (b.condition, b.seed, b.shift_kind, b.delta_deg)
        results[key] = r
        pavs[key] = p

    branch_rows: list[dict] = []
    for b in branches:
        if b.shift_kind == "no_shift":
            continue
        key = (b.condition, b.seed, b.shift_kind, b.delta_deg)
        p = pavs[key]
        ref = pavs[(b.condition, b.seed, "no_shift", 0.0)]
        q_ref = np.interp(p.times_ms, ref.times_ms, ref.axis_deg)
        raw = np.asarray(axial_displacement_deg(p.axis_deg, q_ref), float)
        dq = directed_displacement(raw, b.delta_deg)
        vals = {}
        for tm in READOUT_TIMES_MS:
            d = float(np.interp(b.checkpoint_ms + tm, p.times_ms, dq))
            vals[f"dq_{int(tm)}ms"] = d
            vals[f"R_{int(tm)}ms"] = float(recovery_fraction(d, b.delta_deg))
        d0 = float(np.interp(b.checkpoint_ms + VELOCITY_WINDOW_MS[0], p.times_ms, dq))
        d1 = float(np.interp(b.checkpoint_ms + VELOCITY_WINDOW_MS[1], p.times_ms, dq))
        vel, rate = effective_restoring_rate(
            d0, d1, b.delta_deg, (VELOCITY_WINDOW_MS[1] - VELOCITY_WINDOW_MS[0]) / 1000.0
        )
        trace = _trace_metric_at(results[key], p.times_ms, p.axis_deg, b.checkpoint_ms + 1000.0)
        branch_rows.append({
            "branch_id": b.branch_id,
            "condition": b.condition,
            "seed": int(b.seed),
            "shift_kind": b.shift_kind,
            "delta_deg": float(b.delta_deg),
            "abs_delta_deg": abs(float(b.delta_deg)),
            "restoring_velocity_0p5_1s_deg_per_s": vel,
            "effective_restoring_rate_s_inv": rate,
            **vals,
            **{f"trace_1s_{k}": v for k, v in trace.items()},
        })

    # Collapse +/- directions within seed before any across-seed inference.
    grouped: dict[tuple[str, str, float, int], list[dict]] = defaultdict(list)
    for r in branch_rows:
        grouped[(r["condition"], r["shift_kind"], r["abs_delta_deg"], r["seed"])].append(r)
    seed_rows: list[dict] = []
    metric_names = [
        "restoring_velocity_0p5_1s_deg_per_s", "effective_restoring_rate_s_inv",
        "dq_500ms", "dq_1000ms", "dq_2000ms", "R_500ms", "R_1000ms", "R_2000ms",
        "trace_1s_activity_minus_u_deg", "trace_1s_activity_minus_r_deg",
        "trace_1s_u_resultant", "trace_1s_r_resultant", "trace_1s_u_mass", "trace_1s_r_mass",
    ]
    for (condition, kind, mag, seed), items in sorted(grouped.items()):
        row = {"condition": condition, "shift_kind": kind, "abs_delta_deg": mag, "seed": seed, "n_directions": len(items)}
        for m in metric_names:
            vals = np.asarray([x[m] for x in items], float)
            row[m] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else float("nan")
        seed_rows.append(row)

    summary_rows: list[dict] = []
    sg: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for r in seed_rows:
        sg[(r["condition"], r["shift_kind"], r["abs_delta_deg"])].append(r)
    for gi, ((condition, kind, mag), items) in enumerate(sorted(sg.items())):
        row = {"condition": condition, "shift_kind": kind, "abs_delta_deg": mag, "n_seeds": len(items)}
        for m in ("effective_restoring_rate_s_inv", "R_1000ms", "R_2000ms"):
            est, lo, hi = _bootstrap_mean(np.asarray([x[m] for x in items], float), bootstrap_repeats, 24000 + gi)
            row[m] = est; row[f"{m}_ci_low"] = lo; row[f"{m}_ci_high"] = hi
        summary_rows.append(row)

    # One seed-level restoring-field slope per condition from activity_shift only.
    field_rows: list[dict] = []
    for condition in sorted({r["condition"] for r in seed_rows}):
        byseed = defaultdict(list)
        for r in seed_rows:
            if r["condition"] == condition and r["shift_kind"] == "activity_shift":
                byseed[r["seed"]].append(r)
        seed_rates = []
        for seed, items in sorted(byseed.items()):
            x = np.asarray([r["abs_delta_deg"] for r in items], float)
            y = np.asarray([r["restoring_velocity_0p5_1s_deg_per_s"] for r in items], float)
            good = np.isfinite(x) & np.isfinite(y)
            if np.count_nonzero(good) < 2:
                k = float("nan")
            else:
                # Through-origin local field: v_restore ~= k * |delta|.
                k = float(np.dot(x[good], y[good]) / np.dot(x[good], x[good]))
            field_rows.append({"condition": condition, "seed": seed, "field_rate_s_inv": k})
        est, lo, hi = _bootstrap_mean(np.asarray([r["field_rate_s_inv"] for r in field_rows if r["condition"] == condition]), bootstrap_repeats, 25000 + len(field_rows))
        field_rows.append({"condition": condition, "seed": "summary", "field_rate_s_inv": est, "ci_low": lo, "ci_high": hi})

    # Full-dynamic selective surgery: how much of activity-shift restoration is removed?
    surgery_rows: list[dict] = []
    dyn = {(r["shift_kind"], r["abs_delta_deg"], r["seed"]): r for r in seed_rows if r["condition"] == "dynamic"}
    for mag in sorted({r["abs_delta_deg"] for r in seed_rows if r["condition"] == "dynamic"}):
        for kind in ("activity_u_shift", "activity_r_shift", "co_shift"):
            diffs = []
            for seed in sorted({r["seed"] for r in seed_rows if r["condition"] == "dynamic"}):
                base = dyn.get(("activity_shift", mag, seed)); surg = dyn.get((kind, mag, seed))
                if base is not None and surg is not None:
                    diffs.append(float(base["R_2000ms"] - surg["R_2000ms"]))
            if diffs:
                est, lo, hi = _bootstrap_mean(np.asarray(diffs), bootstrap_repeats, 26000 + int(round(mag * 100)) + len(surgery_rows))
                surgery_rows.append({
                    "abs_delta_deg": mag,
                    "surgery": kind,
                    "paired_seed_count": len(diffs),
                    "activity_R2s_minus_surgery_R2s": est,
                    "ci_low": lo,
                    "ci_high": hi,
                })

    # Time-resolved activity-vs-u/R mismatch for Full dynamic branches.
    trace_rows: list[dict] = []
    for b in branches:
        if b.condition != "dynamic":
            continue
        key = (b.condition, b.seed, b.shift_kind, b.delta_deg)
        r = results[key]
        if r.stp_trace_times_ms.size == 0:
            continue
        p = pavs[key]
        mask = (r.stp_trace_times_ms >= b.checkpoint_ms) & (r.stp_trace_times_ms <= b.checkpoint_ms + 2000.0)
        # Keep 25-ms output cadence to limit file size; raw recorder remains at simulator sample cadence.
        idx = np.flatnonzero(mask)[::5]
        q = np.interp(r.stp_trace_times_ms[idx], p.times_ms, p.axis_deg)
        for ii, qq in zip(idx, q):
            trace_rows.append({
                "seed": b.seed,
                "shift_kind": b.shift_kind,
                "delta_deg": b.delta_deg,
                "elapsed_ms": float(r.stp_trace_times_ms[ii] - b.checkpoint_ms),
                "activity_axis_deg": float(qq),
                "u_axis_deg": float(r.stp_u_axis_deg[ii]),
                "r_axis_deg": float(r.stp_r_axis_deg[ii]),
                "activity_minus_u_deg": float(axial_displacement_deg(qq, r.stp_u_axis_deg[ii])),
                "activity_minus_r_deg": float(axial_displacement_deg(qq, r.stp_r_axis_deg[ii])),
                "u_resultant": float(r.stp_u_resultant[ii]),
                "r_resultant": float(r.stp_r_resultant[ii]),
                "u_mass": float(r.stp_u_mass[ii]),
                "r_mass": float(r.stp_r_mass[ii]),
            })

    _write_csv(out / "phase14_branch_metrics.csv", branch_rows)
    _write_csv(out / "phase14_seed_metrics.csv", seed_rows)
    _write_csv(out / "phase14_condition_summary.csv", summary_rows)
    _write_csv(out / "phase14_restoring_field.csv", field_rows)
    _write_csv(out / "phase14_surgery_comparisons.csv", surgery_rows)
    _write_csv(out / "phase14_trace_mismatch_timeseries.csv", trace_rows)

    summary = {
        "experiment": "phase14_attractor_tether",
        "statistical_unit": "prehistory seed; +/- shift directions are averaged within seed",
        "restoring_velocity_window_ms": list(VELOCITY_WINDOW_MS),
        "readout_times_ms": list(READOUT_TIMES_MS),
        "bootstrap_repeats": int(bootstrap_repeats),
        "field_definition": "v_restore=(dq_dir(0.5s)-dq_dir(1.0s))/0.5s; field_rate fits v_restore=k*|delta| through origin within each seed",
        "trace_definition": "u centroid weights=max(u-U,0); R centroid weights=max(1-R,0); doubled-angle axial centroid",
        "condition_summary": summary_rows,
        "surgery_comparisons": surgery_rows,
    }
    write_json(summary, out / "phase14_tether_summary.json")
    return summary
