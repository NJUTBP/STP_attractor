from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv

import numpy as np

from analysis.pav import axial_displacement_deg, compute_axial_pav
from config.schema import ProjectConfig
from experiment.phase15_tether_confirmation import branch_path, iter_phase15_branches
from io_utils import load_result, write_json

READOUT_MS = 2000.0
TRACE_RETENTION_MS = 250.0


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key); fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def _bootstrap_mean(values: np.ndarray, repeats: int, seed: int) -> tuple[float, float, float]:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(repeats, x.size))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _directed(raw: float | np.ndarray, delta: float) -> np.ndarray:
    return (1.0 if delta >= 0 else -1.0) * np.asarray(raw, float)


def _recovery(dq_dir: float, delta: float) -> float:
    return float(1.0 - dq_dir / abs(delta))


def _trace_axis_at(result, which: str, t_abs: float) -> float:
    if result.stp_trace_times_ms.size == 0:
        return float("nan")
    arr = result.stp_u_axis_deg if which == "u" else result.stp_r_axis_deg
    return float(np.interp(t_abs, result.stp_trace_times_ms, arr))


def analyze_phase15_tether_confirmation(
    cfg: ProjectConfig,
    results_dir: str | Path,
    *,
    bootstrap_repeats: int = 10000,
) -> dict:
    root = Path(results_dir)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    branches = iter_phase15_branches()
    missing = [b for b in branches if not branch_path(root, b).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} Phase-15 branch files are missing")

    pavs = {}
    results = {}
    for b in branches:
        r = load_result(branch_path(root, b))
        p = compute_axial_pav(r.spike_times_e_ms, r.spike_angles_e_deg, cfg.simulation, cfg.analysis)
        key = (b.condition, b.seed, b.shift_kind, b.delta_deg)
        results[key] = r; pavs[key] = p

    branch_rows = []
    for b in branches:
        if b.shift_kind == "no_shift":
            continue
        key = (b.condition, b.seed, b.shift_kind, b.delta_deg)
        p = pavs[key]
        ref = pavs[(b.condition, b.seed, "no_shift", 0.0)]
        q_ref = np.interp(p.times_ms, ref.times_ms, ref.axis_deg)
        raw = np.asarray(axial_displacement_deg(p.axis_deg, q_ref), float)
        dq_dir = _directed(raw, b.delta_deg)
        d2 = float(np.interp(b.checkpoint_ms + READOUT_MS, p.times_ms, dq_dir))
        branch_rows.append({
            "condition": b.condition, "seed": b.seed, "shift_kind": b.shift_kind,
            "delta_deg": b.delta_deg, "abs_delta_deg": abs(b.delta_deg),
            "dq_2000ms": d2, "R_2000ms": _recovery(d2, b.delta_deg),
        })

    grouped = defaultdict(list)
    for row in branch_rows:
        grouped[(row["condition"], row["shift_kind"], row["abs_delta_deg"], row["seed"])].append(row)
    seed_rows = []
    for (condition, kind, mag, seed), rows in sorted(grouped.items()):
        vals = np.asarray([r["R_2000ms"] for r in rows], float)
        seed_rows.append({
            "condition": condition, "shift_kind": kind, "abs_delta_deg": mag,
            "seed": seed, "n_directions": len(rows), "R_2000ms": float(np.mean(vals)),
        })

    lookup = {(r["condition"], r["shift_kind"], r["abs_delta_deg"], r["seed"]): r for r in seed_rows}
    contrast_specs = [
        ("PRIMARY_carrier_u", ("dynamic", "activity_shift"), ("dynamic", "activity_u_shift")),
        ("PRIMARY_route_output_vs_release", ("u_output_only", "activity_shift"), ("u_release_only", "activity_shift")),
        ("SUPPORT_co_shift", ("dynamic", "activity_shift"), ("dynamic", "co_shift")),
        ("SUPPORT_output_vs_static", ("u_output_only", "activity_shift"), ("static_gain", "activity_shift")),
        ("SUPPORT_dynamic_vs_r_shift", ("dynamic", "activity_shift"), ("dynamic", "activity_r_shift")),
        ("SUPPORT_dynamic_vs_output", ("dynamic", "activity_shift"), ("u_output_only", "activity_shift")),
    ]
    contrast_rows = []
    for mag in (11.25, 22.5):
        for ci, (name, a, b) in enumerate(contrast_specs):
            diffs = []
            for seed in sorted({r["seed"] for r in seed_rows}):
                ra = lookup.get((a[0], a[1], mag, seed)); rb = lookup.get((b[0], b[1], mag, seed))
                if ra is not None and rb is not None:
                    diffs.append(ra["R_2000ms"] - rb["R_2000ms"])
            est, lo, hi = _bootstrap_mean(np.asarray(diffs), bootstrap_repeats, 31500 + int(mag * 10) + ci)
            contrast_rows.append({
                "contrast": name, "abs_delta_deg": mag, "paired_seed_count": len(diffs),
                "R2s_difference": est, "ci_low": lo, "ci_high": hi,
                "direction_expected_positive": name != "SUPPORT_dynamic_vs_r_shift" and name != "SUPPORT_dynamic_vs_output",
            })

    # Direct hidden-state readout: after an activity-only shift, how much of the old
    # spatial trace remains at 250 ms relative to the same-seed no-shift trajectory?
    trace_seed_rows = []
    for mag in (11.25, 22.5):
        for seed in sorted({b.seed for b in branches}):
            vals_u, vals_r = [], []
            for delta in (mag, -mag):
                bkey = ("dynamic", seed, "activity_shift", float(delta))
                r = results[bkey]
                ref = results[("dynamic", seed, "no_shift", 0.0)]
                t0 = next(b.checkpoint_ms for b in branches if b.condition == "dynamic" and b.seed == seed)
                t = t0 + TRACE_RETENTION_MS
                u = _trace_axis_at(r, "u", t); ur = _trace_axis_at(ref, "u", t)
                rr = _trace_axis_at(r, "r", t); rref = _trace_axis_at(ref, "r", t)
                u_move = float(_directed(axial_displacement_deg(u, ur), delta))
                r_move = float(_directed(axial_displacement_deg(rr, rref), delta))
                vals_u.append(1.0 - u_move / mag)
                vals_r.append(1.0 - r_move / mag)
            trace_seed_rows.append({
                "seed": seed, "abs_delta_deg": mag,
                "u_retention_250ms": float(np.mean(vals_u)),
                "r_retention_250ms": float(np.mean(vals_r)),
                "u_minus_r_retention_250ms": float(np.mean(vals_u) - np.mean(vals_r)),
            })

    trace_summary_rows = []
    for mag in (11.25, 22.5):
        x = np.asarray([r["u_minus_r_retention_250ms"] for r in trace_seed_rows if r["abs_delta_deg"] == mag], float)
        est, lo, hi = _bootstrap_mean(x, bootstrap_repeats, 32000 + int(mag * 10))
        trace_summary_rows.append({
            "abs_delta_deg": mag, "paired_seed_count": int(x.size),
            "u_minus_r_retention_250ms": est, "ci_low": lo, "ci_high": hi,
        })

    _write_csv(out / "phase15_seed_metrics.csv", seed_rows)
    _write_csv(out / "phase15_confirmatory_contrasts.csv", contrast_rows)
    _write_csv(out / "phase15_trace_retention_seed.csv", trace_seed_rows)
    _write_csv(out / "phase15_trace_retention_summary.csv", trace_summary_rows)

    primary = [r for r in contrast_rows if r["abs_delta_deg"] == 11.25 and r["contrast"].startswith("PRIMARY")]
    decision = {
        r["contrast"]: ("supported" if np.isfinite(r["ci_low"]) and r["ci_low"] > 0 else "not_confirmed")
        for r in primary
    }
    summary = {
        "experiment": "phase15_tether_confirmation",
        "statistical_unit": "independent prehistory seed; +/- shift directions averaged within seed",
        "primary_delta_deg": 11.25,
        "primary_readout": "R_2000ms",
        "primary_decisions": decision,
        "confirmatory_contrasts": contrast_rows,
        "trace_retention_summary": trace_summary_rows,
        "interpretation_guard": (
            "Confirmation requires both primary 11.25-degree contrasts to have paired-bootstrap 95% CI above zero. "
            "Trace retention and 22.5-degree results are supporting, not substitutes for failed primary contrasts."
        ),
    }
    write_json(summary, out / "phase15_tether_confirmation_summary.json")
    return summary
