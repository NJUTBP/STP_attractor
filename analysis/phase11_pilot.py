from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
from statistics import median

import numpy as np

from analysis.pav import axial_displacement_deg, compute_axial_pav
from config.schema import ProjectConfig
from experiment.phase11_pilot import PilotBranch, branch_path, iter_pilot_branches
from io_utils import load_result, write_json


RECOVERY_TIMES_MS = (500.0, 1000.0, 2000.0, 4000.0)
FORMULA_VERSION = "R = 1 - dq_dir / abs(delta); dq_dir = sign(delta) * wrap(q_int - q_noshift)"


def directed_displacement(raw_dq: float | np.ndarray, delta_deg: float) -> np.ndarray:
    """Positive values lie along the applied shift."""
    sign = 1.0 if delta_deg >= 0 else -1.0
    return sign * np.asarray(raw_dq, dtype=float)


def recovery_fraction(dq_dir: float | np.ndarray, delta_deg: float) -> np.ndarray:
    """Fraction of the imposed shift that has been undone.

    1 = back at the no_shift continuation, 0 = still sitting at the shifted
    location.  The denominator is |Δ|; using signed Δ after direction
    normalization double-counts the minus sign.
    """
    return 1.0 - np.asarray(dq_dir, dtype=float) / abs(float(delta_deg))


def _mean_abs(times: np.ndarray, values: np.ndarray, t0: float, t1: float) -> float:
    mask = (times >= t0) & (times <= t1)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    t = times[mask]
    v = np.abs(values[mask])
    area = np.trapezoid(v, t) if hasattr(np, "trapezoid") else np.trapz(v, t)
    return float(area / max(t1 - t0, 1e-9))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _summarize(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["condition"], r["shift_kind"])].append(r)

    condition_rows = []
    for (cond, kind), items in sorted(grouped.items()):
        def med(key: str) -> float:
            return float(median(float(x[key]) for x in items))

        condition_rows.append({
            "condition": cond,
            "shift_kind": kind,
            "n": len(items),
            "R_500ms_median": med("R_500ms"),
            "R_1000ms_median": med("R_1000ms"),
            "R_2000ms_median": med("R_2000ms"),
            "R_4000ms_median": med("R_4000ms"),
            "auc_0p1_4s_median": med("auc_0p1_4s"),
            "dq_dir_4s_median": med("dq_4s"),
            "pav_len_ratio_4s_median": med("pav_len_ratio_4s") if "pav_len_ratio_4s" in items[0] else float("nan"),
        })

    pairs = []
    by: dict[tuple[str, int, float], dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by[(r["condition"], int(r["seed"]), float(r["delta_deg"]))][r["shift_kind"]] = r
    for (cond, seed, delta), d in sorted(by.items()):
        if "activity_shift" in d and "co_shift" in d:
            tri = float(d["co_shift"]["auc_0p1_4s"]) - float(d["activity_shift"]["auc_0p1_4s"])
        else:
            tri = float("nan")
        if "trace_shift" in d:
            tdi = float(d["trace_shift"]["dq_2s"]) / abs(delta)
        else:
            tdi = float("nan")
        pairs.append({
            "condition": cond,
            "seed": seed,
            "delta_deg": delta,
            "TRI_auc": tri,
            "TDI_2s": tdi,
            "activity_R_4s": float(d["activity_shift"]["R_4000ms"]) if "activity_shift" in d else float("nan"),
            "co_R_4s": float(d["co_shift"]["R_4000ms"]) if "co_shift" in d else float("nan"),
            "trace_R_4s": float(d["trace_shift"]["R_4000ms"]) if "trace_shift" in d else float("nan"),
        })

    tri_summary = {}
    for cond in sorted({p["condition"] for p in pairs}):
        tris = [p["TRI_auc"] for p in pairs if p["condition"] == cond and np.isfinite(p["TRI_auc"])]
        tdis = [p["TDI_2s"] for p in pairs if p["condition"] == cond and np.isfinite(p["TDI_2s"])]
        tri_summary[cond] = {
            "n_pairs": len(tris),
            "TRI_median": float(median(tris)) if tris else None,
            "TRI_frac_positive": (sum(x > 0 for x in tris) / len(tris)) if tris else None,
            "TDI_median": float(median(tdis)) if tdis else None,
        }

    return {
        "formula_version": FORMULA_VERSION,
        "n_intervention_branches": len(rows),
        "recovery_times_ms": list(RECOVERY_TIMES_MS),
        "condition_medians": condition_rows,
        "tri_tdi_by_condition": tri_summary,
        "pairs": pairs,
    }


def analyze_phase11_anchor_pilot(cfg: ProjectConfig, results_dir: str | Path) -> dict:
    root = Path(results_dir)
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    branches = iter_pilot_branches()

    missing = [b for b in branches if not branch_path(root, b).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} branch NPZ files missing under {root / 'branches'}; "
            "cannot rebuild metrics from spikes."
        )

    traces: dict[tuple[str, int, str, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for b in branches:
        result = load_result(branch_path(root, b))
        pav = compute_axial_pav(
            result.spike_times_e_ms,
            result.spike_angles_e_deg,
            cfg.simulation,
            cfg.analysis,
        )
        traces[(b.condition, b.seed, b.shift_kind, b.delta_deg)] = (
            pav.times_ms,
            pav.axis_deg,
            pav.length,
        )

    rows = []
    for b in branches:
        if b.shift_kind == "no_shift":
            continue
        t_int, q_int, len_int = traces[(b.condition, b.seed, b.shift_kind, b.delta_deg)]
        t_ref, q_ref_trace, len_ref = traces[(b.condition, b.seed, "no_shift", 0.0)]
        q_ref = np.interp(t_int, t_ref, q_ref_trace)
        raw_dq = np.asarray(axial_displacement_deg(q_int, q_ref), float)
        dq_dir = directed_displacement(raw_dq, b.delta_deg)
        t0 = b.checkpoint_ms
        rec: dict[str, float] = {}
        for T in RECOVERY_TIMES_MS:
            val = float(np.interp(t0 + T, t_int, dq_dir))
            rec[f"R_{int(T)}ms"] = float(recovery_fraction(val, b.delta_deg))
        auc = _mean_abs(t_int, dq_dir, t0 + 100.0, t0 + 4000.0)
        len_ref_i = np.interp(t_int, t_ref, len_ref)
        len_ratio = np.divide(len_int, len_ref_i, out=np.full_like(len_int, np.nan), where=len_ref_i > 0)
        rows.append({
            "branch_id": b.branch_id,
            "condition": b.condition,
            "seed": int(b.seed),
            "shift_kind": b.shift_kind,
            "delta_deg": float(b.delta_deg),
            "checkpoint_ms": float(b.checkpoint_ms),
            "dq_raw_4s": float(np.interp(t0 + 4000.0, t_int, raw_dq)),
            "dq_0p5s": float(np.interp(t0 + 500.0, t_int, dq_dir)),
            "dq_1s": float(np.interp(t0 + 1000.0, t_int, dq_dir)),
            "dq_2s": float(np.interp(t0 + 2000.0, t_int, dq_dir)),
            "dq_4s": float(np.interp(t0 + 4000.0, t_int, dq_dir)),
            "auc_0p1_4s": auc,
            "pav_len_0p5s": float(np.interp(t0 + 500.0, t_int, len_int)),
            "pav_len_4s": float(np.interp(t0 + 4000.0, t_int, len_int)),
            "pav_len_ratio_0p5s": float(np.interp(t0 + 500.0, t_int, len_ratio)),
            "pav_len_ratio_4s": float(np.interp(t0 + 4000.0, t_int, len_ratio)),
            **rec,
        })

    summary = _summarize(rows)
    _write_csv(out_dir / "branch_metrics.csv", rows)
    _write_csv(out_dir / "condition_summary.csv", summary["condition_medians"])
    _write_csv(out_dir / "pair_metrics.csv", summary["pairs"])
    write_json({k: v for k, v in summary.items() if k != "pairs"}, out_dir / "pilot_summary.json")
    write_json({"pairs": summary["pairs"]}, out_dir / "pair_metrics.json")
    return summary
