#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.pav import circular_axis_mean_deg, circular_axis_sd_deg, summarize_attractor
from analysis.statistics import paired_bootstrap_mean_difference, paired_bootstrap_sd_difference
from analysis.theta import analyze_theta
from config import load_config
from io_utils import load_result, read_json, write_json

CONDITIONS = ("dynamic", "baseline", "static_gain")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze phase-4 results into causal endpoints.")
    p.add_argument("--config", default=str(ROOT / "config" / "phase4_causal.toml"))
    p.add_argument("--results", default=str(ROOT / "results" / "phase4_causal"))
    return p.parse_args()


def _load_regime(results_dir: Path, regime: str) -> dict[str, list]:
    out: dict[str, list] = {}
    for condition in CONDITIONS:
        paths = sorted((results_dir / regime / condition).glob("seed_*.npz"))
        out[condition] = [load_result(p) for p in paths]
    return out


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _paired_arrays(rows_by_condition: dict[str, dict[int, dict]], key: str, a: str, b: str, validity_key: str | None = None):
    seeds = sorted(set(rows_by_condition[a]) & set(rows_by_condition[b]))
    xa, xb = [], []
    for seed in seeds:
        ra, rb = rows_by_condition[a][seed], rows_by_condition[b][seed]
        if validity_key is not None and (not ra[validity_key] or not rb[validity_key]):
            xa.append(np.nan)
            xb.append(np.nan)
        else:
            xa.append(ra[key])
            xb.append(rb[key])
    return np.asarray(xa, float), np.asarray(xb, float), seeds


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    results_dir = Path(args.results)
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    attractor_results = _load_regime(results_dir, "attractor")
    attractor_summaries = []
    attr_map: dict[str, dict[int, dict]] = {c: {} for c in CONDITIONS}
    for condition, results in attractor_results.items():
        for result in results:
            summary, _ = summarize_attractor(result, cfg.simulation, cfg.analysis)
            row = asdict(summary)
            row["valid_for_drift"] = not summary.localization_failure
            attractor_summaries.append(row)
            attr_map[condition][summary.seed] = row
    _write_csv(attractor_summaries, analysis_dir / "attractor_summary.csv")

    theta_results = _load_regime(results_dir, "theta")
    theta_summaries = []
    theta_map: dict[str, dict[int, dict]] = {c: {} for c in CONDITIONS}
    for condition, results in theta_results.items():
        for result in results:
            summary, _ = analyze_theta(result, cfg.simulation, cfg.analysis)
            row = asdict(summary)
            theta_summaries.append(row)
            theta_map[condition][summary.seed] = row
    _write_csv(theta_summaries, analysis_dir / "theta_summary.csv")

    stats: dict = {"attractor": {}, "theta": {}, "interpretation_guards": {}}
    for condition in CONDITIONS:
        rows = list(attr_map[condition].values())
        failures = sum(bool(r["localization_failure"]) for r in rows)
        frac = failures / len(rows) if rows else float("nan")
        valid_disp = np.array([r["displacement_deg"] for r in rows if not r["localization_failure"]], float)
        stats["attractor"][condition] = {
            "n_total": len(rows),
            "n_localization_failures": failures,
            "localization_failure_fraction": frac,
            "endpoint_sd_deg_valid_trials": circular_axis_sd_deg(valid_disp),
            "mean_displacement_deg_valid_trials": circular_axis_mean_deg(valid_disp),
        }

    for other in ("baseline", "static_gain"):
        a, b, seeds = _paired_arrays(attr_map, "displacement_deg", "dynamic", other, validity_key="valid_for_drift")
        boot = paired_bootstrap_sd_difference(
            a, b, repeats=cfg.analysis.bootstrap_repeats, seed=831 + len(other)
        )
        stats["attractor"][f"dynamic_minus_{other}_sd"] = asdict(boot) | {"paired_seed_count": int(np.sum(np.isfinite(a) & np.isfinite(b)))}

    for condition in CONDITIONS:
        rows = list(theta_map[condition].values())
        f = np.array([r["dominant_frequency_hz"] for r in rows], float)
        q = np.array([r["normalized_peak_power"] for r in rows], float)
        stats["theta"][condition] = {
            "n": len(rows),
            "mean_dominant_frequency_hz": float(np.mean(f)) if f.size else float("nan"),
            "mean_normalized_peak_power": float(np.mean(q)) if q.size else float("nan"),
            "theta_band_fraction": float(np.mean([r["dominant_in_theta_band"] for r in rows])) if rows else float("nan"),
        }

    for other in ("baseline", "static_gain"):
        fa, fb, _ = _paired_arrays(theta_map, "dominant_frequency_hz", "dynamic", other)
        qa, qb, _ = _paired_arrays(theta_map, "normalized_peak_power", "dynamic", other)
        stats["theta"][f"dynamic_minus_{other}_frequency"] = asdict(
            paired_bootstrap_mean_difference(fa, fb, repeats=cfg.analysis.bootstrap_repeats, seed=1121 + len(other))
        )
        stats["theta"][f"dynamic_minus_{other}_peak_power"] = asdict(
            paired_bootstrap_mean_difference(qa, qb, repeats=cfg.analysis.bootstrap_repeats, seed=2211 + len(other))
        )

    frac_cut = cfg.analysis.localization_failure_fraction_cutoff
    stats["interpretation_guards"]["localization_failure_fraction_cutoff"] = frac_cut
    stats["interpretation_guards"]["conditions_exceeding_failure_cutoff"] = [
        c for c in CONDITIONS if stats["attractor"][c]["localization_failure_fraction"] > frac_cut
    ]
    stats["interpretation_guards"]["note"] = (
        "If a condition exceeds the localization-failure cutoff, do not interpret its PAV SD as a pure drift effect. "
        "Treat the manipulation as affecting attractor localization/persistence."
    )

    manifest = read_json(results_dir / "experiment_manifest.json") if (results_dir / "experiment_manifest.json").exists() else {}
    stats["p_match"] = {
        "attractor": manifest.get("p_match_attractor"),
        "theta": manifest.get("p_match_theta"),
    }
    write_json(stats, analysis_dir / "statistics.json")
    print(f"Wrote analysis to {analysis_dir}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
