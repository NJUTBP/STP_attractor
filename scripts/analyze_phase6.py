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

from analysis.oscillation import diagnose_oscillation
from analysis.pav import circular_axis_mean_deg, circular_axis_sd_deg, summarize_attractor
from analysis.statistics import (
    paired_bootstrap_circular_mean_difference,
    paired_bootstrap_mean_absolute_displacement_difference,
    paired_bootstrap_mean_difference,
)
from analysis.theta import analyze_theta
from config import load_config
from io_utils import load_result, write_json

CONDITIONS = ("dynamic", "baseline", "static_gain")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-6 follow-up analysis using existing phase-4 raw NPZ files only.")
    p.add_argument("--config", default=str(ROOT / "config" / "phase4_causal.toml"))
    p.add_argument("--results", default=str(ROOT / "results" / "phase4_causal"))
    return p.parse_args()


def load_regime(results_dir: Path, regime: str) -> dict[str, list]:
    out = {}
    for condition in CONDITIONS:
        paths = sorted((results_dir / regime / condition).glob("seed_*.npz"))
        if not paths:
            raise FileNotFoundError(f"No raw NPZ results found for {regime}/{condition}")
        out[condition] = [load_result(p) for p in paths]
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def paired_by_seed(rows: list[dict], condition_a: str, condition_b: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    ma = {int(r["seed"]): r for r in rows if r["condition"] == condition_a}
    mb = {int(r["seed"]): r for r in rows if r["condition"] == condition_b}
    seeds = sorted(set(ma) & set(mb))
    return (
        np.asarray([ma[s][key] for s in seeds], dtype=float),
        np.asarray([mb[s][key] for s in seeds], dtype=float),
    )


def endpoint_summary(rows: list[dict], repeats: int, seed_base: int) -> dict:
    out: dict = {}
    for condition in CONDITIONS:
        x = np.asarray([r["displacement_deg"] for r in rows if r["condition"] == condition and not r["localization_failure"]], float)
        out[condition] = {
            "n": int(x.size),
            "circular_mean_displacement_deg": circular_axis_mean_deg(x),
            "circular_sd_deg": circular_axis_sd_deg(x),
            "mean_absolute_displacement_deg": float(np.mean(np.abs(x))) if x.size else float("nan"),
        }
    for j, other in enumerate(("baseline", "static_gain")):
        a, b = paired_by_seed(rows, "dynamic", other, "displacement_deg")
        out[f"dynamic_minus_{other}_circular_mean"] = asdict(
            paired_bootstrap_circular_mean_difference(a, b, repeats=repeats, seed=seed_base + 10 * j + 1)
        )
        out[f"dynamic_minus_{other}_mean_absolute_displacement"] = asdict(
            paired_bootstrap_mean_absolute_displacement_difference(a, b, repeats=repeats, seed=seed_base + 10 * j + 2)
        )
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    results_dir = Path(args.results)
    out_dir = results_dir / "analysis_phase6"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Re-use the already-computed 0-pA theta runs as a zero-noise attractor diagnostic.
    theta_results = load_regime(results_dir, "theta")
    zero_rows: list[dict] = []
    osc_rows: list[dict] = []
    for condition, results in theta_results.items():
        for result in results:
            attr, _ = summarize_attractor(result, cfg.simulation, cfg.analysis)
            zero_rows.append(asdict(attr))

            theta, psd = analyze_theta(result, cfg.simulation, cfg.analysis)
            mask = (result.sample_times_ms >= cfg.analysis.theta_start_ms) & (result.sample_times_ms <= cfg.analysis.theta_end_ms)
            diag = diagnose_oscillation(
                result.plfp_gaba_e[mask],
                cfg.simulation.sampling_rate_hz,
                psd.frequency_hz,
                psd.psd,
                min_frequency_hz=2.0,
                max_frequency_hz=cfg.analysis.max_psd_hz,
            )
            row = asdict(theta)
            row.update(
                {
                    "acf_best_frequency_hz": diag.acf_best_frequency_hz,
                    "acf_best_correlation": diag.acf_best_correlation,
                    "acf_first_peak_frequency_hz": diag.acf_first_peak_frequency_hz,
                    "acf_first_peak_correlation": diag.acf_first_peak_correlation,
                    "spectral_peaks_json": diag.spectral_peaks_json(),
                }
            )
            osc_rows.append(row)

    write_csv(zero_rows, out_dir / "zero_noise_attractor_summary.csv")
    write_csv(osc_rows, out_dir / "oscillation_diagnostics.csv")

    # 2) Recompute the 60-pA endpoint summaries with explicit systematic-position diagnostics.
    attr_results = load_regime(results_dir, "attractor")
    noisy_rows: list[dict] = []
    for condition, results in attr_results.items():
        for result in results:
            summary, _ = summarize_attractor(result, cfg.simulation, cfg.analysis)
            noisy_rows.append(asdict(summary))
    write_csv(noisy_rows, out_dir / "noise60_attractor_summary.csv")

    stats: dict = {
        "zero_noise_attractor": endpoint_summary(zero_rows, cfg.analysis.bootstrap_repeats, 6400),
        "noise60_attractor_secondary": endpoint_summary(noisy_rows, cfg.analysis.bootstrap_repeats, 6500),
        "oscillation_diagnostics": {},
        "notes": {
            "attractor": (
                "The phase-4 primary random-drift endpoint remains the 60-pA across-trial PAV SD. "
                "The mean/circular-mean endpoint analyses here are secondary diagnostics for deterministic/systematic drift."
            ),
            "oscillation": (
                "ACF frequencies are diagnostic cross-checks for harmonic ambiguity; they do not replace the pre-specified PSD endpoint."
            ),
        },
    }

    for condition in CONDITIONS:
        rows = [r for r in osc_rows if r["condition"] == condition]
        for key in ("acf_best_frequency_hz", "acf_first_peak_frequency_hz"):
            vals = np.asarray([r[key] for r in rows], float)
            stats["oscillation_diagnostics"].setdefault(condition, {})[f"mean_{key}"] = float(np.mean(vals))
            stats["oscillation_diagnostics"][condition][f"sd_{key}"] = float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan")
        stats["oscillation_diagnostics"][condition]["spectral_peaks_seed0"] = next(
            (json.loads(r["spectral_peaks_json"]) for r in rows if int(r["seed"]) == 0), []
        )

    for other in ("baseline", "static_gain"):
        a, b = paired_by_seed(osc_rows, "dynamic", other, "acf_best_frequency_hz")
        stats["oscillation_diagnostics"][f"dynamic_minus_{other}_acf_best_frequency"] = asdict(
            paired_bootstrap_mean_difference(a, b, repeats=cfg.analysis.bootstrap_repeats, seed=6700 + len(other))
        )

    write_json(stats, out_dir / "phase6_followup.json")
    print(f"Wrote phase-6 follow-up analysis to {out_dir}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
