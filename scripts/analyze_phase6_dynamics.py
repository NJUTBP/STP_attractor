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

from analysis.flow import summarize_flow, unwrap_axial_trace
from analysis.pav import axial_displacement_deg, circular_axis_sd_deg, summarize_attractor
from analysis.statistics import paired_bootstrap_mean_absolute_displacement_difference, paired_bootstrap_sd_difference
from config import load_config
from io_utils import load_result, write_json
from plot.dynamics import plot_plfp_acf, plot_plfp_segment, plot_unwrapped_pav

CONDITIONS = ("dynamic", "baseline", "static_gain")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose deterministic zero-noise attractor flow using existing NPZ results.")
    p.add_argument("--config", default=str(ROOT / "config" / "phase4_causal.toml"))
    p.add_argument("--results", default=str(ROOT / "results" / "phase4_causal"))
    return p.parse_args()


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def mean_sd(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    return {
        "mean": float(np.mean(x)) if x.size else float("nan"),
        "sd": float(np.std(x, ddof=1)) if x.size > 1 else float("nan"),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    results_dir = Path(args.results)
    out_dir = results_dir / "analysis_phase6_dynamics"
    fig_dir = results_dir / "figures_phase6_dynamics"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    traces: dict[str, list] = {c: [] for c in CONDITIONS}
    seed0_results = {}

    for condition in CONDITIONS:
        paths = sorted((results_dir / "theta" / condition).glob("seed_*.npz"))
        if not paths:
            raise FileNotFoundError(f"No zero-noise raw NPZ files found for theta/{condition}")
        for path in paths:
            result = load_result(path)
            _, pav = summarize_attractor(result, cfg.simulation, cfg.analysis)
            flow_trace = unwrap_axial_trace(pav, target_axis_deg=cfg.analysis.target_axis_deg, start_ms=1000.0)
            flow = summarize_flow(flow_trace)
            row = {"seed": result.seed, "condition": condition, "noise_pa": result.noise_pa} | asdict(flow)
            rows.append(row)
            traces[condition].append(flow_trace)
            if result.seed == 0:
                seed0_results[condition] = result

    write_csv(rows, out_dir / "zero_noise_flow_summary.csv")

    # Exploratory paired endpoint diagnostic: because the same seeds were used at 0 and 60 pA,
    # compare each 60-pA endpoint with its own zero-noise counterfactual rather than with the cue.
    # This removes the large condition-specific deterministic offset at the endpoint.
    noise_rows: list[dict] = []
    for condition in CONDITIONS:
        zero_paths = sorted((results_dir / "theta" / condition).glob("seed_*.npz"))
        noisy_paths = sorted((results_dir / "attractor" / condition).glob("seed_*.npz"))
        zero_by_seed = {load_result(p).seed: load_result(p) for p in zero_paths}
        noisy_by_seed = {load_result(p).seed: load_result(p) for p in noisy_paths}
        for seed in sorted(set(zero_by_seed) & set(noisy_by_seed)):
            zsum, _ = summarize_attractor(zero_by_seed[seed], cfg.simulation, cfg.analysis)
            nsum, _ = summarize_attractor(noisy_by_seed[seed], cfg.simulation, cfg.analysis)
            delta = float(axial_displacement_deg(nsum.pav_deg_at_target, zsum.pav_deg_at_target))
            noise_rows.append({
                "seed": seed,
                "condition": condition,
                "zero_noise_pav_deg": zsum.pav_deg_at_target,
                "noise60_pav_deg": nsum.pav_deg_at_target,
                "noise_induced_endpoint_delta_deg": delta,
                "absolute_noise_induced_endpoint_delta_deg": abs(delta),
            })
    write_csv(noise_rows, out_dir / "noise_induced_endpoint_summary.csv")

    stats = {"zero_noise_flow": {}, "noise_induced_endpoint": {}, "interpretation": {}}
    for condition in CONDITIONS:
        cr = [r for r in rows if r["condition"] == condition]
        stats["zero_noise_flow"][condition] = {
            "n": len(cr),
            "velocity_1_3_deg_s": mean_sd([r["velocity_1_3_deg_s"] for r in cr]),
            "velocity_3_5_deg_s": mean_sd([r["velocity_3_5_deg_s"] for r in cr]),
            "velocity_5_7_deg_s": mean_sd([r["velocity_5_7_deg_s"] for r in cr]),
            "velocity_6_8_deg_s": mean_sd([r["velocity_6_8_deg_s"] for r in cr]),
            "shift_1_7_deg": mean_sd([r["shift_1_7_deg"] for r in cr]),
            "shift_1_8_deg": mean_sd([r["shift_1_8_deg"] for r in cr]),
            "displacement_7s_deg_unwrapped": mean_sd([r["displacement_7s_deg"] for r in cr]),
            "displacement_8s_deg_unwrapped": mean_sd([r["displacement_8s_deg"] for r in cr]),
        }

    for condition in CONDITIONS:
        x = np.asarray([r["noise_induced_endpoint_delta_deg"] for r in noise_rows if r["condition"] == condition], float)
        stats["noise_induced_endpoint"][condition] = {
            "n_overlap": int(x.size),
            "circular_sd_deg": circular_axis_sd_deg(x),
            "mean_absolute_delta_deg": float(np.mean(np.abs(x))) if x.size else float("nan"),
            "mean_signed_delta_deg": float(np.mean(x)) if x.size else float("nan"),
        }

    maps = {c: {int(r["seed"]): r for r in noise_rows if r["condition"] == c} for c in CONDITIONS}
    for j, other in enumerate(("baseline", "static_gain")):
        seeds = sorted(set(maps["dynamic"]) & set(maps[other]))
        a = np.asarray([maps["dynamic"][k]["noise_induced_endpoint_delta_deg"] for k in seeds], float)
        b = np.asarray([maps[other][k]["noise_induced_endpoint_delta_deg"] for k in seeds], float)
        stats["noise_induced_endpoint"][f"dynamic_minus_{other}_sd"] = asdict(
            paired_bootstrap_sd_difference(a, b, repeats=cfg.analysis.bootstrap_repeats, seed=7200 + j)
        )
        stats["noise_induced_endpoint"][f"dynamic_minus_{other}_mean_absolute_delta"] = asdict(
            paired_bootstrap_mean_absolute_displacement_difference(
                a, b, repeats=cfg.analysis.bootstrap_repeats, seed=7300 + j
            )
        )

    stats["interpretation"] = {
        "primary_question": "Does zero-noise PAV motion persist late in the run, or settle after an early transient?",
        "late_velocity_metric": "velocity_6_8_deg_s from a linear fit to unwrapped axial PAV between 6 and 8 s.",
        "noise_induced_endpoint_note": (
            "Exploratory post-hoc diagnostic using only seeds shared by the 0-pA and 60-pA runs. "
            "For each condition/seed it measures the axial endpoint difference q60(7s)-q0(7s), thereby "
            "comparing noise against that seed's deterministic zero-noise trajectory rather than against the cue."
        ),
        "warning": (
            "If late zero-noise velocity remains appreciably nonzero, this operating point is a traveling/flow state rather "
            "than a stationary attractor. In that case, phase-4 60-pA PAV SD should not yet be interpreted as pure random-drift stability."
        ),
    }
    write_json(stats, out_dir / "phase6_dynamics.json")

    plot_unwrapped_pav(traces, fig_dir / "zero_noise_unwrapped_pav.png")
    if all(c in seed0_results for c in CONDITIONS):
        plot_plfp_segment(seed0_results, fig_dir / "zero_noise_plfp_seed0.png")
        plot_plfp_acf(seed0_results, fig_dir / "zero_noise_plfp_acf_seed0.png")

    print(f"Wrote phase-6 dynamics analysis to {out_dir}")
    print(f"Wrote phase-6 dynamics figures to {fig_dir}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
