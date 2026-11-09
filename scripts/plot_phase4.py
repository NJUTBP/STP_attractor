#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.pav import summarize_attractor
from analysis.theta import analyze_theta
from config import load_config
from io_utils import load_result
from plot import plot_efficacy_profiles, plot_endpoint_distributions, plot_mean_psd

CONDITIONS = ("dynamic", "baseline", "static_gain")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create the minimal phase-4 result figures.")
    p.add_argument("--config", default=str(ROOT / "config" / "phase4_causal.toml"))
    p.add_argument("--results", default=str(ROOT / "results" / "phase4_causal"))
    return p.parse_args()


def _load(results_dir: Path, regime: str):
    return {
        c: [load_result(p) for p in sorted((results_dir / regime / c).glob("seed_*.npz"))]
        for c in CONDITIONS
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    results_dir = Path(args.results)
    figs = results_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    attr = _load(results_dir, "attractor")
    attr_summaries = []
    for results in attr.values():
        for r in results:
            s, _ = summarize_attractor(r, cfg.simulation, cfg.analysis)
            attr_summaries.append(s)
    plot_endpoint_distributions(attr_summaries, figs / "attractor_endpoint_distribution.png")
    plot_efficacy_profiles(attr, figs / "ei_efficacy_profiles_attractor.png")

    theta = _load(results_dir, "theta")
    psd_by_condition = {c: [] for c in CONDITIONS}
    for condition, results in theta.items():
        for r in results:
            _, psd = analyze_theta(r, cfg.simulation, cfg.analysis)
            psd_by_condition[condition].append(psd)
    plot_mean_psd(psd_by_condition, figs / "theta_mean_psd.png", cfg.analysis.max_psd_hz)
    plot_efficacy_profiles(theta, figs / "ei_efficacy_profiles_theta.png")
    print(f"Wrote figures to {figs}")


if __name__ == "__main__":
    main()
