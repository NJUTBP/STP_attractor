#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.mechanism_map import analyze_phase8_mechanism_map
from config import load_config, load_phase8_sweep
from experiment.mechanism_map import run_phase8_mechanism_map
from io_utils import write_json
from plot.mechanism_map import plot_phase8_mechanism_map
from utils import git_revision


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the large Phase-8 STP time-scale mechanism map.")
    p.add_argument("--base-config", default=str(ROOT / "config" / "phase4_causal.toml"))
    p.add_argument("--sweep-config", default=str(ROOT / "config" / "phase8_mechanism_map.toml"))
    p.add_argument("--output", default=str(ROOT / "results" / "phase8_mechanism_map"))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dry-run", action="store_true", help="Print grid/run count without simulating.")
    p.add_argument("--force", action="store_true", help="Recompute existing NPZ files instead of resuming.")
    p.add_argument("--analysis-only", action="store_true", help="Skip simulation and regenerate analysis/figures.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = load_config(args.base_config)
    sweep = load_phase8_sweep(args.sweep_config)
    out = Path(args.output)
    print(f"Phase-8 grid: {len(sweep.tau_facilitation_ms)} x {len(sweep.tau_recovery_ms)} = {len(sweep.combos)} combinations")
    print(f"Paired seeds per combination: {len(sweep.seeds)}")
    print(f"Expected simulations: {sweep.expected_run_count()}")
    if args.dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        {
            "git_revision": git_revision(ROOT),
            "base_config": str(Path(args.base_config).resolve()),
            "sweep_config": str(Path(args.sweep_config).resolve()),
        },
        out / "run_metadata.json",
    )
    if not args.analysis_only:
        run_phase8_mechanism_map(
            base,
            sweep,
            output_dir=out,
            workers=max(1, args.workers),
            resume=not args.force,
        )
    summary = analyze_phase8_mechanism_map(base, sweep, results_dir=out)
    figures = plot_phase8_mechanism_map(out, sweep)
    print("Phase-8 analysis complete.")
    print("Current-parameter anchor:")
    print(summary.get("current_parameter_anchor"))
    print("Theta-region fractions:", summary.get("theta_region_fraction"))
    print("Stationary-region fractions:", summary.get("stationary_region_fraction"))
    print(f"Wrote {len(figures)} figures to {out / 'figures'}")


if __name__ == "__main__":
    main()
