#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.p0_msd import analyze_p0_msd
from config import load_config


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Analysis-only P0: full axial MSD(t) and early/late effective diffusion "
            "from existing Phase-12/13 raw simulation results."
        )
    )
    p.add_argument("--base-config", default=str(ROOT / "config" / "phase4_causal.toml"))
    p.add_argument("--phase12-results", default=str(ROOT / "results" / "phase12_attractor"))
    p.add_argument("--phase13-results", default=str(ROOT / "results" / "phase13_attractor_confirmation"))
    p.add_argument("--output", default=str(ROOT / "results" / "p0_msd"))
    p.add_argument("--phase13-scope", choices=("anchor", "all"), default="anchor")
    p.add_argument("--bootstrap-repeats", type=int, default=5000)
    p.add_argument("--skip-phase12", action="store_true")
    p.add_argument("--skip-phase13", action="store_true")
    args = p.parse_args()

    if args.skip_phase12 and args.skip_phase13:
        raise SystemExit("Nothing to analyze: both Phase 12 and Phase 13 were skipped.")
    if args.bootstrap_repeats < 100:
        raise SystemExit("--bootstrap-repeats must be >= 100")

    cfg = load_config(args.base_config)
    phase12 = None if args.skip_phase12 else args.phase12_results
    phase13 = None if args.skip_phase13 else args.phase13_results
    summary = analyze_p0_msd(
        cfg,
        phase12_results=phase12,
        phase13_results=phase13,
        output_dir=args.output,
        phase13_scope=args.phase13_scope,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    print("P0 analysis complete; no simulations were run.")
    print("Wrote", Path(args.output) / "p0_msd_summary.json")
    if "phase12" in summary:
        print("Phase-12 MSD/diffusion outputs:", Path(args.output) / "phase12")
    if "phase13" in summary:
        print("Phase-13 MSD/diffusion outputs:", Path(args.output) / "phase13")


if __name__ == "__main__":
    main()
