#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.phase15_tether_confirmation import analyze_phase15_tether_confirmation
from config import load_config
from experiment.phase15_tether_confirmation import iter_phase15_branches, run_phase15_tether_confirmation


def main() -> None:
    p = argparse.ArgumentParser(description="Phase-15 independent-seed confirmation of the u-carried attractor tether mechanism.")
    p.add_argument("--base-config", default=str(ROOT / "config" / "phase4_causal.toml"))
    p.add_argument("--output", default=str(ROOT / "results" / "phase15_tether_confirmation"))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--analysis-only", action="store_true")
    p.add_argument("--bootstrap-repeats", type=int, default=10000)
    args = p.parse_args()

    branches = iter_phase15_branches()
    print(f"Phase-15 branches: {len(branches)}")
    print("Confirmatory scope: independent seeds 200-215; anchor only; no parameter grid; no noise sweep.")
    print("Primary delta/readout: 11.25 deg, R_2000ms.")
    if args.dry_run:
        return

    cfg = load_config(args.base_config)
    out = Path(args.output)
    if not args.analysis_only:
        run_phase15_tether_confirmation(
            cfg,
            output_dir=out,
            config_path=str(Path(args.base_config).resolve()),
            workers=max(1, args.workers),
            resume=not args.force,
        )
    summary = analyze_phase15_tether_confirmation(cfg, out, bootstrap_repeats=args.bootstrap_repeats)
    print("Wrote", out / "analysis" / "phase15_tether_confirmation_summary.json")
    print("Primary decisions:", summary.get("primary_decisions", {}))


if __name__ == "__main__":
    main()
