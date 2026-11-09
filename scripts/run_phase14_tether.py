#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.phase14_tether import analyze_phase14_tether
from config import load_config
from experiment.phase14_tether import iter_phase14_branches, run_phase14_tether


def main() -> None:
    p = argparse.ArgumentParser(description="Phase-14 anchor tether mechanism: restoring field + selective u/R state surgery.")
    p.add_argument("--base-config", default=str(ROOT / "config" / "phase4_causal.toml"))
    p.add_argument("--checkpoint-dir", default=str(ROOT / "results" / "phase11_anchor_pilot"))
    p.add_argument("--output", default=str(ROOT / "results" / "phase14_attractor_tether"))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--analysis-only", action="store_true")
    p.add_argument("--bootstrap-repeats", type=int, default=5000)
    args = p.parse_args()

    branches = iter_phase14_branches()
    print(f"Phase-14 branches: {len(branches)}")
    print("Scientific scope: anchor only; no parameter grid; no noise sweep.")
    if args.dry_run:
        return

    cfg = load_config(args.base_config)
    out = Path(args.output)
    if not args.analysis_only:
        run_phase14_tether(
            cfg,
            output_dir=out,
            config_path=str(Path(args.base_config).resolve()),
            checkpoint_dir=args.checkpoint_dir or None,
            workers=max(1, args.workers),
            resume=not args.force,
        )
    summary = analyze_phase14_tether(cfg, out, bootstrap_repeats=args.bootstrap_repeats)
    print("Wrote", out / "analysis" / "phase14_tether_summary.json")
    print("Surgery comparisons:")
    for row in summary.get("surgery_comparisons", []):
        print(row)


if __name__ == "__main__":
    main()
