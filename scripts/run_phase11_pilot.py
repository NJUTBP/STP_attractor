#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.phase11_pilot import analyze_phase11_anchor_pilot
from config import load_config
from experiment.phase11_pilot import PILOT_DELTA_DEG, iter_pilot_branches, run_phase11_anchor_pilot
from model.state import aligned_roll_bins


def main() -> None:
    p = argparse.ArgumentParser(description="Phase-11 anchor pilot: 248 zero-noise translation branches.")
    p.add_argument("--base-config", default=str(ROOT / "config" / "phase4_causal.toml"))
    p.add_argument("--output", default=str(ROOT / "results" / "phase11_anchor_pilot"))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--analysis-only", action="store_true")
    args = p.parse_args()

    branches = iter_pilot_branches()
    e_bins, i_bins = aligned_roll_bins(PILOT_DELTA_DEG, 1024, 256)
    print(f"Phase-11 anchor pilot branches: {len(branches)}")
    print(f"Δ={PILOT_DELTA_DEG}° → E roll {e_bins}, I roll {i_bins}")
    counts: dict[str, int] = {}
    for b in branches:
        counts[b.condition] = counts.get(b.condition, 0) + 1
    for k, n in counts.items():
        print(f"  {k}: {n}")
    if args.dry_run:
        return

    cfg = load_config(args.base_config)
    out = Path(args.output)
    if not args.analysis_only:
        run_phase11_anchor_pilot(
            cfg,
            output_dir=out,
            config_path=str(Path(args.base_config).resolve()),
            workers=max(1, args.workers),
            resume=not args.force,
        )
    summary = analyze_phase11_anchor_pilot(cfg, out)
    print("formula:", summary["formula_version"])
    print("Wrote", out / "analysis" / "branch_metrics.csv")
    print("Wrote", out / "analysis" / "condition_summary.csv")
    print("Wrote", out / "analysis" / "pair_metrics.csv")
    for row in summary["condition_medians"]:
        print(
            f"  {row['condition']:16s} {row['shift_kind']:16s} "
            f"R4={row['R_4000ms_median']:+.3f} AUC={row['auc_0p1_4s_median']:.2f}"
        )
    for cond, stats in summary["tri_tdi_by_condition"].items():
        print(
            f"  {cond:16s} TRI={stats['TRI_median']} "
            f"P(TRI>0)={stats['TRI_frac_positive']} TDI={stats['TDI_median']}"
        )


if __name__ == "__main__":
    main()

