#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.phase12_attractor import analyze_phase12_attractor
from config import load_config
from experiment.phase12_attractor import iter_module_a_jobs, iter_module_b_jobs, run_phase12


def main() -> None:
    p = argparse.ArgumentParser(description="Phase-12 attractor: paired short diffusion + 11.25° shift.")
    p.add_argument("--base-config", default=str(ROOT / "config" / "phase4_causal.toml"))
    p.add_argument("--output", default=str(ROOT / "results" / "phase12_attractor"))
    p.add_argument("--checkpoint-dir", default="", help="Reuse Phase-11 checkpoints if set.")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--analysis-only", action="store_true")
    p.add_argument("--skip-a", action="store_true")
    p.add_argument("--skip-b", action="store_true")
    args = p.parse_args()

    a_jobs = iter_module_a_jobs()
    b_jobs = iter_module_b_jobs()
    print(f"Module A diffusion jobs: {len(a_jobs)}")
    print(f"Module B 11.25° jobs: {len(b_jobs)}")
    if args.dry_run:
        return

    cfg = load_config(args.base_config)
    out = Path(args.output)
    ckpt = args.checkpoint_dir or None
    if not args.analysis_only:
        run_phase12(
            output_dir=out,
            config_path=str(Path(args.base_config).resolve()),
            checkpoint_dir=ckpt,
            workers=max(1, args.workers),
            resume=not args.force,
            run_module_a=not args.skip_a,
            run_module_b=not args.skip_b,
        )
    summary = analyze_phase12_attractor(cfg, out)
    print("Wrote", out / "analysis" / "phase12_summary.json")
    for row in summary.get("module_a_summary", []):
        print(
            f"  A {row['condition']:16s} n={row['noise_pa']:>5} "
            f"D={row['D_short_median']:.4g} |dq|2s={row['abs_dq_2000ms_median']:.3f}"
        )
    for row in summary.get("module_b_summary", []):
        print(
            f"  B {row['condition']:16s} {row['shift_kind']:16s} "
            f"R4={row['R_4000ms_median']:+.3f}"
        )


if __name__ == "__main__":
    main()
