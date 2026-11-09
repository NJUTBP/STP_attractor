#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.phase13_confirmation import analyze_phase13_attractor_confirmation
from config import load_config, load_phase13_confirmatory
from experiment.phase13_confirmation import load_frozen_controls, run_phase13_attractor_confirmation


def main() -> None:
    p = argparse.ArgumentParser(description="Phase-13 long-horizon attractor confirmation (frozen zero-noise gain).")
    p.add_argument("--base-config", default=str(ROOT / "config" / "phase4_causal.toml"))
    p.add_argument("--grid", choices=("anchor", "full"), default="anchor")
    p.add_argument("--confirmatory-config", default="")
    p.add_argument("--phase8-results", default="")
    p.add_argument("--output", default=str(ROOT / "results" / "phase13_attractor_confirmation"))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--analysis-only", action="store_true")
    args = p.parse_args()

    cfg_path = args.confirmatory_config or str(
        ROOT / ("config/phase13_full_grid.toml" if args.grid == "full" else "config/phase13_anchor.toml")
    )
    base = load_config(args.base_config)
    confirm = load_phase13_confirmatory(cfg_path)
    phase8 = args.phase8_results or None
    if args.grid == "full" and not phase8:
        default8 = ROOT / "results" / "phase8_mechanism_map"
        phase8 = str(default8) if default8.exists() else None
        if phase8 is None:
            raise SystemExit("Full grid requires --phase8-results pointing at the Phase-8 mechanism map.")

    controls, ctx = load_frozen_controls(confirm, base, phase8)
    print(f"Phase-13 grid: {args.grid}")
    print(f"Combos: {len(confirm.combos)}  modes: {list(confirm.temporal_modes)}")
    print(f"Seeds: {len(confirm.seeds)}  noise: {list(confirm.noise_levels_pa)}")
    print(f"Expected simulations: {confirm.expected_run_count()}")
    print("Frozen p_match:", controls)
    print("p_match source:", {cid: info.get("source") for cid, info in ctx.items()})
    if args.dry_run:
        return

    out = Path(args.output)
    if not args.analysis_only:
        run_phase13_attractor_confirmation(
            base,
            confirm,
            output_dir=out,
            phase8_results=phase8,
            workers=max(1, args.workers),
            resume=not args.force,
        )
    summary = analyze_phase13_attractor_confirmation(base, confirm, results_dir=out)
    print("Primary anchor:")
    print(summary.get("primary_confirmatory_anchor"))


if __name__ == "__main__":
    main()
