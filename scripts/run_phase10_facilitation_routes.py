#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from analysis.facilitation_coupling import analyze_phase10_facilitation_coupling
from config import load_config, load_phase8_sweep
from experiment.facilitation_coupling import run_phase10_facilitation_coupling_map
from io_utils import write_json
from plot.facilitation_coupling import plot_phase10_facilitation_coupling
from utils import git_revision


def args()->argparse.Namespace:
    p=argparse.ArgumentParser(description="Run Phase-10 facilitation-route factorial mechanism map.")
    p.add_argument("--base-config",default=str(ROOT/"config"/"phase4_causal.toml"))
    p.add_argument("--sweep-config",default=str(ROOT/"config"/"phase8_mechanism_map.toml"))
    p.add_argument("--source-phase8",default=str(ROOT/"results"/"phase8_mechanism_map"))
    p.add_argument("--source-phase9",default=str(ROOT/"results"/"phase9_gain_rescued_depletion"))
    p.add_argument("--output",default=str(ROOT/"results"/"phase10_facilitation_routes"))
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--force",action="store_true")
    p.add_argument("--analysis-only",action="store_true")
    return p.parse_args()


def main()->None:
    a=args(); base=load_config(a.base_config); sweep=load_phase8_sweep(a.sweep_config)
    combos=len(sweep.combos); seeds=len(sweep.seeds); modes=3
    expected=combos*seeds + 2*combos*seeds*modes + 2*combos*seeds*modes
    print(f"Phase-10 combinations: {combos}; seeds/combo: {seeds}; new hybrid modes: {modes}")
    print(f"Expected new simulations: {expected} (Full-u probes + 2 calibration waves + final 0/60-pA runs)")
    if a.dry_run:
        return
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    write_json({
        "git_revision":git_revision(ROOT),
        "source_phase8":str(Path(a.source_phase8).resolve()),
        "source_phase9":str(Path(a.source_phase9).resolve()),
    },out/"run_metadata.json")
    if not a.analysis_only:
        run_phase10_facilitation_coupling_map(
            base,sweep,
            source_phase8_dir=a.source_phase8,
            source_phase9_dir=a.source_phase9,
            output_dir=out,
            workers=max(1,a.workers),
            resume=not a.force,
        )
    summary=analyze_phase10_facilitation_coupling(
        base,sweep,
        source_phase8_dir=a.source_phase8,
        source_phase9_dir=a.source_phase9,
        results_dir=out,
    )
    figures=plot_phase10_facilitation_coupling(out,sweep)
    print("Phase-10 complete.")
    print(summary)
    print(f"Wrote {len(figures)} figures to {out/'figures'}")


if __name__=="__main__":
    main()
