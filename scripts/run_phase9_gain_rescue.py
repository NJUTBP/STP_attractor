#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.gain_rescue import analyze_phase9_gain_rescue
from config import load_config, load_phase8_sweep
from experiment.gain_rescue import run_phase9_gain_rescued_depletion
from io_utils import write_json
from plot.gain_rescue import plot_phase9_gain_rescue
from utils import git_revision


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Phase-9 gain-rescued depletion mechanism reconstruction.")
    p.add_argument("--base-config", default=str(ROOT / "config" / "phase4_causal.toml"))
    p.add_argument("--sweep-config", default=str(ROOT / "config" / "phase8_mechanism_map.toml"))
    p.add_argument("--source", default=str(ROOT / "results" / "phase8_mechanism_map"), help="Completed Phase-8 results directory.")
    p.add_argument("--output", default=str(ROOT / "results" / "phase9_gain_rescued_depletion"))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--analysis-only", action="store_true")
    return p.parse_args()


def main() -> None:
    a = args(); base = load_config(a.base_config); sweep = load_phase8_sweep(a.sweep_config)
    expected = len(sweep.combos) * len(sweep.seeds) * 3
    print(f"Phase-9 combinations: {len(sweep.combos)}; seeds/combo: {len(sweep.seeds)}; expected new simulations: {expected}")
    if a.dry_run:
        return
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    write_json({"git_revision": git_revision(ROOT), "source_phase8": str(Path(a.source).resolve())}, out / "run_metadata.json")
    if not a.analysis_only:
        run_phase9_gain_rescued_depletion(base, sweep, source_phase8_dir=a.source, output_dir=out, workers=max(1, a.workers), resume=not a.force)
    summary = analyze_phase9_gain_rescue(base, sweep, source_phase8_dir=a.source, results_dir=out)
    figures = plot_phase9_gain_rescue(out, sweep)
    print("Phase-9 complete.")
    print(summary)
    print(f"Wrote {len(figures)} figures to {out / 'figures'}")


if __name__ == "__main__":
    main()
