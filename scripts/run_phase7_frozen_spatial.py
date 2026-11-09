#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.frozen_spatial import analyze_frozen_spatial
from config import load_config
from experiment.frozen_spatial import run_frozen_spatial
from io_utils import read_json, write_json
from plot.frozen_spatial import plot_frozen_profile, plot_theta_psd_with_frozen
from utils import git_revision


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the phase-7 frozen-spatial-profile E→I control.")
    p.add_argument("--config", default=str(ROOT / "config" / "phase4_causal.toml"))
    p.add_argument("--source", default=str(ROOT / "results" / "phase6_corrected_full"))
    p.add_argument("--output", default=str(ROOT / "results" / "phase7_frozen_spatial"))
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    source = Path(args.source)
    out = Path(args.output)
    if not (source / "theta" / "dynamic").exists():
        raise SystemExit(
            f"Source corrected results are missing theta/dynamic NPZ files: {source}. "
            "Point --source to the full corrected phase-6 results directory, not an analysis-only ZIP export."
        )

    out.mkdir(parents=True, exist_ok=True)
    write_json(
        {
            "git_revision": git_revision(ROOT),
            "config_path": str(Path(args.config).resolve()),
            "source_results": str(source.resolve()),
        },
        out / "run_metadata.json",
    )

    manifest = run_frozen_spatial(
        cfg,
        source_dir=source,
        output_dir=out,
        workers=max(1, args.workers),
    )
    summary, psd_by_condition = analyze_frozen_spatial(
        cfg,
        source_dir=source,
        output_dir=out,
    )

    with np.load(out / "frozen_profile.npz", allow_pickle=False) as z:
        theta_e = z["theta_e_deg"].copy()
        profile = z["efficacy_by_e"].copy()
    scalar_p_match = manifest["profile"].get("source_p_match_theta")
    plot_frozen_profile(
        theta_e,
        profile,
        baseline=cfg.stf.u_baseline,
        scalar_p_match=scalar_p_match,
        output=out / "figures" / "frozen_profile.png",
    )
    plot_theta_psd_with_frozen(
        psd_by_condition,
        out / "figures" / "theta_psd_with_frozen.png",
        max_hz=cfg.analysis.max_psd_hz,
    )

    print(f"Wrote frozen-spatial control to {out}")
    print(__import__("json").dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
