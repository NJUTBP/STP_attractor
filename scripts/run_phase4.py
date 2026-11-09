#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config
from experiment import run_phase4, run_preflight
from io_utils import write_json
from utils import git_revision


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the phase-4 E→I STF causal experiment.")
    p.add_argument("--config", default=str(ROOT / "config" / "phase4_causal.toml"))
    p.add_argument("--output", default=str(ROOT / "results" / "phase4_causal"))
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--workers", type=int, default=1, help="Parallel worker processes for full A/B/C runs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json({"git_revision": git_revision(ROOT), "config_path": str(Path(args.config).resolve())}, out / "run_metadata.json")

    if not args.skip_preflight:
        summary = run_preflight(cfg, out)
        print("Preflight:", summary)
        if args.preflight_only:
            return
        if not summary["passed"]:
            raise SystemExit(
                "Preflight failed: clean E→I-only dynamic-STF model did not satisfy both theta-band and localization checks. "
                "Inspect preflight outputs before running the full experiment."
            )

    manifest = run_phase4(cfg, out, workers=max(1, args.workers))
    print("Completed phase-4 experiment.")
    print("p_match attractor:", manifest["p_match_attractor"])
    print("p_match theta:", manifest["p_match_theta"])


if __name__ == "__main__":
    main()
