#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config
from io_utils import save_result
from model import simulate


def main() -> None:
    p = argparse.ArgumentParser(description="Run one diagnostic GC-STP simulation.")
    p.add_argument("--config", default=str(ROOT / "config" / "phase4_causal.toml"))
    p.add_argument("--condition", choices=["dynamic", "baseline", "static_gain"], default="dynamic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--static-gain", type=float)
    p.add_argument("--output", default=str(ROOT / "results" / "single_run.npz"))
    args = p.parse_args()
    cfg = load_config(args.config)
    result = simulate(
        cfg,
        condition=args.condition,
        seed=args.seed,
        noise_pa=args.noise,
        static_gain=args.static_gain,
    )
    save_result(result, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
