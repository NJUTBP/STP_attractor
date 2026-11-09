#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.flow import summarize_flow, unwrap_axial_trace
from analysis.pav import summarize_attractor
from config import load_config
from model import simulate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Minimal zero-noise diagnostic after the I->E connectivity-kernel correction. "
            "Runs current dynamic, current frozen-baseline, and a reference-equivalent no-STP network."
        )
    )
    p.add_argument("--config", default=str(ROOT / "config" / "phase4_causal.toml"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output",
        default=str(ROOT / "results" / "phase6_connectivity_diagnostic.json"),
    )
    return p.parse_args()


def flow_for(result, cfg) -> dict:
    summary, pav = summarize_attractor(result, cfg.simulation, cfg.analysis)
    flow = summarize_flow(
        unwrap_axial_trace(pav, target_axis_deg=cfg.analysis.target_axis_deg, start_ms=1000.0)
    )
    return {
        "condition": result.condition,
        "seed": result.seed,
        "noise_pa": result.noise_pa,
        "pav_at_7s_deg": summary.pav_deg_at_target,
        "localization_ratio": summary.localization_ratio,
        "localization_failure": summary.localization_failure,
        **asdict(flow),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Current operating point, zero noise.  These two runs are sufficient to tell
    # whether the previously observed common-direction traveling flow survives the
    # corrected I->E connectivity construction.
    dynamic = simulate(cfg, condition="dynamic", seed=args.seed, noise_pa=0.0)
    baseline = simulate(cfg, condition="baseline", seed=args.seed, noise_pa=0.0)

    # Public-reference-equivalent operating point in the refactored simulator:
    # sigma_EI=30 deg, g_NMDA(E->I)=0.4 nS, reference cue half-width=22.5 deg,
    # and no E->I plastic efficacy multiplier (static multiplier = 1).
    ref_cfg = replace(
        cfg,
        network=replace(cfg.network, sigma_ei_deg=30.0, g_nmda_ei_ns=0.4),
        simulation=replace(cfg.simulation, cue_half_width_deg=22.5),
    )
    ref_equiv = simulate(
        ref_cfg,
        condition="static_gain",
        static_gain=1.0,
        seed=args.seed,
        noise_pa=0.0,
    )

    report = {
        "current_dynamic": flow_for(dynamic, cfg),
        "current_baseline": flow_for(baseline, cfg),
        "reference_equivalent_no_stp": flow_for(ref_equiv, ref_cfg),
        "interpretation": {
            "stationary_operational_threshold_deg_s": 1.0,
            "logic": [
                "If current dynamic and baseline become ~stationary after the kernel fix, the previous traveling flow was an implementation artifact; rerun the causal experiment before interpreting any old phase-4/6 results.",
                "If current networks still travel but the reference-equivalent network is ~stationary, the current operating-point parameters place the model in a traveling regime.",
                "If the reference-equivalent network also travels appreciably, inspect remaining differences from the public reference implementation (especially the integration scheme) before further scientific experiments.",
            ],
            "important": "Results produced before the connectivity-kernel correction should not be used as publication-grade causal evidence until reproduced with the corrected network.",
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote diagnostic to {output}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
