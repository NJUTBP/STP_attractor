#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.flow import summarize_flow, unwrap_axial_trace
from analysis.pav import summarize_attractor
from analysis.theta import analyze_theta
from config import load_config
from experiment import run_regime
from io_utils import load_result, write_json

CONDITIONS = ("dynamic", "baseline", "static_gain")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Five-seed zero-noise pilot after the corrected I->E connectivity kernel. "
            "Runs dynamic/baseline/static-gain, re-estimates p_match, checks stationarity, "
            "and tests whether dynamic STF still produces theta."
        )
    )
    p.add_argument("--config", default=str(ROOT / "config" / "phase4_causal.toml"))
    p.add_argument("--output", default=str(ROOT / "results" / "phase6_corrected_pilot"))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    return p.parse_args()


def _condition_paths(out: Path, condition: str) -> list[Path]:
    return sorted((out / "zero_noise" / condition).glob("seed_*.npz"))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(x) for x in args.seeds)
    if not seeds:
        raise SystemExit("At least one seed is required.")

    p_match = run_regime(
        cfg,
        name="zero_noise",
        noise_pa=0.0,
        seeds=seeds,
        output_dir=out,
        workers=max(1, args.workers),
    )

    report: dict = {
        "seeds": list(seeds),
        "p_match_zero_noise": float(p_match),
        "conditions": {},
    }

    stationary_flags = []
    for condition in CONDITIONS:
        flows = []
        thetas = []
        localization_failures = 0
        for path in _condition_paths(out, condition):
            r = load_result(path)
            attr, pav = summarize_attractor(r, cfg.simulation, cfg.analysis)
            localization_failures += int(attr.localization_failure)
            flow = summarize_flow(
                unwrap_axial_trace(
                    pav,
                    target_axis_deg=cfg.analysis.target_axis_deg,
                    start_ms=1000.0,
                )
            )
            flows.append(flow)
            theta, _ = analyze_theta(r, cfg.simulation, cfg.analysis)
            thetas.append(theta)

        late_v = np.asarray([f.velocity_6_8_deg_s for f in flows], float)
        shift = np.asarray([f.shift_1_8_deg for f in flows], float)
        freq = np.asarray([t.dominant_frequency_hz for t in thetas], float)
        power = np.asarray([t.normalized_peak_power for t in thetas], float)
        theta_band = np.asarray([t.dominant_in_theta_band for t in thetas], bool)

        cond = {
            "n": len(flows),
            "localization_failures": int(localization_failures),
            "mean_velocity_6_8_deg_s": float(np.mean(late_v)),
            "mean_absolute_velocity_6_8_deg_s": float(np.mean(np.abs(late_v))),
            "mean_shift_1_8_deg": float(np.mean(shift)),
            "mean_absolute_shift_1_8_deg": float(np.mean(np.abs(shift))),
            "max_absolute_shift_1_8_deg": float(np.max(np.abs(shift))),
            "mean_dominant_frequency_hz": float(np.mean(freq)),
            "sd_dominant_frequency_hz": float(np.std(freq, ddof=1)) if freq.size > 1 else 0.0,
            "mean_normalized_peak_power": float(np.mean(power)),
            "theta_band_fraction": float(np.mean(theta_band)),
        }
        # Operational guard only.  Persistent pre-fix flow produced ~33-103 deg
        # of 1->8 s shift; a <10 deg mean absolute shift is intentionally lenient
        # and is used only to decide whether a full rerun is sensible.
        cond["stationary_enough_for_full_rerun"] = bool(
            localization_failures == 0 and cond["mean_absolute_shift_1_8_deg"] < 10.0
        )
        stationary_flags.append(cond["stationary_enough_for_full_rerun"])
        report["conditions"][condition] = cond

    dynamic_theta_ok = report["conditions"]["dynamic"]["theta_band_fraction"] >= 0.8
    report["decision_guard"] = {
        "stationarity_rule": "No localization failures and mean |PAV shift 1->8 s| < 10 deg for each condition.",
        "dynamic_theta_rule": "At least 80% of dynamic-STF pilot seeds have dominant frequency in 4-12 Hz.",
        "all_conditions_stationary_enough": bool(all(stationary_flags)),
        "dynamic_theta_reproduced": bool(dynamic_theta_ok),
        "recommended_to_rerun_full_phase4": bool(all(stationary_flags) and dynamic_theta_ok),
        "note": (
            "This is a low-cost implementation/operating-point gate, not publication-grade inference. "
            "If it passes, rerun the pre-specified full causal experiment in a fresh results directory."
        ),
    }

    write_json(report, out / "pilot_summary.json")
    print(f"Wrote corrected pilot to {out}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
