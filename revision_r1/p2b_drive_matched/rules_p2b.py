"""P2b pre-registered definitions shared by the runner, the analysis and the tests (PREREGISTRATION.md).

Continuous-gain convention throughout (the paper's model, frozen simulator via run_segment_p1b, which is
bitwise identical for convention="continuous").  Only the calibration statistic changes:

  C1 (paper)  drive-weighted mean efficacy   sum_t,i w_i s_i p_i / sum_t,i w_i s_i
  C3 (P2b)    total E->I drive               D = < (1/N_I) sum_i w_i s_i(t) p_i(t) >_t      (as in P1b)

Calibrated parameter: k (temporal_gain_scale) of the output-only and release-only models, p (static_gain) of the
static control.  The full model is never re-run.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # Python <= 3.10
    import tomli as tomllib

CODE_ROOT = Path(__file__).resolve().parents[2]

# model -> (simulator condition, calibrated keyword of run_segment_p1b)
MODELS = {
    "output_only": ("u_output_only", "temporal_gain_scale"),
    "release_only": ("u_release_only", "temporal_gain_scale"),
    "static": ("static_gain", "static_gain"),
}
CAL_MODELS = ("output_only", "release_only", "static")
ROUTE_MODELS = ("output_only", "release_only")

# Values already reported for the C1-matched models (context in tables and figures only; no rule uses them).
C1_PUBLISHED_HZ = {
    "full": (6.4453125, "Phase 6, 10 trials (Fig. S1)"),
    "static": (14.375, "Phase 6, 10 trials (main text 14.38 Hz)"),
    "output_only": (11.181641, "Phase 10 tf_1000_tr_20, 4 trials"),
    "release_only": (6.347656, "Phase 10 tf_1000_tr_20, 4 trials"),
}
P1B_EVENT_STATIC_HZ = (8.22265625, "P1b drive-matched event-scaled static, 10 trials")


@dataclass(frozen=True)
class P2bConfig:
    base_config: Path
    target_dir: Path
    full_frequency_dir: Path
    wave0_dirs: dict
    c1_static_dir: Path
    output: Path
    seeds_calibration: tuple
    seeds_frequency: tuple
    initial: dict
    calibration: dict
    decision: dict
    statistics: dict


def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else CODE_ROOT / p


def load_p2b_config(path=None) -> P2bConfig:
    path = Path(path) if path else Path(__file__).with_name("p2b_config.toml")
    raw = tomllib.load(path.open("rb"))
    P = raw["paths"]
    return P2bConfig(
        base_config=_resolve(P["base_config"]),
        target_dir=_resolve(P["target_dir"]),
        full_frequency_dir=_resolve(P["full_frequency_dir"]),
        wave0_dirs={"output_only": _resolve(P["wave0_output_only"]),
                    "release_only": _resolve(P["wave0_release_only"])},
        c1_static_dir=_resolve(P["c1_static_dir"]),
        output=_resolve(P["output"]),
        seeds_calibration=tuple(int(s) for s in raw["seeds"]["calibration"]),
        seeds_frequency=tuple(int(s) for s in raw["seeds"]["frequency"]),
        initial=dict(raw["initial"]), calibration=dict(raw["calibration"]),
        decision=dict(raw["decision"]), statistics=dict(raw["statistics"]))


def run_kwargs(model: str, param: float) -> dict:
    """Keyword arguments of run_segment_p1b (continuous convention) for one calibrated model."""
    cond, key = MODELS[model]
    return {"convention": "continuous", "condition": cond, key: float(param)}


def seed_path(d, s: int) -> Path:
    return Path(d) / f"seed_{int(s):04d}.npz"


# ------------------------------------------------------------------------------------------- drive (C3)
def window_steps(cfg) -> int:
    """Number of integration steps inside the gain-match window of a full trial (same products as the simulator)."""
    dt = cfg.simulation.dt_ms
    a = cfg.analysis
    steps = int(round(cfg.simulation.duration_ms / dt))
    t = np.arange(steps) * dt
    return int(np.count_nonzero((a.gain_match_start_ms <= t) & (t <= a.gain_match_end_ms)))


def drive_from_frozen_result(result, cfg) -> tuple[float, float]:
    """(numerator, denominator) of D for a frozen-simulator file of a temporal condition.
    In the continuous convention the C3 numerator equals gain_match_numerator exactly (P1b QC1)."""
    if result.condition == "static_gain":
        raise ValueError("frozen static files do not store the C3 numerator; static drive needs a P2b run")
    return float(result.gain_match_numerator), float(window_steps(cfg)) * float(cfg.network.ni)


def drive_of_paths(paths, cfg) -> float:
    """Pooled C3 drive of several trials: P2b/P1b files carry the accumulator, frozen files the p_match numerator."""
    from io_utils import load_result
    from revision_r1.common.simulator_event import load_p1b_extras
    num = den = 0.0
    for p in paths:
        ex = load_p1b_extras(p)
        if "drive_num" in ex:
            num += float(ex["drive_num"])
            den += float(ex["drive_steps"]) * float(ex["ni"])
        else:
            n, d = drive_from_frozen_result(load_result(p), cfg)
            num += n
            den += d
    if den <= 0:
        raise RuntimeError("no drive samples in the gain-match window")
    return num / den


def c1_efficacy(result) -> float:
    """C1 statistic of one trial: gain_match ratio (temporal conditions) or the fixed efficacy (static)."""
    if result.condition == "static_gain":
        return float(result.static_gain)
    den = float(result.gain_match_denominator)
    return float(result.gain_match_numerator) / den if den > 0 else float("nan")


def check_existing_run(result, condition: str, seed: int, k_expected: float | None = None,
                       rel_tol: float = 1e-6) -> None:
    """Raise if an existing file is not the continuous 0-pA run that the protocol needs."""
    if result.condition != condition:
        raise ValueError(f"condition {result.condition} != {condition}")
    if int(result.seed) != int(seed):
        raise ValueError(f"seed {result.seed} != {seed}")
    if float(result.noise_pa) != 0.0:
        raise ValueError(f"noise {result.noise_pa} pA != 0")
    k = float(result.temporal_gain_scale)
    if k_expected is not None and abs(k - k_expected) > rel_tol * abs(k_expected):
        raise ValueError(f"temporal_gain_scale {k} differs from {k_expected}")


# ------------------------------------------------------------------------------------------- calibration core
def calibrate_core(items: list[dict], run_wave, cal: dict) -> list[dict]:
    """Lock-step calibration shared by the runner and the tests.

    items: dicts with model, param (next value to run), target, history [(wave, param, D, rel_error, dir)].
    run_wave(item, wave) -> (D, dir) runs the calibration seeds at item["param"] and returns their pooled drive.
    Stops an item when rel_error < tolerance; otherwise the next value comes from next_param (P1b rule).
    """
    from revision_r1.common.drive_rules import next_param
    tol = float(cal["tolerance"])
    for it in items:
        it.setdefault("history", [])
        last = it["history"][-1] if it["history"] else None
        it["converged"] = bool(last and last["rel_error"] < tol)
        if it["converged"]:
            it["final_param"], it["converged_dir"] = last["param"], last["dir"]
        elif last is not None:
            pts = [(h["param"], h["D"]) for h in it["history"]]
            it["param"] = next_param(pts, it["target"], float(cal["max_step_factor"]))
    for wave in range(1, int(cal["max_waves"]) + 1):
        todo = [it for it in items if not it["converged"]]
        if not todo:
            break
        results = run_wave(todo, wave)                      # list of (D, dir) in the order of todo
        for it, (D, wdir) in zip(todo, results):
            rel = abs(D - it["target"]) / it["target"]
            it["history"].append(dict(wave=wave, param=float(it["param"]), D=float(D), rel_error=float(rel),
                                      dir=str(wdir)))
            if rel < tol:
                it.update(converged=True, final_param=float(it["param"]), converged_dir=str(wdir))
            else:
                pts = [(h["param"], h["D"]) for h in it["history"]]
                it["param"] = next_param(pts, it["target"], float(cal["max_step_factor"]))
    for it in items:
        if not it["converged"]:
            it.update(final_param=float(it["param"]), converged_dir=None)
    return items


# ------------------------------------------------------------------------------------------- verdict
def p2b_verdict(mean_hz: dict, theta_fraction: dict, valid: dict, decision: dict) -> dict:
    """Pre-registered verdict (PREREGISTRATION.md section 5).

    mean_hz: model -> mean dominant frequency over the 10 paired seeds (keys full, output_only, release_only, static).
    theta_fraction: model -> fraction of trials with a dominant frequency in (4, 12] Hz.
    valid: model -> calibration validity (final relative drive error <= validity).
    """
    out = {"verdict": None, "reasons": [], "static_label": None}
    bad = [m for m in ROUTE_MODELS if not valid.get(m, False)]
    if bad:
        out.update(verdict="NOT_EVALUABLE", reasons=[f"calibration_invalid:{m}" for m in bad])
    else:
        e_rel = abs(mean_hz["release_only"] - mean_hz["full"])
        e_out = abs(mean_hz["output_only"] - mean_hz["full"])
        c_rel = e_rel <= decision["release_max_abs_error_hz"]
        c_out = e_out >= decision["output_min_abs_error_hz"]
        out.update(release_abs_error_hz=float(e_rel), output_abs_error_hz=float(e_out),
                   criteria={"release_tracks_full": bool(c_rel), "output_departs": bool(c_out)})
        if c_rel and c_out:
            out["verdict"] = "ROBUST_TO_DRIVE_MATCHING"
        else:
            out["verdict"] = "DEPENDS_ON_MATCHING_CRITERION"
            if not c_out:
                out["reasons"].append("C_output: output-only within 2 Hz of full when matched on total drive")
            if not c_rel:
                out["reasons"].append("C_release: release-only more than 1 Hz from full when matched on total drive")
    if not valid.get("static", False):
        out["static_label"] = "STATIC_NOT_EVALUABLE"
    else:
        d_st = abs(mean_hz["static"] - mean_hz["full"])
        in_theta = theta_fraction.get("static", 0.0) >= decision["min_theta_fraction"]
        out["static_abs_difference_hz"] = float(d_st)
        if d_st < decision["static_min_abs_difference_hz"]:
            out["static_label"] = "STATIC_WITHIN_2HZ"
        elif in_theta:
            out["static_label"] = "STATIC_DEPARTS_IN_THETA"
        else:
            out["static_label"] = "STATIC_DEPARTS_OUTSIDE_THETA"
    return out
