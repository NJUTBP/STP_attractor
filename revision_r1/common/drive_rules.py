"""P1b pre-registered definitions shared by the runner and the analysis (PREREGISTRATION.md sections 3-6)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # Python <= 3.10
    import tomli as tomllib

CODE_ROOT = Path(__file__).resolve().parents[2]

# model name -> (simulator condition, name of the calibrated parameter)
MODELS = {
    "full": ("dynamic", "nmda_scale"),
    "output_only": ("u_output_only", "event_gain"),
    "release_only": ("u_release_only", "event_gain"),
    "static": ("static_gain", "static_gain"),
}
ROUTE_MODELS = ("output_only", "release_only", "static")


@dataclass(frozen=True)
class P1bConfig:
    base_config: Path
    target_dir: Path
    output: Path
    seeds_calibration: tuple
    seeds_frequency: tuple
    seeds_restoration: tuple
    calibration: dict
    gate: dict
    decision: dict
    restoration: dict
    statistics: dict


def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else CODE_ROOT / p


def load_p1b_config(path=None) -> P1bConfig:
    path = Path(path) if path else Path(__file__).with_name("p1b_config.toml")
    raw = tomllib.load(path.open("rb"))
    return P1bConfig(
        base_config=_resolve(raw["paths"]["base_config"]),
        target_dir=_resolve(raw["paths"]["target_dir"]),
        output=_resolve(raw["paths"]["output"]),
        seeds_calibration=tuple(int(s) for s in raw["seeds"]["calibration"]),
        seeds_frequency=tuple(int(s) for s in raw["seeds"]["frequency"]),
        seeds_restoration=tuple(int(s) for s in raw["seeds"]["restoration"]),
        calibration=dict(raw["calibration"]), gate=dict(raw["gate"]), decision=dict(raw["decision"]),
        restoration=dict(raw["restoration"]), statistics=dict(raw["statistics"]))


def run_kwargs(model: str, g: float, param: float) -> dict:
    """Keyword arguments of run_segment_p1b for one event-scaled model.
    full: param is g itself.  Route variants and static: g fixed, param = k or p_s."""
    cond, pname = MODELS[model]
    kw = dict(convention="event", condition=cond, nmda_scale=float(g))
    if pname == "event_gain":
        kw["event_gain"] = float(param)
    elif pname == "static_gain":
        kw["static_gain"] = float(param)
    return kw


# ------------------------------------------------------------------------------------------- calibration
def next_param(history: list[tuple[float, float]], target: float, max_step: float = 4.0) -> float:
    """Next parameter value from the (parameter, D) history.
    First step multiplicative (p * target / D, the Phase-17 rule); later steps secant in log-log coordinates when
    the local slope lies in [0.1, 3], otherwise multiplicative.  Each step is limited to a factor max_step."""
    p_n, d_n = history[-1]
    if not (np.isfinite(d_n) and d_n > 0 and p_n > 0):
        raise RuntimeError(f"cannot update calibration from p={p_n}, D={d_n}")
    step = np.log(target) - np.log(d_n)
    slope = 1.0
    if len(history) >= 2:
        p_m, d_m = history[-2]
        if p_m > 0 and d_m > 0 and abs(np.log(p_n) - np.log(p_m)) > 1e-9:
            s = (np.log(d_n) - np.log(d_m)) / (np.log(p_n) - np.log(p_m))
            if 0.1 <= s <= 3.0:
                slope = s
    dlog = np.clip(step / slope, -np.log(max_step), np.log(max_step))
    return float(p_n * np.exp(dlog))


def pooled_drive(extras_list) -> float:
    num = sum(float(e["drive_num"]) for e in extras_list)
    den = sum(float(e["drive_steps"]) * float(e["ni"]) for e in extras_list)
    if den <= 0:
        raise RuntimeError("no drive samples in the gain-match window")
    return num / den


def target_from_continuous(results) -> float:
    """C3 drive of existing continuous full runs: pooled gain_match_numerator / (steps x N_I).
    results: iterable of (SimulationResult, cfg)."""
    num = den = 0.0
    for r, cfg in results:
        dt = cfg.simulation.dt_ms
        a = cfg.analysis
        steps = int(round(cfg.simulation.duration_ms / dt))
        t = np.arange(steps) * dt                     # same products as step * dt in the simulator
        n_win = int(np.count_nonzero((a.gain_match_start_ms <= t) & (t <= a.gain_match_end_ms)))
        num += float(r.gain_match_numerator)
        den += float(n_win) * float(cfg.network.ni)
    return num / den


# ------------------------------------------------------------------------------------------- per-trial metrics
def trial_metrics(cfg, result, smoke: bool = False) -> dict:
    """Frequency, oscillation strength, localization and stationarity with the paper's functions."""
    from analysis.flow import summarize_flow, unwrap_axial_trace
    from analysis.pav import summarize_attractor
    from analysis.theta import analyze_theta
    out = dict(frequency_hz=np.nan, theta=np.nan, normalized_peak_power=np.nan, localization_ratio=np.nan,
               localization_failure=np.nan, pav_length_median=np.nan, shift_1_8_deg=np.nan,
               n_e_spikes=int(np.size(result.spike_times_e_ms)), n_i_spikes=int(np.size(result.spike_times_i_ms)))
    try:
        t, _ = analyze_theta(result, cfg.simulation, cfg.analysis)
        out.update(frequency_hz=float(t.dominant_frequency_hz), theta=float(t.dominant_in_theta_band),
                   normalized_peak_power=float(t.normalized_peak_power))
    except Exception:
        if not smoke:
            raise
    try:
        a, trace = summarize_attractor(result, cfg.simulation, cfg.analysis)
        out.update(localization_ratio=float(a.localization_ratio), localization_failure=float(a.localization_failure))
        m = (trace.times_ms >= cfg.analysis.theta_start_ms) & (trace.times_ms <= cfg.analysis.theta_end_ms)
        out["pav_length_median"] = float(np.median(trace.length[m])) if np.any(m) else np.nan
        f = summarize_flow(unwrap_axial_trace(trace, target_axis_deg=cfg.analysis.target_axis_deg,
                                              start_ms=cfg.analysis.theta_start_ms))
        out["shift_1_8_deg"] = float(f.shift_1_8_deg)
    except Exception:
        if not smoke:
            raise
    return out


# ------------------------------------------------------------------------------------------- gate (C or not C)
def evaluate_gate(rows: list[dict], ref_pav_length: float, gate: dict) -> dict:
    """G1-G4 on the drive-matched event-scaled full model (one row per frequency seed)."""
    loc = np.array([r["localization_failure"] for r in rows], float)
    shift = np.array([r["shift_1_8_deg"] for r in rows], float)
    pav = np.array([r["pav_length_median"] for r in rows], float)
    qpk = np.array([r["normalized_peak_power"] for r in rows], float)
    loc_fail = float(np.mean(np.where(np.isfinite(loc), loc, 1.0)))            # missing counts as failure
    stationary = float(np.mean(np.isfinite(shift) & (np.abs(shift) < gate["stationary_max_shift_deg"])))
    pav_ratio = float(np.nanmedian(pav) / ref_pav_length) if (np.isfinite(ref_pav_length) and ref_pav_length > 0
                                                               and np.isfinite(pav).any()) else float("nan")
    osc = float(np.mean(np.isfinite(qpk) & (qpk >= gate["min_normalized_peak_power"])))
    checks = {
        "G1_localization_failure_fraction": loc_fail,
        "G1_pass": loc_fail <= gate["max_localization_failure_fraction"],
        "G2_stationary_fraction": stationary,
        "G2_pass": stationary >= gate["min_stationary_fraction"],
        "G3_pav_length_ratio": pav_ratio,
        "G3_pass": bool(np.isfinite(pav_ratio) and pav_ratio >= gate["min_pav_length_ratio"]),
        "G4_oscillatory_fraction": osc,
        "G4_pass": osc >= gate["min_oscillatory_fraction"],
        "reference_pav_length_median": ref_pav_length,
        "n_trials": len(rows),
    }
    checks["pass"] = bool(checks["G1_pass"] and checks["G2_pass"] and checks["G3_pass"] and checks["G4_pass"])
    return checks


# ------------------------------------------------------------------------------------------- frequency verdict
def frequency_verdict(freq: dict, theta_full, valid: dict, gate_result: dict, decision: dict) -> dict:
    """freq: model -> array of dominant frequencies over the paired frequency seeds (same order).
    valid: model -> calibration validity.  Returns the pre-registered verdict A / A0 / B / C / NOT_EVALUABLE."""
    out = {"verdict": None, "reasons": []}
    if not valid.get("full", False):
        out.update(verdict="NOT_EVALUABLE", reasons=["full_calibration_invalid"])
        return out
    if not gate_result.get("pass", False):
        failed = [k[:2] for k in ("G1_pass", "G2_pass", "G3_pass", "G4_pass") if not gate_result.get(k, False)]
        out.update(verdict="C", reasons=[f"gate_failed:{','.join(failed)}"])
        return out
    bad = [m for m in ROUTE_MODELS if not valid.get(m, False)]
    if bad:
        out.update(verdict="NOT_EVALUABLE", reasons=[f"calibration_invalid:{m}" for m in bad])
        return out
    mf = {m: float(np.nanmean(np.asarray(v, float))) for m, v in freq.items()}
    e_rel = abs(mf["release_only"] - mf["full"])
    e_out = abs(mf["output_only"] - mf["full"])
    d_st = abs(mf["static"] - mf["full"])
    th = float(np.nanmean(np.asarray(theta_full, float)))
    c_rel = e_rel <= decision["release_max_abs_error_hz"]
    c_out = e_out >= decision["output_min_abs_error_hz"]
    c_st = d_st >= decision["static_min_abs_difference_hz"]
    mapping = bool(c_rel and c_out and c_st)
    out.update(mean_frequency_hz=mf, release_abs_error_hz=e_rel, output_abs_error_hz=e_out,
               static_abs_difference_hz=d_st, full_theta_fraction=th,
               criteria={"release_tracks_full": c_rel, "output_departs": c_out, "static_departs": c_st,
                         "route_mapping": mapping,
                         "full_theta": th >= decision["min_theta_fraction"]})
    if mapping and th >= decision["min_theta_fraction"]:
        out["verdict"] = "A"
    elif mapping:
        out["verdict"] = "A0"
        out["reasons"].append("route_mapping_kept_but_full_not_theta")
    else:
        out["verdict"] = "B"
        if not c_st:
            out["reasons"].append("B_flat: full not >= 2 Hz from static (STP no longer changes frequency)")
        if not c_out:
            out["reasons"].append("B_output: output-only within 2 Hz of full")
        if not c_rel:
            out["reasons"].append("B_release: release-only more than 1 Hz from full")
    return out


# ------------------------------------------------------------------------------------------- statistics
def bootstrap_mean_ci(x, n: int = 10000, seed: int = 20261012) -> tuple[float, float, float]:
    """Mean and percentile 95 % CI over trials (paired differences are passed in as x)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, x.size, size=(n, x.size))].mean(axis=1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def restoration_verdict(F_full_act, F_full_actu, F_out, F_rel, min_full: float, n: int, seed: int) -> dict:
    """Arrays are per-trial F_rest(2 s) at 11.25 deg (directions already averaged), paired by trial."""
    m_full = bootstrap_mean_ci(F_full_act, n, seed)
    c_u = bootstrap_mean_ci(np.asarray(F_full_act, float) - np.asarray(F_full_actu, float), n, seed + 1)
    c_r = bootstrap_mean_ci(np.asarray(F_out, float) - np.asarray(F_rel, float), n, seed + 2)
    crit = {"full_restores": bool(m_full[0] >= min_full and m_full[1] > 0),
            "requires_retained_u": bool(c_u[1] > 0),
            "output_route": bool(c_r[1] > 0)}
    return {"full_activity_only": m_full, "activity_only_minus_activity_u": c_u,
            "output_only_minus_release_only": c_r, "criteria": crit,
            "verdict": "OUTPUT_ROUTE_RESTORATION_PRESERVED" if all(crit.values()) else "NOT_PRESERVED"}


def _sim_task(task: dict) -> tuple[str, float, int]:
    """One 8-s trial via run_segment_p1b; resumes if the file already exists with matching metadata."""
    from config import load_config
    from revision_r1.common.simulator_event import load_p1b_extras, run_segment_p1b, save_p1b
    path = Path(task["path"])
    if path.exists():
        ex = load_p1b_extras(path)
        return str(path), float(ex["drive_num"]), int(ex["drive_steps"])
    cfg = load_config(task["base_config"])
    if task.get("duration_ms") is not None:
        from dataclasses import replace
        cfg = replace(cfg, simulation=replace(cfg.simulation, duration_ms=float(task["duration_ms"])),
                      analysis=replace(cfg.analysis, gain_match_start_ms=float(task["gm_start_ms"]),
                                       gain_match_end_ms=float(task["gm_end_ms"])))
    res, _st, ex = run_segment_p1b(cfg, seed=int(task["seed"]), noise_pa=0.0, **task["kwargs"])
    save_p1b(res, ex, path)
    return str(path), float(ex["drive_num"]), int(ex["drive_steps"])


def _pool(tasks: list[dict], workers: int) -> list[Path]:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    if not tasks:
        return []
    workers = max(1, min(int(workers), len(tasks)))
    if workers == 1:
        for t in tasks:
            _sim_task(t)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_sim_task, t) for t in tasks]
            for f in as_completed(futs):
                f.result()
    return [Path(t["path"]) for t in tasks]


def _replay_stats(path, condition, cfg) -> tuple[dict, dict]:
    """R- / spikes-per-cycle diagnostics by exact replay of u, R from the recorded spikes."""
    from revision_r1.t2_efficacy_stats.event_stats import event_distribution_metrics
    from revision_r1.t2_efficacy_stats.reconstruct import TEMPORAL, load_trial, replay_events
    raw = load_trial(path)
    ne = cfg.network.ne
    t0, t1 = cfg.analysis.theta_window_start_ms, cfg.analysis.theta_window_end_ms
    ev = (replay_events(raw, U=cfg.stf.u_baseline, tau_f_ms=cfg.stf.tau_facilitation_ms,
                        tau_r_ms=cfg.stf.tau_recovery_ms, mode=condition, ne=ne)
          if condition in TEMPORAL else None)
    return event_distribution_metrics(ev, raw, mode=condition, U=cfg.stf.u_baseline, t0=t0, t1=t1, ne=ne)
