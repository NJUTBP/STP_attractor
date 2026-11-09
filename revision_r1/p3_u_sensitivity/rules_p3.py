"""P3 pre-registered definitions (PREREGISTRATION.md sections 4-6), shared by run_p3.py and analyze_p3.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # Python <= 3.10
    import tomli as tomllib

CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("p3_config.toml")
MODES = ("u_output_only", "u_release_only", "mean_u_release")


def load_p3_section(path=None) -> dict:
    raw = tomllib.load(Path(path or DEFAULT_CONFIG).open("rb"))
    p3 = dict(raw["p3"])
    out = Path(p3["output"])
    p3["output"] = out if out.is_absolute() else CODE_ROOT / out
    p3["smoke"] = bool(p3.get("smoke", False))
    return p3


def interpretable(rows: list[dict], p3: dict) -> dict:
    """Localized, stationary, oscillatory regime of the full model at one U (theta not required)."""
    loc = np.array([r["localization_failure"] for r in rows], float)
    sh = np.array([r["shift_1_8_deg"] for r in rows], float)
    q = np.array([r["normalized_peak_power"] for r in rows], float)
    th = np.array([r["theta"] for r in rows], float)
    res = dict(
        localization_failure_fraction=float(np.mean(np.where(np.isfinite(loc), loc, 1.0))),
        stationary_fraction=float(np.mean(np.isfinite(sh) & (np.abs(sh) < p3["stationary_max_shift_deg"]))),
        oscillatory_fraction=float(np.mean(np.isfinite(q) & (q >= p3["min_normalized_peak_power"]))),
        theta_fraction=float(np.nanmean(th)) if np.isfinite(th).any() else float("nan"),
        frequency_hz_mean=float(np.nanmean([r["frequency_hz"] for r in rows])),
    )
    res["interpretable"] = bool(res["localization_failure_fraction"] <= p3["max_localization_failure_fraction"]
                                and res["stationary_fraction"] >= p3["min_stationary_fraction"]
                                and res["oscillatory_fraction"] >= p3["min_oscillatory_fraction"])
    return res


def anchor_check(per_seed_freq: dict, per_seed_shift: dict, p_match: float, release_u: float, ref: dict) -> dict:
    """U = 0.4 raw full must reproduce the frozen reference (Phase 8/10 values in config/anchor_reference.json)."""
    tol = ref["tolerances"]
    rows = []

    def add(name, ours, theirs, strict, warn):
        d = abs(float(ours) - float(theirs))
        rows.append(dict(check=name, ours=float(ours), reference=float(theirs), diff=d,
                         level="PASS" if d <= strict else ("WARN" if d <= warn else "FAIL")))

    for s, f in per_seed_freq.items():
        r = ref["per_seed_theta"].get(str(s))
        if r is None:
            continue
        add(f"seed{s}_frequency_hz", f, r["frequency_hz"], tol["frequency_hz_abs"], tol["warn_frequency_hz_abs"])
        add(f"seed{s}_zero_noise_abs_shift_deg", abs(per_seed_shift[s]), r["zero_noise_abs_shift_1_8_deg"],
            tol["angle_deg_abs"], tol["warn_angle_deg_abs"])
    add("pooled_p_match", p_match, ref["full_p_match_theta"], tol["p_match_abs"],
        tol["warn_p_match_rel"] * ref["full_p_match_theta"])
    add("mean_spike_release_u", release_u, ref["full_mean_spike_release_u"], tol["release_u_abs"], 1e-3)
    lv = [r["level"] for r in rows]
    status = "FAIL" if (not rows or "FAIL" in lv) else ("WARN" if "WARN" in lv else "PASS")
    return dict(status=status, checks=rows)


def route_mapping(f_full: float, f_rel: float, f_out: float, decision: dict) -> dict:
    e_rel, e_out = abs(f_rel - f_full), abs(f_out - f_full)
    return dict(release_abs_error_hz=e_rel, output_abs_error_hz=e_out,
                holds=bool(e_rel <= decision["release_route_max_abs_error_hz"]
                           and e_out >= decision["output_route_min_abs_error_hz"]))


def overall_verdict(per_u: list[dict], p3: dict) -> dict:
    """per_u rows: u, interpretable, calibration_valid (all route modes), mapping_holds (or None)."""
    ref_u = float(p3["reference_u"])
    non_ref = [r for r in per_u if abs(r["u"] - ref_u) > 1e-12]
    n_interp_nonref = sum(1 for r in non_ref if r["interpretable"])
    evaluable = [r for r in per_u if r["interpretable"] and r.get("calibration_valid") and r.get("mapping_holds") is not None]
    holds = [r["u"] for r in evaluable if r["mapping_holds"]]
    fails = [r["u"] for r in evaluable if not r["mapping_holds"]]
    not_eval = [r["u"] for r in per_u if r["interpretable"] and not (r.get("calibration_valid") and r.get("mapping_holds") is not None)]
    if not evaluable:
        v = "NOT_EVALUABLE"
    elif not fails and n_interp_nonref >= int(p3["min_interpretable_nonreference"]) and not not_eval:
        v = "U_ROBUST"
    elif not fails:
        v = "HOLDS_WHERE_EVALUABLE"          # mapping held everywhere it could be tested, but < 3 non-reference U
    elif holds:
        v = "PARTIAL"
    else:
        v = "FAILS"
    return dict(verdict=v, mapping_holds_at_u=holds, mapping_fails_at_u=fails, not_evaluable_u=not_eval,
                interpretable_u=[r["u"] for r in per_u if r["interpretable"]],
                n_interpretable_nonreference=n_interp_nonref)
