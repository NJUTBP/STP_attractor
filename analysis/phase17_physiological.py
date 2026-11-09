"""Phase-17 analysis — implements phase17/PREREGISTRATION.md sections 4-5 exactly.

Outputs (in <output>/analysis/):
  gate_report.json, full_summary.csv, route_summary.csv, restoration_by_trial.csv,
  restoration_mean_ci.csv, decisions.json, phase17_overview.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config
from experiment.phase17_physiological import (Phase17Config, _seed_paths, config_for, is_functional,
                                              pooled_gain, pooled_release_u, read_manifest, select_combos)
from io_utils import load_result, read_json, write_json


# ----------------------------------------------------------------------------- per-seed metrics (Phase-8 definitions)
def per_seed_metrics(cfg, r0, r60=None, smoke=False) -> dict:
    from analysis.flow import summarize_flow, unwrap_axial_trace
    from analysis.pav import axial_displacement_deg, summarize_attractor
    from analysis.theta import analyze_theta
    out = dict(frequency_hz=np.nan, theta=np.nan, zero_noise_abs_shift_1_8_deg=np.nan,
               noise_endpoint_delta_60pa_deg=np.nan, loc_fail_0pa=np.nan, loc_fail_60pa=np.nan)
    def guard(fn):
        try:
            fn()
        except Exception:
            if not smoke:
                raise
    def th():
        t, _ = analyze_theta(r0, cfg.simulation, cfg.analysis)
        out.update(frequency_hz=float(t.dominant_frequency_hz), theta=float(t.dominant_in_theta_band))
    guard(th)
    a0 = p0 = None
    def att0():
        nonlocal a0, p0
        a0, p0 = summarize_attractor(r0, cfg.simulation, cfg.analysis)
        out["loc_fail_0pa"] = float(a0.localization_failure)
    guard(att0)
    def flow():
        f = summarize_flow(unwrap_axial_trace(p0, target_axis_deg=cfg.analysis.target_axis_deg, start_ms=1000.0))
        out["zero_noise_abs_shift_1_8_deg"] = float(abs(f.shift_1_8_deg))
    guard(flow)
    if r60 is not None:
        def att60():
            a60, _ = summarize_attractor(r60, cfg.simulation, cfg.analysis)
            out["loc_fail_60pa"] = float(a60.localization_failure)
            out["noise_endpoint_delta_60pa_deg"] = float(axial_displacement_deg(a60.pav_deg_at_target, a0.pav_deg_at_target))
        guard(att60)
    return out


def summarize_runs(cfg, theta_dir, att_dir, seeds, smoke=False) -> dict:
    rows = []
    for p0, p60 in zip(_seed_paths(Path(theta_dir), seeds),
                       _seed_paths(Path(att_dir), seeds) if att_dir else [None] * len(seeds)):
        r60 = load_result(p60) if (p60 is not None and Path(p60).exists()) else None
        rows.append(per_seed_metrics(cfg, load_result(p0), r60, smoke))
    d = pd.DataFrame(rows)
    return dict(frequency_hz=float(d.frequency_hz.mean()), frequency_sd_hz=float(d.frequency_hz.std(ddof=1)) if len(d) > 1 else 0.0,
                theta_fraction=float(d.theta.mean()), localization_failure_fraction_0pa=float(d.loc_fail_0pa.mean()),
                localization_failure_fraction_60pa=float(d.loc_fail_60pa.mean()),
                noise_mean_abs_delta_deg=float(d.noise_endpoint_delta_60pa_deg.abs().mean()),
                zero_noise_mean_abs_shift_deg=float(d.zero_noise_abs_shift_1_8_deg.mean()),
                p_match=pooled_gain(_seed_paths(Path(theta_dir), seeds)),
                mean_release_u=pooled_release_u(_seed_paths(Path(theta_dir), seeds)))


# ----------------------------------------------------------------------------- restoration (Phase-16 module-D estimator)
def restoration_table(root: Path, readout_ms: float, window_ms: float) -> pd.DataFrame:
    from phase16.reduced_model import dax, pav_short
    data = {}
    for f in sorted(Path(root).rglob("branches/*.npz")):
        z = np.load(f); m = json.loads(str(z["meta"]))
        data[(m["combo_id"], m["variant"], m["condition"], m["seed"], m["kind"], float(m["delta"]))] = z
    grid = np.array([float(readout_ms)])
    rows = []
    for key, z in data.items():
        cid, v, cond, seed, kind, delta = key
        if kind == "no_shift" or delta <= 0:
            continue
        ref = data.get((cid, v, cond, seed, "no_shift", 0.0)); neg = data.get((cid, v, cond, seed, kind, -delta))
        if ref is None or neg is None:
            continue
        pr = pav_short(ref["spike_t"], ref["spike_a"], grid, window_ms)
        q = np.nanmean([s * dax(pav_short(b["spike_t"], b["spike_a"], grid, window_ms), pr) for s, b in ((1, z), (-1, neg))], axis=0)
        rows.append(dict(combo_id=cid, variant=v, condition=cond, kind=kind, seed=seed, abs_delta=delta,
                         F_2s=float(1.0 - q[0] / delta)))
    return pd.DataFrame(rows)


def ci_table(df: pd.DataFrame, keys) -> pd.DataFrame:
    from phase16.reduced_model import bootstrap_mean_ci
    rows = []
    for k, g in df.groupby(keys):
        m, lo, hi = bootstrap_mean_ci(g.F_2s.values)
        rows.append(dict(zip(keys, k), F_2s_mean=m, ci_low=lo, ci_high=hi, n=int(g.F_2s.notna().sum())))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- gate (PREREGISTRATION section 4)
def analyze_gate(cfg17: Phase17Config, out: Path, smoke=False) -> dict:
    out = Path(out); ref = read_json(cfg17.anchor_reference); tol = ref["tolerances"]
    base = load_config(cfg17.base_config)
    g = out / "gate"
    checks = []
    def add(name, ours, theirs, strict, warn, rel=False):
        diff = abs(ours - theirs) / abs(theirs) if rel else abs(ours - theirs)
        lvl = "PASS" if diff <= strict else ("WARN" if diff <= warn else "FAIL")
        checks.append(dict(check=name, ours=ours, reference=theirs, diff=diff, level=lvl))
    for s in cfg17.seeds:
        m = per_seed_metrics(base, load_result(g / "full" / "theta" / f"seed_{s:04d}.npz"),
                             load_result(g / "full" / "attractor" / f"seed_{s:04d}.npz"), smoke)
        r = ref["per_seed_theta"].get(str(s))
        if r is None:
            continue
        add(f"seed{s}_frequency_hz", m["frequency_hz"], r["frequency_hz"], tol["frequency_hz_abs"], tol["warn_frequency_hz_abs"])
        add(f"seed{s}_zero_noise_abs_shift_deg", m["zero_noise_abs_shift_1_8_deg"], r["zero_noise_abs_shift_1_8_deg"],
            tol["angle_deg_abs"], tol["warn_angle_deg_abs"])
        add(f"seed{s}_noise_endpoint_delta_deg", m["noise_endpoint_delta_60pa_deg"], r["noise_endpoint_delta_60pa_deg"],
            tol["angle_deg_abs"], tol["warn_angle_deg_abs"])
    paths = _seed_paths(g / "full" / "theta", cfg17.seeds)
    if tuple(cfg17.seeds) == (0, 1, 2, 3):
        add("pooled_p_match_theta", pooled_gain(paths), ref["full_p_match_theta"], tol["p_match_abs"], tol["warn_p_match_rel"] * ref["full_p_match_theta"])
        add("mean_spike_release_u", pooled_release_u(paths), ref["full_mean_spike_release_u"], tol["release_u_abs"], 1e-3)
    R = cfg17.restoration
    rt = restoration_table(g / "restoration", R["readout_ms"], R["window_ms"])
    for cond, key in (("dynamic", "moduleD_F2s_dynamic"), ("u_output_only", "moduleD_F2s_u_output_only")):
        sub = rt[(rt.condition == cond) & (rt.kind == "activity_shift")] if len(rt) else rt
        for row in sub.itertuples() if len(sub) else []:
            k = f"{int(row.seed)}_{row.abs_delta:g}"
            if k in ref[key]:
                add(f"F2s_{cond}_seed{int(row.seed)}_{row.abs_delta:g}", row.F_2s, ref[key][k], tol["F2s_abs"], tol["warn_F2s_abs"])
    levels = [c["level"] for c in checks]
    status = "FAIL" if (not checks or "FAIL" in levels) else ("WARN" if "WARN" in levels else "PASS")
    if smoke:
        status = "NOT_APPLICABLE (smoke test)"
    rep = dict(status=status, n_checks=len(checks), n_pass=levels.count("PASS"), n_warn=levels.count("WARN"),
               n_fail=levels.count("FAIL"), checks=checks,
               rule="PASS/WARN -> continue (WARN must be reported); FAIL -> stop (PREREGISTRATION section 4)")
    (out / "analysis").mkdir(parents=True, exist_ok=True)
    write_json(rep, out / "analysis" / "gate_report.json")
    return rep


# ----------------------------------------------------------------------------- main analysis (PREREGISTRATION section 5)
def analyze_all(cfg17: Phase17Config, out: Path, smoke=False) -> dict:
    out = Path(out); A = out / "analysis"; A.mkdir(parents=True, exist_ok=True)
    base = load_config(cfg17.base_config)
    D = cfg17.decision
    full_rows, route_rows = [], []
    for cid, u, tf, tr in select_combos(cfg17, None):
        man = read_manifest(out, cid)
        if not man:
            continue
        cfg = config_for(base, u, tf, tr)
        for v in cfg17.variants:
            if v not in man or "full" not in man[v]:
                continue
            f = man[v]["full"]
            att = f.get("attractor_dir")
            summ = summarize_runs(cfg, f["theta_dir"], att, cfg17.seeds, smoke)
            valid = True if v == "raw" else bool(f.get("valid", False))
            functional = is_functional(summ, cfg17)
            full_rows.append(dict(combo_id=cid, variant=v, u_baseline=u, tau_facilitation_ms=tf, tau_recovery_ms=tr,
                                  final_scale=f["final_scale"], calibration_converged=f.get("converged", True),
                                  calibration_relative_error=f.get("final_relative_error", np.nan),
                                  calibration_valid=valid, functional=functional, **summ))
            if "routes" in man[v]:
                row = dict(combo_id=cid, variant=v, u_baseline=u, tau_facilitation_ms=tf, tau_recovery_ms=tr,
                           full_frequency_hz=summ["frequency_hz"], full_valid=valid, full_functional=functional)
                for mode, r in man[v]["routes"].items():
                    ms = summarize_runs(cfg, r["theta_dir"], r.get("attractor_dir"), cfg17.seeds, smoke)
                    row[f"{mode}_frequency_hz"] = ms["frequency_hz"]
                    row[f"{mode}_abs_frequency_error_hz"] = abs(ms["frequency_hz"] - summ["frequency_hz"])
                    row[f"{mode}_theta_fraction"] = ms["theta_fraction"]
                    row[f"{mode}_final_scale"] = r["final_scale"]
                    row[f"{mode}_calibration_valid"] = bool(r.get("valid", False))
                    row[f"{mode}_noise_mean_abs_delta_deg"] = ms["noise_mean_abs_delta_deg"]
                if {"u_release_only_abs_frequency_error_hz", "u_output_only_abs_frequency_error_hz"} <= set(row):
                    row["route_dissociation_holds"] = route_holds(row["u_release_only_abs_frequency_error_hz"],
                                                                  row["u_output_only_abs_frequency_error_hz"], D)
                route_rows.append(row)
    fs = pd.DataFrame(full_rows); rs = pd.DataFrame(route_rows)
    fs.to_csv(A / "full_summary.csv", index=False); rs.to_csv(A / "route_summary.csv", index=False)
    R = cfg17.restoration
    rt = restoration_table(out / "restoration", R["readout_ms"], R["window_ms"])
    rt.to_csv(A / "restoration_by_trial.csv", index=False)
    rc = ci_table(rt, ["combo_id", "variant", "condition", "kind", "abs_delta"]) if len(rt) else pd.DataFrame()
    rc.to_csv(A / "restoration_mean_ci.csv", index=False)
    decisions = {}
    d0 = float(min(R["deltas_deg"]))
    for v in cfg17.variants:
        sub = fs[(fs.variant == v)] if len(fs) else fs
        eligible = sub[(sub.calibration_valid) & (sub.functional)] if len(sub) else sub
        dec = dict(n_combos_analysed=int(len(sub)), n_calibration_invalid=int((~sub.calibration_valid).sum()) if len(sub) else 0,
                   n_nonfunctional=int((~sub.functional).sum()) if len(sub) else 0, n_eligible=int(len(eligible)),
                   eligible=sorted(eligible.combo_id.tolist()) if len(eligible) else [])
        if len(rs) and "route_dissociation_holds" in rs:
            r_el = rs[(rs.variant == v) & rs.combo_id.isin(dec["eligible"])]
            frac = float(r_el.route_dissociation_holds.mean()) if len(r_el) else np.nan
            dec["route_dissociation"] = dict(n=int(len(r_el)), fraction_holds=frac, verdict=_verdict(frac, D, len(r_el)),
                                             holds_in=sorted(r_el[r_el.route_dissociation_holds].combo_id.tolist()))
        if len(rc):
            c_el = rc[(rc.variant == v) & (rc.condition == "dynamic") & (rc.kind == "activity_shift")
                      & (rc.abs_delta == d0) & rc.combo_id.isin(dec["eligible"])]
            present = np.array([restoration_present(m, lo, D) for m, lo in zip(c_el.F_2s_mean, c_el.ci_low)], bool)
            frac = float(present.mean()) if len(c_el) else np.nan
            dec["restoration"] = dict(n=int(len(c_el)), fraction_present=frac, verdict=_verdict(frac, D, len(c_el)),
                                      present_in=sorted(c_el.combo_id[present].tolist()),
                                      F2s_range=[float(c_el.F_2s_mean.min()), float(c_el.F_2s_mean.max())] if len(c_el) else None)
        decisions[v] = dec
    decisions["note"] = ("matched is the primary variant (PREREGISTRATION section 2); smoke outputs are not interpretable"
                         if not smoke else "SMOKE TEST — not interpretable")
    write_json(decisions, A / "decisions.json")
    try:
        _figure(fs, rs, rc, cfg17, A / "phase17_overview.png", d0)
    except Exception as e:  # figure is optional
        print("figure skipped:", e)
    return decisions


def route_holds(release_err_hz: float, output_err_hz: float, D: dict) -> bool:
    """PREREGISTRATION 5.3: release route reproduces the full frequency and the output route does not."""
    return bool(release_err_hz <= D["release_route_max_abs_error_hz"] and output_err_hz >= D["output_route_min_abs_error_hz"])


def restoration_present(mean_F2s: float, ci_low: float, D: dict) -> bool:
    """PREREGISTRATION 5.4."""
    return bool(mean_F2s >= D["restoration_min_F2s"] and ci_low > 0)


def _verdict(frac, D, n):
    if n == 0 or not np.isfinite(frac):
        return "no eligible combos"
    if frac >= D["generalizes_fraction"]:
        return "generalizes"
    if frac <= D["fails_fraction"]:
        return "fails"
    return "partial"


def _figure(fs, rs, rc, cfg17, path, d0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    us, tfs, trs = cfg17.u_baseline, cfg17.tau_facilitation_ms, cfg17.tau_recovery_ms
    v = "matched" if "matched" in cfg17.variants else cfg17.variants[0]
    panels = [("full frequency (Hz)", lambda cid: _get(fs, cid, v, "frequency_hz"), "viridis", (0, 15)),
              ("|output-only − full| (Hz)", lambda cid: _get(rs, cid, v, "u_output_only_abs_frequency_error_hz"), "magma_r", (0, 8)),
              ("|release-only − full| (Hz)", lambda cid: _get(rs, cid, v, "u_release_only_abs_frequency_error_hz"), "magma_r", (0, 8)),
              (f"F(2 s), {d0:g}° shift", lambda cid: _getF(rc, cid, v, d0), "RdBu", (-0.4, 0.4))]
    from experiment.phase17_physiological import combo_id
    fig, ax = plt.subplots(len(us), len(panels), figsize=(3.2 * len(panels), 2.8 * len(us)), squeeze=False,
                           constrained_layout=True)
    for i, u in enumerate(us):
        for j, (title, get, cmap, lim) in enumerate(panels):
            M = np.array([[get(combo_id(u, tf, tr)) for tr in trs] for tf in tfs], float)
            im = ax[i, j].imshow(M, origin="lower", cmap=cmap, vmin=lim[0], vmax=lim[1], aspect="auto")
            for (a, b), val in np.ndenumerate(M):
                nf = _get(fs, combo_id(u, tfs[a], trs[b]), v, "functional")
                txt = "NF" if nf is False or nf == 0.0 else (f"{val:.2f}" if np.isfinite(val) else "–")
                ax[i, j].text(b, a, txt, ha="center", va="center", fontsize=7)
            ax[i, j].set_xticks(range(len(trs))); ax[i, j].set_xticklabels([f"{t:g}" for t in trs])
            ax[i, j].set_yticks(range(len(tfs))); ax[i, j].set_yticklabels([f"{t:g}" for t in tfs])
            ax[i, j].set_xlabel("τ_r (ms)"); ax[i, j].set_ylabel("τ_f (ms)")
            ax[i, j].set_title(f"U = {u:g}: {title}", fontsize=8)
            fig.colorbar(im, ax=ax[i, j], shrink=0.7)
    fig.suptitle(f"Phase 17 ({v} variant); NF = non-functional (pre-registered criterion)", fontsize=9)
    fig.savefig(path, dpi=180)


def _get(df, cid, v, col):
    if df is None or not len(df) or col not in df:
        return np.nan
    s = df[(df.combo_id == cid) & (df.variant == v)]
    return s[col].iloc[0] if len(s) else np.nan


def _getF(rc, cid, v, d0):
    if rc is None or not len(rc):
        return np.nan
    s = rc[(rc.combo_id == cid) & (rc.variant == v) & (rc.condition == "dynamic") & (rc.kind == "activity_shift") & (rc.abs_delta == d0)]
    return float(s.F_2s_mean.iloc[0]) if len(s) else np.nan
