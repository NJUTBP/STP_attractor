#!/usr/bin/env python3
"""Statistical recovery/audit for Phase 16.

This script does not modify simulation outputs. It adds the seed-level/cluster-level
inference that is required by PHASE16_PLAN_zh.md but is not fully emitted by the
original analyze_phase16.py.

Examples
--------
python phase16/recovery_analysis.py --module A --output results/phase16
python phase16/recovery_analysis.py --module B --output results/phase16
python phase16/recovery_analysis.py --module C --output results/phase16 --bootstrap-repeats 10000
python phase16/recovery_analysis.py --module D --output results/phase16 --bootstrap-repeats 10000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402
from phase16.analyze_phase16 import load, profiles, unwrap  # noqa: E402
from phase16.reduced_model import bootstrap_mean_ci, dax, fit_msd  # noqa: E402


def _boot_mean(x, n=10000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan, 0
    m, lo, hi = bootstrap_mean_ci(x, n=n, seed=seed)
    return m, lo, hi, int(x.size)


def _fit_gain(g: pd.DataFrame) -> tuple[float, float]:
    x = g["minus_o"].to_numpy(float)
    y = g["v"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    den = float(np.dot(x, x))
    if len(x) < 2 or den <= 0:
        return np.nan, np.nan
    beta = float(np.dot(x, y) / den)
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = np.nan if sst <= 0 else float(1 - np.sum((y - beta * x) ** 2) / sst)
    return beta, r2


def _cluster_sample(df: pd.DataFrame, sampled_seeds: np.ndarray) -> pd.DataFrame:
    # Repeating a seed repeats its whole within-trial block, preserving repeated time points/directions.
    parts = [df[df.seed == int(s)] for s in sampled_seeds]
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()


def audit_A(root: Path, out: Path, nboot: int):
    adir = root / "module_A" / "analysis"
    rest = pd.read_csv(adir / "A_restoration_by_trial.csv")
    vel = pd.read_csv(adir / "A_velocity_by_trial.csv")

    # P1: paired clamp-minus-normal contrasts at the preregistered 11.25 deg, 600 ms endpoint.
    s = rest[np.isclose(rest.abs_delta, 11.25) & np.isclose(rest.t_ms, 600.0)]
    p = s.pivot(index="seed", columns=["condition", "mode"], values="F")
    p1 = []
    for cond in ("dynamic", "u_output_only", "u_release_only"):
        d = (p[(cond, "clamp")] - p[(cond, "normal")]).dropna().to_numpy(float)
        m, lo, hi, nn = _boot_mean(d, nboot)
        p1.append(dict(condition=cond, endpoint="F_clamp(600ms)-F_normal(600ms)", abs_delta=11.25,
                       mean=m, ci_low=lo, ci_high=hi, n_pairs=nn))
    pd.DataFrame(p1).to_csv(out / "A_P1_paired_contrasts.csv", index=False)

    # P2a: small-offset linear force coefficient, fitted within each seed through the origin.
    small = vel[(vel["mode"] == "clamp") & (vel.abs_delta <= 16.875)].copy()
    slopes = []
    for (cond, seed), g in small.groupby(["condition", "seed"]):
        x = g.abs_delta.to_numpy(float); y = g.v_toward_old_deg_s.to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        den = float(np.dot(x[ok], x[ok]))
        beta = np.nan if ok.sum() < 3 or den <= 0 else float(np.dot(x[ok], y[ok]) / den)
        slopes.append(dict(condition=cond, seed=int(seed), small_delta_slope_per_s=beta))
    ds = pd.DataFrame(slopes)
    ds.to_csv(out / "A_P2_small_delta_slopes_by_trial.csv", index=False)
    sm = []
    for cond, g in ds.groupby("condition"):
        m, lo, hi, nn = _boot_mean(g.small_delta_slope_per_s, nboot)
        sm.append(dict(condition=cond, mean_slope_per_s=m, ci_low=lo, ci_high=hi, n=nn))
    pd.DataFrame(sm).to_csv(out / "A_P2_small_delta_slopes_summary.csv", index=False)

    # P2b: clamp force curve and paired full-minus-output contrasts at each offset.
    fc = []
    clamp = vel[vel["mode"] == "clamp"]
    pv = clamp.pivot(index="seed", columns=["condition", "abs_delta"], values="v_toward_old_deg_s")
    deltas = sorted(clamp.abs_delta.unique())
    for d in deltas:
        x = (pv[("dynamic", d)] - pv[("u_output_only", d)]).dropna().to_numpy(float)
        m, lo, hi, nn = _boot_mean(x, nboot)
        fc.append(dict(abs_delta=d, contrast="dynamic_minus_u_output_only", mean=m, ci_low=lo, ci_high=hi, n_pairs=nn))
    pd.DataFrame(fc).to_csv(out / "A_P2_route_contrasts.csv", index=False)

    peak = []
    for cond, g in clamp.groupby("condition"):
        means = g.groupby("abs_delta", as_index=False).v_toward_old_deg_s.mean()
        row = means.loc[means.v_toward_old_deg_s.idxmax()]
        peak.append(dict(condition=cond, peak_delta_deg=float(row.abs_delta), peak_velocity_deg_s=float(row.v_toward_old_deg_s)))
    pd.DataFrame(peak).to_csv(out / "A_P2_peak_force.csv", index=False)


def audit_B(root: Path, out: Path, nboot: int):
    bdir = root / "module_B" / "analysis"
    kin = pd.read_csv(bdir / "B_kinematics.csv")
    fits = pd.read_csv(bdir / "B_two_variable_fits.csv")
    reg = kin[(kin.t_ms >= 20) & (kin.t_ms <= 500)].copy()
    reg["minus_o"] = -reg.drive_offset
    reg["v"] = reg.v_toward_old
    reg = reg[np.isfinite(reg.minus_o) & np.isfinite(reg.v)]
    reg["group"] = reg.condition.astype(str) + "|" + reg.kind.astype(str)

    groups = sorted(reg.group.unique())
    point = []
    b, r2 = _fit_gain(reg)
    point.append(dict(scope="pooled", gain_per_s=b, r2=r2, n_rows=len(reg), n_seeds=reg.seed.nunique()))
    for name, g in reg.groupby("group"):
        b, r2 = _fit_gain(g)
        point.append(dict(scope=name, gain_per_s=b, r2=r2, n_rows=len(g), n_seeds=g.seed.nunique()))
    pd.DataFrame(point).to_csv(out / "B_P3_gain_point_estimates.csv", index=False)

    seeds = np.array(sorted(reg.seed.unique()), int)
    rng = np.random.default_rng(0)
    scopes = ["pooled"] + groups
    boots = {s: [] for s in scopes}
    diffs = {s: [] for s in groups}
    for _ in range(nboot):
        sample = rng.choice(seeds, size=len(seeds), replace=True)
        rb = _cluster_sample(reg, sample)
        gp, _ = _fit_gain(rb)
        boots["pooled"].append(gp)
        for name in groups:
            gg = rb[rb.group == name]
            gi, _ = _fit_gain(gg)
            boots[name].append(gi)
            diffs[name].append(gi - gp)
    rows = []
    for scope in scopes:
        arr = np.asarray(boots[scope], float); arr = arr[np.isfinite(arr)]
        pe = next(x["gain_per_s"] for x in point if x["scope"] == scope)
        rows.append(dict(scope=scope, gain_per_s=pe,
                         ci_low=np.percentile(arr, 2.5) if len(arr) else np.nan,
                         ci_high=np.percentile(arr, 97.5) if len(arr) else np.nan,
                         n_seed_clusters=len(seeds)))
    pd.DataFrame(rows).to_csv(out / "B_P3_gain_cluster_bootstrap.csv", index=False)
    rows = []
    for name in groups:
        arr = np.asarray(diffs[name], float); arr = arr[np.isfinite(arr)]
        g0 = next(x["gain_per_s"] for x in point if x["scope"] == name)
        p0 = next(x["gain_per_s"] for x in point if x["scope"] == "pooled")
        rows.append(dict(scope=name, difference_from_pooled=g0-p0,
                         ci_low=np.percentile(arr, 2.5) if len(arr) else np.nan,
                         ci_high=np.percentile(arr, 97.5) if len(arr) else np.nan,
                         n_seed_clusters=len(seeds)))
    pd.DataFrame(rows).to_csv(out / "B_P3_group_minus_pooled_cluster_bootstrap.csv", index=False)

    # P4: seed-level parameters and comparison to the Module-A clamp estimate of a.
    summ = []
    for (cond, d), g in fits.groupby(["condition", "abs_delta"]):
        for col in ("a", "b", "F_inf", "rms"):
            m, lo, hi, nn = _boot_mean(g[col].to_numpy(float), nboot)
            summ.append(dict(condition=cond, abs_delta=d, parameter=col, mean=m, ci_low=lo, ci_high=hi, n=nn))
        bminus = g.b.to_numpy(float) - 1.0  # 1/tau_f at the 1000-ms anchor, in s^-1
        m, lo, hi, nn = _boot_mean(bminus, nboot)
        summ.append(dict(condition=cond, abs_delta=d, parameter="b_minus_1_over_tau_f", mean=m, ci_low=lo, ci_high=hi, n=nn))
    pd.DataFrame(summ).to_csv(out / "B_P4_parameter_summary.csv", index=False)

    af = root / "module_A" / "analysis" / "A_rate_fits_by_trial.csv"
    if af.exists():
        A = pd.read_csv(af)
        A = A[A["mode"] == "clamp"][["condition", "abs_delta", "seed", "a"]].rename(columns={"a": "a_A_clamp"})
        B = fits[["condition", "abs_delta", "seed", "a"]].rename(columns={"a": "a_B"})
        M = B.merge(A, on=["condition", "abs_delta", "seed"], how="inner")
        M["difference_B_minus_A"] = M.a_B - M.a_A_clamp
        M.to_csv(out / "B_P4_a_comparison_by_trial.csv", index=False)
        rows = []
        for (cond, d), g in M.groupby(["condition", "abs_delta"]):
            m, lo, hi, nn = _boot_mean(g.difference_B_minus_A.to_numpy(float), nboot)
            rows.append(dict(condition=cond, abs_delta=d, mean_difference=m, ci_low=lo, ci_high=hi, n_pairs=nn))
        pd.DataFrame(rows).to_csv(out / "B_P4_a_comparison_summary.csv", index=False)


def _curve_from_runs(runs, sampled_seeds=None):
    if sampled_seeds is None:
        chosen = runs
    else:
        chosen = []
        for s in sampled_seeds:
            chosen.extend([r for r in runs if r["seed"] == int(s)])
    n = min(len(r["t"]) for r in chosen)
    t = chosen[0]["t"][:n]
    P = np.stack([r["phi"][:n] for r in chosen])
    out = {"t": t, "msd": np.mean(P**2, axis=0)}
    if all("delta_eff" in r for r in chosen):
        D = np.stack([r["delta_eff"][:n] for r in chosen])
        out["var_delta_eff"] = np.var(D, axis=0)
    if all("delta_u" in r for r in chosen):
        Du = np.stack([r["delta_u"][:n] for r in chosen])
        out["var_delta_u"] = np.var(Du, axis=0)
    return out


def _slope(t, y, lo, hi):
    m = (t >= lo) & (t <= hi) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(t[m], y[m], 1)[0])


def audit_C(root: Path, out: Path, cfg, nboot: int):
    data = load(root, "C")
    U = cfg.stf.u_baseline
    traj = []
    for _key, (meta, d) in data.items():
        p = profiles(meta, d, U)
        rec = dict(condition=meta["condition"], noise=float(meta["noise"]), seed=int(meta["seed"]), draw=int(meta["draw"]),
                   t=p["t"].astype(float)/1000.0, phi=unwrap(p["phi"]))
        if "psi_u" in p:
            # Descriptive only: the plan already established that the global u center is not
            # the reduced model's force coordinate.
            rec["delta_u"] = dax(p["phi"], p["psi_u"])
        if "drive_offset" in p:
            # Primary relative coordinate for P5: delta_eff = phi - psi_drive = -o.
            rec["delta_eff"] = -np.asarray(p["drive_offset"], float)
        traj.append(rec)

    # Freeze the quantitative P5 target *before looking at Module-C results*: use the
    # deterministic full-model 11.25-deg F_inf estimated in Module B.  For each bootstrap
    # resample, use the same sampled seed IDs to propagate B uncertainty into the predicted
    # long/short MSD-slope ratio (1-F_inf)^2.
    bfit_path = root / "module_B" / "analysis" / "B_two_variable_fits.csv"
    bfit = None
    if bfit_path.exists():
        tmp = pd.read_csv(bfit_path)
        bfit = tmp[(tmp.condition == "dynamic") & np.isclose(tmp.abs_delta, 11.25)][["seed", "F_inf"]].dropna()
        if bfit.empty:
            bfit = None

    summary, boot_summary, direct_tests = [], [], []
    rng = np.random.default_rng(0)
    for (cond, noise) in sorted({(r["condition"], r["noise"]) for r in traj}):
        runs = [r for r in traj if r["condition"] == cond and r["noise"] == noise]
        seeds = np.array(sorted({r["seed"] for r in runs}), int)
        base = _curve_from_runs(runs)
        t, msd = base["t"], base["msd"]
        stride = max(1, len(t)//120)
        fit = fit_msd(t[::stride], msd[::stride], joint=(cond != "static_gain"))
        early = _slope(t, msd, 0.1, 0.5)
        late = _slope(t, msd, 4.0, min(8.0, float(t[-1])))
        ratio = late/early if np.isfinite(early) and early != 0 else np.nan
        row = dict(condition=cond, noise=noise, n_seeds=len(seeds), n_runs=len(runs),
                   early_msd_slope=early, late_msd_slope=late, empirical_late_early_ratio=ratio, **fit)
        if "var_delta_eff" in base:
            row["late_var_delta_eff_slope"] = _slope(t, base["var_delta_eff"], 2.0, min(8.0, float(t[-1])))
        if "var_delta_u" in base:
            row["late_var_delta_u_slope"] = _slope(t, base["var_delta_u"], 2.0, min(8.0, float(t[-1])))

        predicted = np.nan
        if cond == "dynamic" and bfit is not None:
            fb = bfit[bfit.seed.isin(seeds)].F_inf.to_numpy(float)
            if len(fb):
                fbar = float(np.mean(fb))
                predicted = (1.0 - fbar) ** 2
                row["B_F_inf_11p25_mean"] = fbar
                row["predicted_ratio_from_B"] = predicted
                row["empirical_minus_predicted_ratio"] = ratio - predicted if np.isfinite(ratio) else np.nan
        summary.append(row)

        vals = []
        test_vals = []
        for _ in range(nboot):
            sample = rng.choice(seeds, size=len(seeds), replace=True)
            c = _curve_from_runs(runs, sample)
            tt, yy = c["t"], c["msd"]
            ff = fit_msd(tt[::stride], yy[::stride], joint=(cond != "static_gain"))
            ee = _slope(tt, yy, 0.1, 0.5)
            ll = _slope(tt, yy, 4.0, min(8.0, float(tt[-1])))
            rr = ll/ee if np.isfinite(ee) and ee != 0 else np.nan
            rec = dict(early_msd_slope=ee, late_msd_slope=ll, empirical_late_early_ratio=rr, **ff)
            if "var_delta_eff" in c:
                rec["late_var_delta_eff_slope"] = _slope(tt, c["var_delta_eff"], 2.0, min(8.0, float(tt[-1])))
            if "var_delta_u" in c:
                rec["late_var_delta_u_slope"] = _slope(tt, c["var_delta_u"], 2.0, min(8.0, float(tt[-1])))

            if cond == "dynamic" and bfit is not None:
                # Repeated sampled seed IDs deliberately repeat the corresponding B fit too.
                fmap = dict(zip(bfit.seed.astype(int), bfit.F_inf.astype(float)))
                fb = np.array([fmap[int(s)] for s in sample if int(s) in fmap], float)
                if len(fb):
                    fbar = float(np.mean(fb))
                    pr = (1.0 - fbar) ** 2
                    rec["B_F_inf_11p25_mean"] = fbar
                    rec["predicted_ratio_from_B"] = pr
                    rec["empirical_minus_predicted_ratio"] = rr - pr if np.isfinite(rr) else np.nan
                    test_vals.append(rec["empirical_minus_predicted_ratio"])
            vals.append(rec)

        bv = pd.DataFrame(vals)
        for col in bv.columns:
            x = bv[col].to_numpy(float); x=x[np.isfinite(x)]
            point = row.get(col, np.nan)
            boot_summary.append(dict(condition=cond, noise=noise, parameter=col, point_estimate=point,
                                     ci_low=np.percentile(x,2.5) if len(x) else np.nan,
                                     ci_high=np.percentile(x,97.5) if len(x) else np.nan,
                                     n_seed_clusters=len(seeds)))
        if cond == "dynamic" and np.isfinite(predicted):
            x = np.asarray(test_vals, float); x=x[np.isfinite(x)]
            direct_tests.append(dict(condition=cond, noise=noise,
                                     B_F_inf_11p25_mean=row.get("B_F_inf_11p25_mean", np.nan),
                                     predicted_ratio_from_B=predicted,
                                     empirical_ratio=ratio,
                                     empirical_minus_predicted=ratio-predicted if np.isfinite(ratio) else np.nan,
                                     ci_low=np.percentile(x,2.5) if len(x) else np.nan,
                                     ci_high=np.percentile(x,97.5) if len(x) else np.nan,
                                     n_seed_clusters=len(seeds)))

    pd.DataFrame(summary).to_csv(out / "C_P5_point_estimates.csv", index=False)
    pd.DataFrame(boot_summary).to_csv(out / "C_P5_cluster_bootstrap.csv", index=False)
    pd.DataFrame(direct_tests).to_csv(out / "C_P5_direct_B_prediction_test.csv", index=False)



def audit_D(root: Path, out: Path, nboot: int):
    """P6 audit using seed as the independent unit.

    D is exploratory by design.  We therefore report directional paired contrasts and
    bootstrap CIs rather than binary equivalence claims.  The preregistered directions are:
      * at tf=1000 ms, restoration should decrease as tr increases;
      * tf=250 ms should reduce restoration relative to the 1000-ms anchor;
      * tf=2500 ms is expected to be similar to the anchor, but no equivalence margin was
        preregistered, so this is reported descriptively;
      * the activity-minus-activity+R contrast should become more negative as tr increases.
    """
    ddir = root / "module_D" / "analysis"
    path = ddir / "D_restoration_2s_by_trial.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run analyze_phase16.py --module D first; missing {path}")
    df = pd.read_csv(path)
    req = {"tf", "tr", "condition", "kind", "seed", "abs_delta", "F_2s"}
    miss = req.difference(df.columns)
    if miss:
        raise ValueError(f"D_restoration_2s_by_trial.csv missing columns: {sorted(miss)}")

    # Re-emit grouped means/CIs with an explicit seed-level audit label.
    grouped = []
    for keys, g in df.groupby(["tf", "tr", "condition", "kind", "abs_delta"]):
        m, lo, hi, nn = _boot_mean(g.F_2s.to_numpy(float), nboot)
        tf, tr, cond, kind, delta = keys
        grouped.append(dict(tf=tf, tr=tr, condition=cond, kind=kind, abs_delta=delta,
                            mean_F_2s=m, ci_low=lo, ci_high=hi, n_seeds=nn))
    pd.DataFrame(grouped).to_csv(out / "D_P6_grouped_restoration_seed_bootstrap.csv", index=False)

    def get_series(tf, tr, cond, kind, delta):
        g = df[(np.isclose(df.tf, tf)) & (np.isclose(df.tr, tr)) &
               (df.condition == cond) & (df.kind == kind) & np.isclose(df.abs_delta, delta)]
        return g.set_index("seed").F_2s.sort_index()

    def paired_contrast(label, a, b, expected_direction, extra=None):
        x = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        d = (x.a - x.b).to_numpy(float)
        m, lo, hi, nn = _boot_mean(d, nboot)
        row = dict(contrast=label, mean=m, ci_low=lo, ci_high=hi, n_pairs=nn,
                   expected_direction=expected_direction)
        if extra:
            row.update(extra)
        return row

    deltas = sorted(float(x) for x in df.abs_delta.dropna().unique())
    rows = []
    for delta in deltas:
        # tau_r scaling at tf=1000 ms for both full and output-only routes.
        for cond in ("dynamic", "u_output_only"):
            f5 = get_series(1000.0, 5.0, cond, "activity_shift", delta)
            f20 = get_series(1000.0, 20.0, cond, "activity_shift", delta)
            f80 = get_series(1000.0, 80.0, cond, "activity_shift", delta)
            meta = dict(abs_delta=delta, condition=cond, family="tau_r")
            rows += [
                paired_contrast("F_tr5_minus_tr20", f5, f20, ">0", meta),
                paired_contrast("F_tr20_minus_tr80", f20, f80, ">0", meta),
                paired_contrast("F_tr5_minus_tr80", f5, f80, ">0", meta),
            ]

            # tau_f scaling at tr=20 ms.
            f250 = get_series(250.0, 20.0, cond, "activity_shift", delta)
            f1000 = f20
            f2500 = get_series(2500.0, 20.0, cond, "activity_shift", delta)
            meta2 = dict(abs_delta=delta, condition=cond, family="tau_f")
            rows += [
                paired_contrast("F_tf250_minus_tf1000", f250, f1000, "<0", meta2),
                paired_contrast("F_tf2500_minus_tf1000", f2500, f1000, "~0 descriptive; no equivalence test", meta2),
            ]

        # Does the full model exceed output-only more strongly away from the anchor?
        for tf, tr in ((1000.0, 5.0), (1000.0, 20.0), (1000.0, 80.0),
                       (250.0, 20.0), (2500.0, 20.0)):
            full = get_series(tf, tr, "dynamic", "activity_shift", delta)
            outp = get_series(tf, tr, "u_output_only", "activity_shift", delta)
            rows.append(paired_contrast("F_full_minus_output", full, outp, "descriptive",
                                        dict(abs_delta=delta, condition="dynamic-vs-output",
                                             family="route", tf=tf, tr=tr)))

        # Resource-shift contrast within the full model: activity - activity+R.
        c_by_tr = {}
        for tr in (5.0, 20.0, 80.0):
            act = get_series(1000.0, tr, "dynamic", "activity_shift", delta)
            ar = get_series(1000.0, tr, "dynamic", "activity_r_shift", delta)
            x = pd.concat([act.rename("act"), ar.rename("ar")], axis=1).dropna()
            c = x.act - x.ar
            c_by_tr[tr] = c
            m, lo, hi, nn = _boot_mean(c.to_numpy(float), nboot)
            rows.append(dict(contrast="F_activity_minus_activityR", mean=m, ci_low=lo, ci_high=hi,
                             n_pairs=nn, expected_direction="becomes more negative as tr increases",
                             abs_delta=delta, condition="dynamic", family="resource_shift", tf=1000.0, tr=tr))
        rows.append(paired_contrast("resource_contrast_tr80_minus_tr5", c_by_tr[80.0], c_by_tr[5.0], "<0",
                                    dict(abs_delta=delta, condition="dynamic", family="resource_shift", tf=1000.0)))

    pd.DataFrame(rows).to_csv(out / "D_P6_key_paired_contrasts.csv", index=False)

    # Trial-level trend slopes against log2 time constant provide one compact directional readout.
    trend_rows = []
    for delta in deltas:
        for cond in ("dynamic", "u_output_only"):
            sub = df[(df.condition == cond) & (df.kind == "activity_shift") &
                     np.isclose(df.tf, 1000.0) & df.tr.isin([5.0, 20.0, 80.0]) &
                     np.isclose(df.abs_delta, delta)]
            for seed, g in sub.groupby("seed"):
                if g.tr.nunique() == 3:
                    slope = float(np.polyfit(np.log2(g.tr.to_numpy(float)), g.F_2s.to_numpy(float), 1)[0])
                    trend_rows.append(dict(family="tau_r", condition=cond, abs_delta=delta, seed=int(seed),
                                           slope_per_log2_ms=slope))
        # Same trend for activity - activity+R at tf=1000.
        for seed in sorted(df.seed.unique()):
            vals=[]
            for tr in (5.0,20.0,80.0):
                act=get_series(1000.0,tr,"dynamic","activity_shift",delta)
                ar=get_series(1000.0,tr,"dynamic","activity_r_shift",delta)
                if seed in act.index and seed in ar.index:
                    vals.append((tr,float(act.loc[seed]-ar.loc[seed])))
            if len(vals)==3:
                slope=float(np.polyfit(np.log2([v[0] for v in vals]), [v[1] for v in vals], 1)[0])
                trend_rows.append(dict(family="resource_contrast_vs_tau_r", condition="dynamic", abs_delta=delta,
                                       seed=int(seed), slope_per_log2_ms=slope))
    trdf = pd.DataFrame(trend_rows)
    trdf.to_csv(out / "D_P6_trend_slopes_by_trial.csv", index=False)
    summ=[]
    for (fam,cond,delta),g in trdf.groupby(["family","condition","abs_delta"]):
        m,lo,hi,nn=_boot_mean(g.slope_per_log2_ms.to_numpy(float), nboot)
        exp = "<0" if fam in {"tau_r","resource_contrast_vs_tau_r"} else "descriptive"
        summ.append(dict(family=fam, condition=cond, abs_delta=delta, mean_slope=m,
                         ci_low=lo, ci_high=hi, n=nn, expected_direction=exp))
    pd.DataFrame(summ).to_csv(out / "D_P6_trend_slopes_summary.csv", index=False)

    # Exploratory compensation decomposition at tf=1000 ms.  This is deliberately
    # separate from the preregistered P6 outputs because the compensation pattern
    # was identified after inspecting Module D.  It quantifies the algebraic split
    #     Delta full = Delta output-only + Delta(full - output-only)
    # between tr=5 and 80 ms, using paired seeds and bootstrap CIs.
    comp_rows = []
    for delta in deltas:
        full5 = get_series(1000.0, 5.0, "dynamic", "activity_shift", delta)
        full80 = get_series(1000.0, 80.0, "dynamic", "activity_shift", delta)
        out5 = get_series(1000.0, 5.0, "u_output_only", "activity_shift", delta)
        out80 = get_series(1000.0, 80.0, "u_output_only", "activity_shift", delta)
        x = pd.concat([full5.rename("full5"), full80.rename("full80"),
                       out5.rename("out5"), out80.rename("out80")], axis=1).dropna()
        if len(x) == 0:
            continue
        pieces = {
            "full_tr80_minus_tr5": x.full80 - x.full5,
            "output_tr80_minus_tr5": x.out80 - x.out5,
            "extra_full_minus_output_tr80_minus_tr5":
                (x.full80 - x.out80) - (x.full5 - x.out5),
        }
        for label, vals in pieces.items():
            m, lo, hi, nn = _boot_mean(vals.to_numpy(float), nboot)
            comp_rows.append(dict(abs_delta=delta, component=label, mean=m,
                                  ci_low=lo, ci_high=hi, n_pairs=nn))
        # Numerical identity check, emitted for reproducibility rather than inference.
        residual = (x.full80 - x.full5) - ((x.out80 - x.out5) +
                    ((x.full80 - x.out80) - (x.full5 - x.out5)))
        comp_rows.append(dict(abs_delta=delta, component="identity_residual",
                              mean=float(residual.mean()), ci_low=float(residual.min()),
                              ci_high=float(residual.max()), n_pairs=int(len(residual))))
    pd.DataFrame(comp_rows).to_csv(out / "D_P6_compensation_decomposition.csv", index=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--module", choices=list("ABCD"), required=True)
    p.add_argument("--output", default=str(ROOT / "results" / "phase16"))
    p.add_argument("--config", default=str(ROOT / "config" / "phase4_causal.toml"))
    p.add_argument("--bootstrap-repeats", type=int, default=10000)
    a = p.parse_args()
    root = Path(a.output)
    out = root / f"module_{a.module}" / "analysis" / "recovery"
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(a.config)
    {"A": lambda: audit_A(root, out, a.bootstrap_repeats),
     "B": lambda: audit_B(root, out, a.bootstrap_repeats),
     "C": lambda: audit_C(root, out, cfg, a.bootstrap_repeats),
     "D": lambda: audit_D(root, out, a.bootstrap_repeats)}[a.module]()
    print("wrote", sorted(x.name for x in out.iterdir()))


if __name__ == "__main__":
    main()
