#!/usr/bin/env python3
"""Analyse Phase-16 branches and write CSV/JSON summaries into <output>/module_X/analysis/.

  python phase16/analyze_phase16.py --module A --output results/phase16
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
from phase16.reduced_model import (axial_center, bootstrap_mean_ci, dax, drive_offset, efficacy,  # noqa: E402
                                   fit_clamp_rate, fit_msd, fit_restoration, pav_short)


def load(root, module):
    out = {}
    for f in sorted((Path(root) / f"module_{module}" / "branches").glob("*.npz")):
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        key = (meta["tf"], meta["tr"], meta["condition"], meta["seed"], meta["kind"], meta["delta"],
               meta.get("mode", "normal"), meta.get("noise", 0.0), meta.get("draw"))
        out[key] = (meta, d)
    return out


def ci_table(df, keys, value):
    rows = []
    for k, g in df.groupby(keys):
        m, lo, hi = bootstrap_mean_ci(g[value].values)
        rows.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)), mean=m, ci_low=lo, ci_high=hi, n=g[value].notna().sum()))
    return pd.DataFrame(rows)


def paired_residual(data, base, delta, grid, window):
    """Sign-aligned residual displacement relative to the unshifted branch, averaged over directions."""
    ref = data.get(base[:4] + ("no_shift", 0.0) + base[6:])
    pos = data.get(base[:4] + (base[4], delta) + base[6:])
    neg = data.get(base[:4] + (base[4], -delta) + base[6:])
    if ref is None or pos is None or neg is None:
        return None
    pr = pav_short(ref[1]["spike_t"], ref[1]["spike_a"], grid, window)
    qs = [s * dax(pav_short(b[1]["spike_t"], b[1]["spike_a"], grid, window), pr) for s, b in ((1, pos), (-1, neg))]
    return np.nanmean(qs, axis=0)


def analyze_A(root, out):
    data = load(root, "A")
    T = max(m["T"] for m, _ in data.values())
    w = min(150.0, T / 3.0)                      # 150 ms spans one theta cycle in the full network
    grid = np.arange(w, T + 1e-9, max(5.0, w / 6.0))
    rows, vel = [], []
    for key in data:
        tf, tr, cond, seed, kind, delta, mode, *_ = key
        if kind != "activity_shift" or delta <= 0:
            continue
        q = paired_residual(data, key, delta, grid, w)
        if q is None:
            continue
        rows += [dict(condition=cond, mode=mode, seed=seed, abs_delta=delta, t_ms=g, residual_deg=x, F=1 - x / delta)
                 for g, x in zip(grid, q)]
        m = (grid >= w) & (grid <= min(400.0, T)) & np.isfinite(q)
        if m.sum() >= 3:
            vel.append(dict(condition=cond, mode=mode, seed=seed, abs_delta=delta,
                            v_toward_old_deg_s=-np.polyfit(grid[m] / 1000.0, q[m], 1)[0]))
    df, dv = pd.DataFrame(rows), pd.DataFrame(vel)
    df.to_csv(out / "A_restoration_by_trial.csv", index=False)
    ci_table(df, ["condition", "mode", "abs_delta", "t_ms"], "F").to_csv(out / "A_restoration_mean_ci.csv", index=False)
    if len(dv):
        dv.to_csv(out / "A_velocity_by_trial.csv", index=False)
        ci_table(dv, ["condition", "mode", "abs_delta"], "v_toward_old_deg_s").to_csv(out / "A_force_curve_mean_ci.csv", index=False)
    fits = []
    for (cond, mode, d), g in df.groupby(["condition", "mode", "abs_delta"]):
        for seed, gs in g.groupby("seed"):
            t = gs.t_ms.values / 1000.0; F = gs.F.values
            if mode == "clamp":
                a, rms = fit_clamp_rate(t, F); fits.append(dict(condition=cond, mode=mode, abs_delta=d, seed=seed, a=a, rms=rms))
            else:
                a, b, rms = fit_restoration(t, F); fits.append(dict(condition=cond, mode=mode, abs_delta=d, seed=seed, a=a, b=b,
                                                                    F_inf=a / (a + b), rms=rms))
    pd.DataFrame(fits).to_csv(out / "A_rate_fits_by_trial.csv", index=False)


def profiles(meta, d, U):
    theta = d["theta_e"].astype(float)
    s = d["snap_s"].astype(float)
    phi, _ = axial_center(s, theta)
    res = dict(t=d["snap_t"].astype(float), phi=phi)
    if "snap_u" in d.files:
        u, R = d["snap_u"].astype(float), d["snap_R"].astype(float)
        k = float(meta["scale_kwargs"]["temporal_gain_scale"])
        res["psi_u"], _ = axial_center(np.maximum(u - U, 0.0), theta)
        res["psi_D"], _ = axial_center(np.maximum(1.0 - R, 0.0), theta)
        res["drive_offset"], _ = drive_offset(s, efficacy(u, R, meta["condition"], U, k), theta)
    return res


def analyze_B(root, out, cfg):
    data = load(root, "B"); U = cfg.stf.u_baseline
    rows, fits, reg = [], [], []
    for key, (meta, d) in data.items():
        tf, tr, cond, seed, kind, delta, *_ = key
        if kind == "no_shift" or delta <= 0:
            continue
        refkey = (tf, tr, cond, seed, "no_shift", 0.0) + key[6:]
        if refkey not in data:
            continue
        pref = profiles(*data[refkey], U)
        per_dir = []
        for s_ in (1, -1):
            k2 = key[:5] + (s_ * delta,) + key[6:]
            if k2 not in data:
                break
            p = profiles(*data[k2], U)
            al = dict(t=p["t"], x=s_ * dax(p["phi"], pref["phi"]))
            for name in ("psi_u", "psi_D"):
                if name in p:
                    al[name] = s_ * dax(p[name], pref[name])
            if "drive_offset" in p:
                al["o"] = s_ * (p["drive_offset"] - pref["drive_offset"])
            per_dir.append(al)
        if len(per_dir) < 2:
            continue
        avg = {k: np.nanmean([a[k] for a in per_dir], axis=0) for k in per_dir[0]}
        t = avg["t"]; x = avg["x"]
        xs = pd.Series(x).rolling(5, center=True, min_periods=1).mean().values
        v = -np.gradient(xs, t / 1000.0)
        for i in range(0, len(t), 2):
            rows.append(dict(condition=cond, kind=kind, seed=seed, abs_delta=delta, t_ms=t[i], x=x[i],
                             psi_u=avg.get("psi_u", [np.nan] * len(t))[i], psi_D=avg.get("psi_D", [np.nan] * len(t))[i],
                             drive_offset=avg.get("o", [np.nan] * len(t))[i], v_toward_old=v[i]))
        if kind == "activity_shift":
            m = t >= 50
            a, b, rms = fit_restoration(t[m] / 1000.0, 1 - x[m] / delta)
            fits.append(dict(condition=cond, seed=seed, abs_delta=delta, a=a, b=b, F_inf=a / (a + b), rms=rms))
        if "o" in avg:
            m = (t >= 20) & (t <= 500) & np.isfinite(avg["o"])
            reg += [dict(condition=cond, kind=kind, seed=seed, abs_delta=delta, v=vv, minus_o=-oo) for vv, oo in zip(v[m], avg["o"][m])]
    pd.DataFrame(rows).to_csv(out / "B_kinematics.csv", index=False)
    pd.DataFrame(fits).to_csv(out / "B_two_variable_fits.csv", index=False)
    dr = pd.DataFrame(reg)
    if len(dr):
        gain = []
        for scope, g in [("pooled", dr)] + [(c, g) for c, g in dr.groupby("condition")]:
            A = g.minus_o.values; y = g.v.values
            gcoef = float(np.dot(A, y) / np.dot(A, A)) if np.dot(A, A) > 0 else np.nan
            r2 = 1 - np.sum((y - gcoef * A) ** 2) / np.sum((y - y.mean()) ** 2)
            gain.append(dict(scope=scope, gain_per_s=gcoef, r2=r2, n=len(g)))
        pd.DataFrame(gain).to_csv(out / "B_unified_gain.csv", index=False)


def unwrap(x):
    return np.concatenate([[0.0], np.cumsum(dax(x[1:], x[:-1]))])


def analyze_C(root, out, cfg):
    data = load(root, "C"); U = cfg.stf.u_baseline
    traj = []
    for key, (meta, d) in data.items():
        p = profiles(meta, d, U)
        rec = dict(condition=meta["condition"], noise=meta["noise"], seed=meta["seed"], draw=meta["draw"], t=p["t"],
                   phi=unwrap(p["phi"]))
        if "psi_u" in p:
            rec["delta_u"] = dax(p["phi"], p["psi_u"]); rec["o"] = p["drive_offset"]
        traj.append(rec)
    rows, fits = [], []
    for (cond, noise), grp in pd.DataFrame(traj).groupby(["condition", "noise"]):
        n = min(len(r) for r in grp.t)
        t = grp.t.iloc[0][:n] / 1000.0
        P = np.stack([r[:n] for r in grp.phi]); msd = np.mean(P ** 2, axis=0)
        extra = {}
        if "o" in grp and grp.o.notna().all():
            O = np.stack([r[:n] for r in grp.o]); extra["var_drive_offset"] = np.var(O, axis=0)
            Du = np.stack([r[:n] for r in grp.delta_u]); extra["var_phi_minus_psi_u"] = np.var(Du, axis=0)
        for i in range(n):
            rows.append(dict(condition=cond, noise=noise, t_s=t[i], msd_phi=msd[i], **{k: v[i] for k, v in extra.items()}))
        fits.append(dict(condition=cond, noise=noise, n_runs=len(grp),
                         **fit_msd(t, msd, joint=(cond != "static_gain"))))
    pd.DataFrame(rows).to_csv(out / "C_msd_and_relative_variance.csv", index=False)
    pd.DataFrame(fits).to_csv(out / "C_msd_fits.csv", index=False)


def analyze_D(root, out):
    data = load(root, "D")
    rows = []
    grid = np.array([2000.0])
    for key in data:
        tf, tr, cond, seed, kind, delta, *_ = key
        if kind == "no_shift" or delta <= 0:
            continue
        q = paired_residual(data, key, delta, grid, 400.0)
        if q is not None:
            rows.append(dict(tf=tf, tr=tr, condition=cond, kind=kind, seed=seed, abs_delta=delta, F_2s=1 - q[0] / delta))
    df = pd.DataFrame(rows); df.to_csv(out / "D_restoration_2s_by_trial.csv", index=False)
    ci_table(df, ["tf", "tr", "condition", "kind", "abs_delta"], "F_2s").to_csv(out / "D_restoration_2s_mean_ci.csv", index=False)
    con = []
    for (tf, tr, d, seed), g in df.groupby(["tf", "tr", "abs_delta", "seed"]):
        get = lambda c, k: g[(g.condition == c) & (g.kind == k)].F_2s.mean()
        con.append(dict(tf=tf, tr=tr, abs_delta=d, seed=seed,
                        full_minus_output=get("dynamic", "activity_shift") - get("u_output_only", "activity_shift"),
                        activity_minus_activity_u=get("dynamic", "activity_shift") - get("dynamic", "activity_u_shift"),
                        activity_minus_activity_R=get("dynamic", "activity_shift") - get("dynamic", "activity_r_shift")))
    dc = pd.DataFrame(con)
    if len(dc):
        long = dc.melt(id_vars=["tf", "tr", "abs_delta", "seed"], var_name="contrast", value_name="value")
        ci_table(long, ["tf", "tr", "abs_delta", "contrast"], "value").to_csv(out / "D_contrasts_mean_ci.csv", index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--module", choices=list("ABCD"), required=True)
    p.add_argument("--output", default=str(ROOT / "results" / "phase16"))
    p.add_argument("--config", default=str(ROOT / "config" / "phase4_causal.toml"))
    a = p.parse_args()
    out = Path(a.output) / f"module_{a.module}" / "analysis"; out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(a.config)
    {"A": lambda: analyze_A(a.output, out), "B": lambda: analyze_B(a.output, out, cfg),
     "C": lambda: analyze_C(a.output, out, cfg), "D": lambda: analyze_D(a.output, out)}[a.module]()
    print("wrote", sorted(x.name for x in out.iterdir()))


if __name__ == "__main__":
    main()
