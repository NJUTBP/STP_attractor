"""P4-light: release-fraction reduction of the oscillation frequency (Revision Plan v4, section 9).

Analysis of existing tables only; no simulation.  Everything is specific to the continuous-gain STP->NMDA
convention of the paper (p = uR multiplies the open NMDA gating).

  python revision_r1/p4_light_reduction/analyze_p4.py

Inputs (returned small files):
  results/revision_r1/t4_release_fraction/cells.csv                 fixed-rho networks (mean_u_release mode)
  results/revision_r1/t4_release_fraction/fixed_point_predictions.csv
  model_sweep_tables/phase10-analysis/combo_summary.csv              full-model frequencies and p_match (42 settings)
  results/revision_r1/t2_efficacy/per_trial_metrics.csv              drive statistics of all variants
  results/revision_r1/p3_u_sensitivity/analysis/p3_per_u.csv         U sweep (out-of-sample test)
  results/revision_r1/p1b/precheck/precheck_summary.json             spikes per cycle and R- of the reference runs
Outputs (results/revision_r1/p4_light/):
  fixed_point_21.csv, p3_out_of_sample.csv, kernel_table.csv, mediator_vs_drive.csv, p4_summary.json, p4_reduction.png

Steps
  A  vectorised re-implementation of the T4 F(rho, p; tau_r) interpolation and fixed-point solver; must reproduce
     the returned fixed_point_predictions.csv (checked, |diff| < 1e-6 Hz)
  B  out-of-sample: the same F (built from U = 0.4 calibration targets) and u+(f; U, tau_f) predict the P3 full-model
     frequency at U = 0.2-0.6 with no fitted parameter (p_match of each U is an input, as in A)
  C  single-spike drive kernel s(t) u [1 - rho exp(-t/tau_r)] of the continuous-gain convention versus the
     event-scaled kernel; delay and early-drive deficit as functions of rho and tau_r
  D  across 42 settings: output-only frequency error versus (i) loss of within-cycle drive modulation and (ii) loss of
     total E->I drive under C1 matching (Spearman and partial Spearman)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

R = CODE_ROOT / "results" / "revision_r1"
OUT = R / "p4_light"
PHASE10 = CODE_ROOT / "model_sweep_tables" / "phase10-analysis" / "combo_summary.csv"
TR_MAIN = (5.0, 20.0, 80.0)
U_REF, TAU_F_REF = 0.4, 1000.0
NMDA = dict(tau_x=2.0, tau_nmda=100.0, alpha=0.5)          # Table S1


# ============================================================================== A. fixed point (vectorised)
def u_plus(f_hz, U, tau_f_ms, spikes_per_cycle=1.0):
    """Periodic fixed point of u+ for evenly spaced spikes (identical to t2 metrics.u_plus_periodic)."""
    f = np.asarray(f_hz, float)
    T = 1000.0 / (f * spikes_per_cycle)
    E = np.exp(-T / tau_f_ms)
    return U * (1.0 + (1.0 - U) * (1.0 - E)) / (1.0 - (1.0 - U) * E)


class FTable:
    """F(rho, p) of one tau_r row: same rules as t4 fixed_point.make_F, evaluated on arrays."""

    def __init__(self, table: pd.DataFrame, p_main: float):
        t = table.copy()
        self.p_main = float(p_main)
        main = t[np.isclose(t.p_target, p_main)].sort_values("rho")
        self.rho_m, self.f_m = main.rho.to_numpy(float), main.freq_hz.to_numpy(float)
        sens = t[~np.isclose(t.p_target, p_main)]
        self.levels = []
        for pl in sorted(set(np.round(sens.p_target, 6))):
            s = sens[np.isclose(sens.p_target, pl)].sort_values("rho")
            d = s.freq_hz.to_numpy(float) - self.main(s.rho.to_numpy(float))
            self.levels.append((float(pl), s.rho.to_numpy(float), d))

    def main(self, rho):
        return np.interp(np.asarray(rho, float), self.rho_m, self.f_m, left=np.nan, right=np.nan)

    def __call__(self, rho, p=None):
        rho = np.asarray(rho, float)
        base = self.main(rho)
        if p is None or not self.levels or np.isclose(p, self.p_main):
            return base
        pts = sorted([(self.p_main, np.zeros_like(rho))] +
                     [(pl, np.interp(rho, r, d)) for pl, r, d in self.levels], key=lambda x: x[0])
        ps = np.array([q for q, _ in pts])
        if p < ps[0] - 1e-9 or p > ps[-1] + 1e-9:
            return np.full_like(rho, np.nan)
        ds = np.vstack([d for _, d in pts])                      # (levels, rho)
        j = int(np.clip(np.searchsorted(ps, p) - 1, 0, len(ps) - 2))
        w = (p - ps[j]) / (ps[j + 1] - ps[j])
        return base + (1 - w) * ds[j] + w * ds[j + 1]


def solve_fixed_point(F: FTable, U, tau_f, p, spikes_per_cycle=1.0, f_lo=1.0, f_hi=40.0, step=0.005):
    """First root of g(f) = F(u+(f), p) - f on the T4 grid (identical rule to t4 fixed_point.solve_fixed_point)."""
    fs = np.arange(f_lo, f_hi + step, step)
    g = F(u_plus(fs, U, tau_f, spikes_per_cycle), p) - fs
    ok = np.isfinite(g)
    idx = np.flatnonzero(ok[:-1] & ok[1:] & (np.sign(g[:-1]) != np.sign(g[1:])))
    if idx.size == 0:
        return float("nan"), float("nan")
    i = idx[0]
    f_star = float(fs[i] - g[i] * (fs[i + 1] - fs[i]) / (g[i + 1] - g[i]))
    return f_star, float(u_plus(f_star, U, tau_f, spikes_per_cycle))


def build_rows():
    cells = pd.read_csv(R / "t4_release_fraction" / "cells.csv")
    valid = cells[cells.calibration_valid & np.isfinite(cells.freq_hz)]
    p10 = pd.read_csv(PHASE10)
    rows = {}
    for tr in TR_MAIN:
        r = p10[np.isclose(p10.tau_recovery_ms, tr)]
        p_main = float(r[np.isclose(r.tau_facilitation_ms, 1000.0)].full_p_match_theta.iloc[0])
        rows[tr] = FTable(valid[np.isclose(valid.tau_r, tr)], p_main)
    return cells, valid, p10, rows


def step_A(p10, rows):
    out = []
    for tr in TR_MAIN:
        F = rows[tr]
        for _, s in p10[np.isclose(p10.tau_recovery_ms, tr)].iterrows():
            f_pred, u_pred = solve_fixed_point(F, U_REF, s.tau_facilitation_ms, s.full_p_match_theta)
            out.append(dict(combo_id=s.combo_id, tau_f_ms=s.tau_facilitation_ms, tau_r_ms=tr,
                            p_match_full=s.full_p_match_theta, f_full_hz=s.full_frequency_hz, f_pred_hz=f_pred,
                            abs_err_hz=abs(f_pred - s.full_frequency_hz), u_plus_pred=u_pred,
                            u_rel_full=s.full_mean_spike_release_u,
                            F_at_U_hz=float(F(np.array([U_REF]), s.full_p_match_theta)[0]),
                            f_output_only_hz=s.u_output_only_frequency_hz,
                            f_release_only_hz=s.u_release_only_frequency_hz))
    df = pd.DataFrame(out)
    old = pd.read_csv(R / "t4_release_fraction" / "fixed_point_predictions.csv")
    m = df.merge(old[["combo_id", "f_pred_hz"]], on="combo_id", suffixes=("", "_t4"))
    diff = np.abs(m.f_pred_hz - m.f_pred_hz_t4)
    both_nan = m.f_pred_hz.isna() & m.f_pred_hz_t4.isna()
    reproduced = bool(np.all(both_nan | (diff < 1e-6)))
    err = df.abs_err_hz.to_numpy(float)
    summ = dict(n_settings=int(len(df)), n_predicted=int(np.isfinite(err).sum()),
                median_abs_err_hz=float(np.nanmedian(err)), n_within_1hz=int(np.sum(err <= 1.0)),
                max_abs_err_hz=float(np.nanmax(err)), reproduces_returned_t4_table=reproduced,
                max_diff_vs_returned_hz=float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0,
                not_predictable=df[df.f_pred_hz.isna()].combo_id.tolist(),
                beyond_1hz=df[df.abs_err_hz > 1.0][["combo_id", "f_full_hz", "f_pred_hz"]].to_dict("records"))
    # u+ check: periodic formula at the measured frequency versus the measured spike-weighted u+
    up = u_plus(df.f_full_hz.to_numpy(float), U_REF, df.tau_f_ms.to_numpy(float))
    df["u_plus_formula_at_f_full"] = up
    summ["u_plus_formula_minus_measured"] = dict(
        median=float(np.median(up - df.u_rel_full)), min=float(np.min(up - df.u_rel_full)),
        max=float(np.max(up - df.u_rel_full)))
    return df, summ


# ============================================================================== B. out-of-sample U sweep
def step_B(rows):
    p3 = pd.read_csv(R / "p3_u_sensitivity" / "analysis" / "p3_per_u.csv")
    F = rows[20.0]
    p_all = [F.p_main] + [lv[0] for lv in F.levels]
    p_lo, p_hi = min(p_all), max(p_all)
    out = []
    for _, s in p3.iterrows():
        f_pred, u_pred = solve_fixed_point(F, float(s.u), TAU_F_REF, float(s.p_match))
        f_given_rho = float(F(np.array([float(s.mean_release_u)]), float(s.p_match))[0])
        out.append(dict(U=float(s.u), p_match=float(s.p_match), f_full_hz=float(s.full_frequency_hz),
                        f_pred_fixed_point_hz=f_pred, abs_err_fixed_point_hz=abs(f_pred - s.full_frequency_hz),
                        u_plus_pred=u_pred, u_rel_measured=float(s.mean_release_u),
                        f_pred_given_measured_rho_hz=f_given_rho,
                        f_mean_release_measured_hz=float(s.mean_u_release_frequency_hz),
                        f_output_only_hz=float(s.u_output_only_frequency_hz),
                        F_at_rho_eq_U_hz=float(F(np.array([float(s.u)]), float(s.p_match))[0]),
                        p_in_table_range=bool(p_lo - 1e-9 <= float(s.p_match) <= p_hi + 1e-9),
                        rho_range_of_table=f"{F.rho_m.min():g}-{F.rho_m.max():g}"))
    return pd.DataFrame(out)


# ============================================================================== C. drive kernel
def single_spike_s(T_ms: float, dt: float = 0.05, n_cycles: int = 25):
    """NMDA gating of one neuron firing once every T ms (Table S1 kinetics), last cycle after convergence.
    Returns time since the spike (ms) and s(t)."""
    tx, tn, a = NMDA["tau_x"], NMDA["tau_nmda"], NMDA["alpha"]
    n = int(round(T_ms / dt))
    x = s = 0.0
    for _ in range(n_cycles):
        x += 1.0
        ss = np.empty(n)
        for k in range(n):                          # midpoint, as in the simulator (x decays exactly)
            k1 = -s / tn + a * x * (1 - s)
            sh = s + 0.5 * dt * k1
            xh = x * np.exp(-0.5 * dt / tx)
            k2 = -sh / tn + a * xh * (1 - sh)
            x *= np.exp(-dt / tx)
            s = s + dt * k2
            ss[k] = s
    return np.arange(1, n + 1) * dt, ss


def kernel_metrics(t, s, rho, tau_r, T_ms):
    """Continuous-gain drive kernel s(t)[1 - rho exp(-t/tau_r)] over one period (u is ~constant in a cycle)."""
    k_cont = s * (1.0 - rho * np.exp(-t / tau_r))
    k_event = s
    out = {}
    for name, k in (("cont", k_cont), ("event", k_event)):
        out[f"centroid_ms_{name}"] = float(np.sum(t * k) / np.sum(k))
        out[f"peak_ms_{name}"] = float(t[np.argmax(k)])
        out[f"early_fraction_{name}"] = float(np.sum(k[t <= 0.25 * T_ms]) / np.sum(k))
    out["mean_ratio_cont_over_event"] = float(np.sum(k_cont) / np.sum(k_event))
    out["early_deficit"] = 1.0 - out["early_fraction_cont"] / out["early_fraction_event"]
    out["delay_ms"] = out["centroid_ms_cont"] - out["centroid_ms_event"]
    return out


def step_C(cells):
    main = cells[(cells.family == "main") & cells.calibration_valid & np.isfinite(cells.freq_hz)]
    cache, rows = {}, []
    for _, c in main.iterrows():
        T = 1000.0 / float(c.freq_hz)
        key = round(T, 1)
        if key not in cache:
            cache[key] = single_spike_s(T)
        t, s = cache[key]
        rows.append(dict(rho=float(c.rho), tau_r=float(c.tau_r), p_target=float(c.p_target),
                         freq_hz=float(c.freq_hz), **kernel_metrics(t, s, float(c.rho), float(c.tau_r), T)))
    df = pd.DataFrame(rows)
    rs = {}
    for tr, g in df.groupby("tau_r"):
        rs[str(tr)] = dict(spearman_freq_vs_delay=float(stats.spearmanr(g.freq_hz, g.delay_ms)[0]),
                           n=int(len(g)))
    rs["pooled_spearman_freq_vs_delay"] = float(stats.spearmanr(df.freq_hz, df.delay_ms)[0])
    return df, rs


# ============================================================================== D. mediator versus drive level
def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    res = lambda a, b: a - np.polyval(np.polyfit(b, a, 1), b)  # noqa: E731
    return float(stats.pearsonr(res(rx, rz), res(ry, rz))[0])


def step_D():
    t = pd.read_csv(R / "t2_efficacy" / "per_trial_metrics.csv", low_memory=False)
    t = t[t.combo_id.notna()]
    g = t.groupby(["combo_id", "label"])[["f_proxy_hz", "M4_modulation_depth", "M4_drive_mean"]].mean()
    g = g.unstack("label")
    rows = []
    for cid in g.index:
        full = g.loc[cid]
        for lab, name in (("u_output_only", "output_only"), ("gain_rescued_depletion", "neither"),
                          ("dynamic_static", "static"), ("u_release_only", "release_only"),
                          ("mean_u_release", "mean_release")):
            rows.append(dict(combo_id=cid, model=name,
                             abs_freq_err_hz=abs(full[("f_proxy_hz", lab)] - full[("f_proxy_hz", "dynamic")]),
                             mod_full=full[("M4_modulation_depth", "dynamic")],
                             mod_model=full[("M4_modulation_depth", lab)],
                             drive_ratio=full[("M4_drive_mean", lab)] / full[("M4_drive_mean", "dynamic")]))
    df = pd.DataFrame(rows)
    df["abs_mod_diff"] = (df.mod_model - df.mod_full).abs()
    df["drive_deficit"] = 1.0 - df.drive_ratio
    o = df[df.model == "output_only"]
    res = dict(
        n_settings=int(len(o)),
        spearman_err_vs_mod_diff=float(stats.spearmanr(o.abs_freq_err_hz, o.abs_mod_diff)[0]),
        spearman_err_vs_drive_deficit=float(stats.spearmanr(o.abs_freq_err_hz, o.drive_deficit)[0]),
        spearman_mod_diff_vs_drive_deficit=float(stats.spearmanr(o.abs_mod_diff, o.drive_deficit)[0]),
        partial_err_vs_mod_given_drive=partial_spearman(o.abs_freq_err_hz, o.abs_mod_diff, o.drive_deficit),
        partial_err_vs_drive_given_mod=partial_spearman(o.abs_freq_err_hz, o.drive_deficit, o.abs_mod_diff))
    ratios = {m: dict(median=float(df[df.model == m].drive_ratio.median()),
                      min=float(df[df.model == m].drive_ratio.min()), max=float(df[df.model == m].drive_ratio.max()))
              for m in df.model.unique()}
    res["drive_ratio_under_C1"] = ratios
    return df, res


# ============================================================================== figure
def figure(cells, p10, A, B, C, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(13.0, 3.2))
    # (a) F(rho) curves of the fixed-release networks, main p of each tau_r row
    for tr, col in zip(TR_MAIN, ("C0", "C1", "C2")):
        c = cells[(cells.family == "main") & np.isclose(cells.tau_r, tr)].sort_values("rho")
        okc = c.calibration_valid
        ax[0].plot(c.rho[okc], c.freq_hz[okc], "o-", color=col, ms=3.5, label=f"tau_r = {tr:g} ms")
        ax[0].plot(c.rho[~okc], c.freq_hz[~okc], "x", color=col)
    ax[0].axhspan(4, 12, color="0.92", zorder=0)
    ax[0].set_xlabel("fixed release fraction rho")
    ax[0].set_ylabel("dominant frequency (Hz)")
    ax[0].legend(fontsize=7)
    ax[0].set_title("fixed-release networks (mean efficacy matched)", fontsize=8)
    # (b) predicted versus measured (21 settings) + P3 out of sample
    ok = np.isfinite(A.f_pred_hz)
    ax[1].plot(A.f_full_hz[ok], A.f_pred_hz[ok], "o", ms=4, color="k", label="42-point sweep (tau_r 5-80 ms)")
    okb = np.isfinite(B.f_pred_fixed_point_hz)
    ax[1].plot(B.f_full_hz[okb], B.f_pred_fixed_point_hz[okb], "s", ms=5, mfc="none", color="C3",
               label="U sweep (out of sample)")
    lim = [3.5, 12]
    ax[1].plot(lim, lim, ":", color="0.5")
    ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel("full model (Hz)")
    ax[1].set_ylabel("fixed point f = F(u+(f)) (Hz)")
    ax[1].legend(fontsize=7)
    # (c) kernel illustration at the reference point
    T = 1000.0 / 6.4453125
    t, s = single_spike_s(T)
    for rho, ls in ((0.4, "--"), (0.889, "-")):
        ax[2].plot(t, s * (1 - rho * np.exp(-t / 20.0)) / s.max(), ls, color="C3",
                   label=f"continuous gain, rho = {rho:g}")
    ax[2].plot(t, s / s.max(), color="0.4", label="event-scaled (R- = 1)")
    ax[2].set_xlim(0, T)
    ax[2].set_xlabel("time after spike (ms)")
    ax[2].set_ylabel("E->I drive of one neuron (norm.)")
    ax[2].legend(fontsize=6.5)
    # (d) frequency versus kernel delay
    for tr, col in zip(TR_MAIN, ("C0", "C1", "C2")):
        g = C[np.isclose(C.tau_r, tr)]
        ax[3].plot(g.delay_ms, g.freq_hz, "o", color=col, ms=3.5, label=f"tau_r = {tr:g} ms")
    ax[3].set_xlabel("drive-centroid delay from depletion (ms)")
    ax[3].set_ylabel("dominant frequency (Hz)")
    ax[3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cells, valid, p10, rows = build_rows()
    A, sA = step_A(p10, rows)
    A.to_csv(OUT / "fixed_point_21.csv", index=False)
    B = step_B(rows)
    B.to_csv(OUT / "p3_out_of_sample.csv", index=False)
    C, sC = step_C(cells)
    C.to_csv(OUT / "kernel_table.csv", index=False)
    D, sD = step_D()
    D.to_csv(OUT / "mediator_vs_drive.csv", index=False)
    pre = json.loads((R / "p1b" / "precheck" / "precheck_summary.json").read_text(encoding="utf-8-sig"))
    mo = pre["mean_over_trials"]
    okB = np.isfinite(B.f_pred_fixed_point_hz)
    summary = dict(
        A_fixed_point=sA,
        B_out_of_sample=dict(n_u=int(len(B)), n_predicted=int(okB.sum()),
                             abs_err_hz=B.abs_err_fixed_point_hz.round(4).tolist(),
                             median_abs_err_hz=float(np.nanmedian(B.abs_err_fixed_point_hz)),
                             max_abs_err_hz=float(np.nanmax(B.abs_err_fixed_point_hz))),
        C_kernel=sC, D_mediator_vs_drive=sD,
        E_reference_spiking=dict(frac_active_neuron_cycles_with_one_spike=mo["frac_cycle_with_1_spike"],
                                 Rminus_mean=mo["Rminus_mean"], Rminus_q01=mo["Rminus_q01"],
                                 event_amplitude_mean=mo["event_amp_mean"], event_amplitude_cv=mo["event_amp_cv"],
                                 active_fraction_per_cycle=mo["active_fraction_per_cycle_mean"]),
        note="continuous-gain convention only; exploratory reduction on existing data (T4 was run under its v1 "
             "preregistration before v3 archived it; its rules were not changed)")
    (OUT / "p4_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    figure(cells, p10, A, B, C, OUT / "p4_reduction.png")
    print(json.dumps({"A": {k: sA[k] for k in ("n_predicted", "median_abs_err_hz", "n_within_1hz",
                                                 "reproduces_returned_t4_table")},
                      "B": summary["B_out_of_sample"], "C": sC,
                      "D": {k: v for k, v in sD.items() if k != "drive_ratio_under_C1"}}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
