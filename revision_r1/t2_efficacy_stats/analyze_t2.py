"""T2 summary: per-setting means, paired contrasts against the full model, and the pre-written verdict.

python revision_r1/t2_efficacy_stats/analyze_t2.py --out results/revision_r1/t2_efficacy
Rules (PREREGISTRATION.md, section 5): at the reference setting tf_1000_tr_20, release-only within 10% of full on
M2_cv_p, M4_modulation_depth and M5_release_mean; output-only more than 25% away on the same three; and across
settings Spearman rho > 0.5 between the output-only |frequency error| and its |M4_modulation_depth| difference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

KEY = ["M2_cv_p", "M4_modulation_depth", "M5_release_mean"]
REPORT = KEY + ["M1_p_drive_weighted", "M2_var_p", "M3_drive_width_deg", "M3_drive_peak", "M3_p_spatial_sd",
                "M4_drive_cv", "M4_acf_at_one_period", "M4_acf_first_zero_ms", "M4_drive_freq_hz", "M5_p_pre_mean",
                "M5_p_pre_sd", "M5_rho_spike_mean", "M6_cov_share", "M7_ps_corr", "f_proxy_hz",
                "spikes_per_active_neuron_per_cycle", "u_plus_spike_mean", "u_plus_fixed_point_pred"]
REPORT += ["V3_Rminus_mean", "V3_Rminus_q05", "V3_Rminus_q50", "V3_Rminus_frac_below_0p9",
           "V3_Rminus_frac_below_0p99", "V3_Rminus_mean_first_in_cycle", "V3_Rminus_mean_later_in_cycle",
           "V3_frac_spikes_later_in_cycle", "V3_event_amp_mean", "V3_event_amp_cv",
           "V3_frac_cycle_with_1_spike", "V3_frac_cycle_with_2_spikes", "V3_mean_spikes_per_active_neuron_cycle",
           "V3_spikes_per_cycle_spike_weighted", "V3_frac_isi_below_half_period", "V3_active_fraction_per_cycle_mean"]
REFERENCE = "tf_1000_tr_20"
INTERPRETATION = ("DEFERRED: under Revision Plan v3 (section 7) these are retrospective diagnostics of the "
                  "continuous-gain convention; the v1 labels below are computed for the record only and are not "
                  "mapped to manuscript wording before P1b Decision Gate A (ADDENDUM_V3.md).")


def label(row) -> str:
    c, k = row["condition"], row["temporal_gain_scale"]
    if c == "dynamic":
        return "full" if abs(k - 1.0) < 1e-12 else "full_scaled"
    if c == "depletion_only":
        return "neither" if abs(k - 1.0) > 1e-12 else "depletion_unscaled"
    return {"u_output_only": "output_only", "u_release_only": "release_only", "mean_u_release": "mean_release",
            "static_gain": "static", "frozen_profile": "frozen", "baseline": "baseline",
            "facilitation_only": "facilitation_only"}.get(c, c)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    df = pd.read_csv(out / "per_trial_metrics.csv")
    df["combo_id"] = df["combo_id"].fillna("")
    df["model"] = df.apply(label, axis=1)
    keep = [c for c in REPORT if c in df.columns]
    summ = df.groupby(["combo_id", "model"])[keep].agg(["mean", "std", "count"])
    summ.columns = ["_".join(c) for c in summ.columns]
    summ.reset_index().to_csv(out / "per_setting_summary.csv", index=False)
    # paired (by combo and seed) differences against the full model, absolute and relative
    full = df[df.model == "full"].set_index(["combo_id", "seed"])
    rows = []
    for (cid, model), g in df[df.model != "full"].groupby(["combo_id", "model"]):
        for _, r in g.iterrows():
            if (cid, r.seed) not in full.index:
                continue
            f = full.loc[(cid, r.seed)]
            if isinstance(f, pd.DataFrame):
                f = f.iloc[0]
            rec = {"combo_id": cid, "model": model, "seed": int(r.seed),
                   "f_err_hz": r.f_proxy_hz - f.f_proxy_hz}
            for m in keep:
                rec[f"d_{m}"] = r[m] - f[m]
                rec[f"rel_{m}"] = (r[m] - f[m]) / abs(f[m]) if f[m] not in (0,) and np.isfinite(f[m]) else np.nan
            rows.append(rec)
    con = pd.DataFrame(rows)
    con.to_csv(out / "paired_contrasts_vs_full.csv", index=False)
    verdict: dict = {"reference": REFERENCE, "interpretation_status": INTERPRETATION, "rule_inputs": {}}
    if len(con):
        ref = con[con.combo_id == REFERENCE].groupby("model").mean(numeric_only=True)
        def rel(model, m):
            return float(ref.loc[model, f"rel_{m}"]) if model in ref.index else float("nan")
        rel_rel = {m: rel("release_only", m) for m in KEY}
        rel_out = {m: rel("output_only", m) for m in KEY}
        verdict["rule_inputs"]["release_only_rel_diff"] = rel_rel
        verdict["rule_inputs"]["output_only_rel_diff"] = rel_out
        per = con[con.model == "output_only"].groupby("combo_id").mean(numeric_only=True)
        if len(per) >= 5:
            rho, p = spearmanr(per["f_err_hz"].abs(), per["d_M4_modulation_depth"].abs())
        else:
            rho, p = float("nan"), float("nan")
        verdict["rule_inputs"]["spearman_outputonly_abs_ferr_vs_abs_dmod"] = {"rho": float(rho), "p": float(p),
                                                                           "n_settings": int(len(per))}
        a1 = all(np.isfinite(v) and abs(v) <= 0.10 for v in rel_rel.values())
        a2 = all(np.isfinite(v) and abs(v) > 0.25 for v in rel_out.values())
        a3 = bool(np.isfinite(rho) and rho > 0.5)
        verdict["criteria"] = {"release_within_10pct": a1, "output_beyond_25pct": a2, "spearman_gt_0p5": a3}
        verdict["v1_rule_label"] = ("MEDIATOR_SUPPORTED" if (a1 and a2 and a3) else "MEDIATOR_NOT_SUPPORTED")
    else:
        verdict["v1_rule_label"] = "NO_PAIRED_DATA"
    # compact reference-point table (v3 section 7: one SI table)
    ref_rows = df[df.combo_id == REFERENCE]
    if len(ref_rows):
        tab = ref_rows.groupby("model")[keep].agg(["mean", "std", "count"])
        tab.columns = ["_".join(c) for c in tab.columns]
        tab.reset_index().to_csv(out / "reference_point_table.csv", index=False)
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
