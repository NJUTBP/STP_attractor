"""P3 analysis (PREREGISTRATION.md sections 5-8).  Outputs in <output>/analysis/:
p3_per_u.csv, p3_per_trial.csv, p3_verdict.json, p3_restoration.csv, p3_u_sensitivity.png

  python revision_r1/p3_u_sensitivity/analyze_p3.py
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

from experiment.phase17_physiological import combo_id, config_for, load_phase17_config, read_manifest  # noqa: E402
from revision_r1.p3_u_sensitivity.rules_p3 import (MODES, load_p3_section, overall_verdict,  # noqa: E402
                                                   route_mapping)

LABEL = {"full": "full", "static": "static", "u_output_only": "output-only", "u_release_only": "release-only",
         "mean_u_release": "mean-release"}


def _freqs(cfg, theta_dir, seeds, smoke=False) -> list[dict]:
    from revision_r1.p3_u_sensitivity.run_p3 import trial_rows
    return trial_rows(cfg, Path(theta_dir), seeds, smoke)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).with_name("p3_config.toml")))
    a = ap.parse_args(argv)
    from config import load_config
    from analysis.phase17_physiological import restoration_table
    from phase16.reduced_model import bootstrap_mean_ci
    cfg17 = load_phase17_config(a.config)
    p3 = load_p3_section(a.config)
    out = p3["output"]
    A = out / "analysis"
    A.mkdir(parents=True, exist_ok=True)
    base = load_config(cfg17.base_config)
    seeds = cfg17.seeds
    D = cfg17.decision
    per_u, trials = [], []
    for (u, tf, tr) in cfg17.combos:
        cid = combo_id(u, tf, tr)
        man = read_manifest(out, cid)
        if not man or "raw" not in man:
            continue
        cfg = config_for(base, u, tf, tr)
        full = man["raw"]["full"]
        interp = full["interpretability"]
        row = dict(u=u, combo_id=cid, interpretable=interp["interpretable"], full_theta_fraction=interp["theta_fraction"],
                   full_localization_failure_fraction=interp["localization_failure_fraction"],
                   full_stationary_fraction=interp["stationary_fraction"],
                   full_oscillatory_fraction=interp["oscillatory_fraction"],
                   p_match=full["final_p_match"], mean_release_u=full["mean_release_u"])
        series = {"full": full["per_seed"]}
        sdir = out / "combos" / cid / "raw" / "static" / "theta"
        if all(p.exists() for p in [sdir / f"seed_{s:04d}.npz" for s in seeds]):
            series["static"] = _freqs(cfg, sdir, seeds, p3["smoke"])
        routes = man["raw"].get("routes", {})
        valid = True
        for mode in MODES:
            if mode in routes:
                series[mode] = _freqs(cfg, routes[mode]["theta_dir"], seeds, p3["smoke"])
                row[f"{mode}_scale"] = routes[mode]["final_scale"]
                row[f"{mode}_calibration_rel_error"] = routes[mode]["final_relative_error"]
                row[f"{mode}_calibration_valid"] = bool(routes[mode]["valid"])
                valid &= bool(routes[mode]["valid"])
            else:
                valid = False
        for name, rows in series.items():
            f = np.array([r["frequency_hz"] for r in rows], float)
            row[f"{name}_frequency_hz"] = float(np.nanmean(f))
            row[f"{name}_frequency_sd_hz"] = float(np.nanstd(f, ddof=1)) if np.isfinite(f).sum() > 1 else np.nan
            row[f"{name}_theta_fraction"] = float(np.nanmean([r["theta"] for r in rows]))
            for r in rows:
                trials.append(dict(u=u, model=name, **{k: r[k] for k in ("seed", "frequency_hz", "theta",
                                                                           "normalized_peak_power", "localization_ratio",
                                                                           "localization_failure", "shift_1_8_deg")}))
        row["calibration_valid"] = bool(valid) if routes else None
        if "u_release_only" in series and "u_output_only" in series:
            m = route_mapping(row["full_frequency_hz"], row["u_release_only_frequency_hz"],
                              row["u_output_only_frequency_hz"], D)
            row.update(release_abs_error_hz=m["release_abs_error_hz"], output_abs_error_hz=m["output_abs_error_hz"],
                       mapping_holds=m["holds"])
        else:
            row["mapping_holds"] = None
        if "mean_u_release" in series:
            row["mean_release_abs_error_hz"] = abs(row["mean_u_release_frequency_hz"] - row["full_frequency_hz"])
        if "static" in series:
            row["static_abs_difference_hz"] = abs(row["static_frequency_hz"] - row["full_frequency_hz"])
        per_u.append(row)
    pu = pd.DataFrame(per_u)
    pu.to_csv(A / "p3_per_u.csv", index=False)
    pd.DataFrame(trials).to_csv(A / "p3_per_trial.csv", index=False)
    verdict = overall_verdict(per_u, p3)
    anchor = json.loads((out / "anchor_check.json").read_text(encoding="utf-8")) if (out / "anchor_check.json").exists() else {}
    verdict["anchor_check"] = anchor.get("status")
    verdict["note"] = ("within-model sensitivity of the continuous-gain convention (v3 section 8); "
                       "not a physiological validation")
    # restoration
    R = cfg17.restoration
    rest = {}
    if (out / "restoration").exists():
        rt = restoration_table(out / "restoration", R["readout_ms"], R["window_ms"])
        rt.to_csv(A / "p3_restoration_by_trial.csv", index=False)
        rows = []
        n, sd = int(p3["bootstrap_repeats"]), int(p3["bootstrap_seed"])
        for cid, g in rt[rt.kind == "activity_shift"].groupby("combo_id"):
            u = float(cid.split("_")[1].replace("p", "."))
            for cond, h in g.groupby("condition"):
                m, lo, hi = bootstrap_mean_ci(h.F_2s.values, n, sd)
                rows.append(dict(u=u, combo_id=cid, condition=cond, contrast="", F2s_mean=m, ci_low=lo, ci_high=hi,
                                 n=int(h.F_2s.notna().sum())))
            o = g[g.condition == "u_output_only"].set_index("seed").F_2s
            r_ = g[g.condition == "u_release_only"].set_index("seed").F_2s
            common = sorted(set(o.index) & set(r_.index))
            if common:
                m, lo, hi = bootstrap_mean_ci((o.loc[common] - r_.loc[common]).values, n, sd + 1)
                rows.append(dict(u=u, combo_id=cid, condition="", contrast="output_only - release_only", F2s_mean=m,
                                 ci_low=lo, ci_high=hi, n=len(common)))
        rdf = pd.DataFrame(rows)
        rdf.to_csv(A / "p3_restoration.csv", index=False)
        for u, g in rdf.groupby("u"):
            fr = g[g.condition == "dynamic"]
            c = g[g.contrast != ""]
            rest[str(u)] = dict(
                full_restoration_present=bool(len(fr) and fr.F2s_mean.iloc[0] >= D["restoration_min_F2s"]
                                              and fr.ci_low.iloc[0] > 0),
                output_minus_release_positive=bool(len(c) and c.ci_low.iloc[0] > 0))
    verdict["restoration"] = rest
    (A / "p3_verdict.json").write_text(json.dumps(verdict, indent=2, default=float), encoding="utf-8")
    _figure(pu, A / "p3_u_sensitivity.png", D)
    cols = [c for c in ("u", "interpretable", "full_frequency_hz", "full_theta_fraction", "u_release_only_frequency_hz",
                        "u_output_only_frequency_hz", "mean_u_release_frequency_hz", "static_frequency_hz",
                        "mapping_holds", "calibration_valid") if c in pu.columns]
    print(pu[cols].to_string(index=False))
    print(json.dumps(verdict, indent=2, default=float))
    return 0


def _figure(pu: pd.DataFrame, path: Path, D: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not len(pu):
        return
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for name, mk in (("full", "o"), ("u_release_only", "^"), ("u_output_only", "v"), ("mean_u_release", "s"),
                     ("static", "D")):
        col = f"{name}_frequency_hz"
        if col in pu:
            ax[0].plot(pu.u, pu[col], marker=mk, label=LABEL[name])
    ax[0].axhspan(4, 12, color="0.92", zorder=-1)
    for u, ok in zip(pu.u, pu.interpretable):
        if not ok:
            ax[0].axvline(u, color="r", lw=0.6, ls=":")
    ax[0].set_xlabel("U"); ax[0].set_ylabel("dominant frequency (Hz)"); ax[0].legend(fontsize=7)
    ax[0].set_title("4-trial means; gray: theta band; red dotted: not interpretable", fontsize=8)
    for col, mk, lab in (("release_abs_error_hz", "^", "|release-only - full|"),
                         ("output_abs_error_hz", "v", "|output-only - full|")):
        if col in pu:
            ax[1].plot(pu.u, pu[col], marker=mk, label=lab)
    ax[1].axhline(D["release_route_max_abs_error_hz"], color="C0", ls="--", lw=0.8)
    ax[1].axhline(D["output_route_min_abs_error_hz"], color="C1", ls="--", lw=0.8)
    ax[1].set_xlabel("U"); ax[1].set_ylabel("absolute frequency error (Hz)"); ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
