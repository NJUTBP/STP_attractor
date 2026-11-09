"""P2b analysis (PREREGISTRATION.md sections 4-7).  Outputs in <output>/analysis/.

  python revision_r1/p2b_drive_matched/analyze_p2b.py            (after run_p2b.py --execute)
  python revision_r1/p2b_drive_matched/analyze_p2b.py --smoke    (smoke folder; numbers are not results)

per_trial.csv            frequency, theta, peak power, localization, drift, PAV length, spikes, D, C1 efficacy,
                         active fraction per cycle and spikes per active neuron and cycle, for every model and seed
calibration_summary.csv  wave-by-wave parameter, drive and relative error
contrasts.csv            paired frequency differences model - full with bootstrap CIs
p2b_verdict.json         pre-registered verdict and its inputs
p2b_frequency.png        frequencies under C1 (published) and C3 (P2b); drive of every model relative to D_target
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

from revision_r1.common.drive_rules import bootstrap_mean_ci, trial_metrics  # noqa: E402
from revision_r1.p2b_drive_matched.rules_p2b import (  # noqa: E402
    C1_PUBLISHED_HZ, CAL_MODELS, MODELS, P1B_EVENT_STATIC_HZ, c1_efficacy, drive_of_paths, load_p2b_config,
    p2b_verdict)

ORDER = ("full", "release_only", "output_only", "static")
LABEL = {"full": "full", "release_only": "release-only", "output_only": "output-only", "static": "static"}


def _cycle_stats(path, cfg) -> dict:
    """Active fraction per theta cycle and spikes per active (neuron, cycle), as in the P1b/T2 diagnostics."""
    try:
        from revision_r1.common.drive_rules import _replay_stats
        from io_utils import load_result
        cond = load_result(path).condition
        row, _ = _replay_stats(path, cond, cfg)
        keep = ("active_fraction_per_cycle_mean", "mean_spikes_per_active_neuron_cycle", "f0_pop_hz")
        return {k: row.get(k, np.nan) for k in keep}
    except Exception as exc:                                  # diagnostics only; never blocks the verdict
        return {"active_fraction_per_cycle_mean": np.nan, "mean_spikes_per_active_neuron_cycle": np.nan,
                "f0_pop_hz": np.nan, "cycle_stats_error": str(exc)[:200]}


def trial_rows(model: str, paths: dict, cfg, smoke: bool) -> list[dict]:
    from io_utils import load_result
    rows = []
    for seed, path in sorted(paths.items(), key=lambda kv: int(kv[0])):
        r = load_result(path)
        row = dict(model=model, seed=int(seed), path=str(path), condition=r.condition,
                   temporal_gain_scale=float(r.temporal_gain_scale),
                   static_gain=np.nan if r.static_gain is None else float(r.static_gain))
        row.update(trial_metrics(cfg, r, smoke=smoke))
        row["drive_D"] = drive_of_paths([path], cfg)
        row["c1_efficacy"] = c1_efficacy(r)
        row.update(_cycle_stats(path, cfg))
        rows.append(row)
    return rows


def make_figure(df: pd.DataFrame, cal: pd.DataFrame, verdict: dict, D_target: float, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax = axes[0]
    ax.axhspan(4, 12, color="0.92", zorder=0)
    for i, m in enumerate(ORDER):
        f = df[df.model == m].frequency_hz.to_numpy(float)
        jitter = np.linspace(-0.12, 0.12, max(1, f.size))
        ax.plot(i + jitter, f, "o", ms=3.5, color="k", alpha=0.7, zorder=3)
        if f.size:
            ax.plot([i - 0.22, i + 0.22], [np.nanmean(f)] * 2, color="C3", lw=2, zorder=4)
        ax.plot(i + 0.32, C1_PUBLISHED_HZ[m][0], "D", mfc="none", mec="C0", ms=6, zorder=4)
    ax.plot(3 + 0.32, P1B_EVENT_STATIC_HZ[0], "s", mfc="none", mec="C2", ms=6, zorder=4)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([LABEL[m] for m in ORDER], fontsize=8)
    ax.set_ylabel("dominant frequency (Hz)")
    ax.set_title(f"{verdict.get('verdict')} / {verdict.get('static_label')}\n"
                 "dots: C3-matched (full: existing runs); diamonds: C1 (published); square: P1b event static",
                 fontsize=7)
    ax = axes[1]
    for i, m in enumerate(CAL_MODELS):
        h = cal[cal.model == m]
        ax.plot(h.wave.astype(str).str.replace("final", "f"), h.D / D_target, "o-", label=LABEL[m])
    ax.axhline(1.0, color="k", lw=0.8)
    ax.axhspan(0.98, 1.02, color="0.9")
    ax.set_xlabel("calibration wave (0 = existing C1 runs)")
    ax.set_ylabel("C3 drive / D_target")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-ms", type=float, default=1200.0)
    a = ap.parse_args(argv)
    from config import load_config
    pc = load_p2b_config(a.config)
    cfg = load_config(pc.base_config)
    if a.smoke:
        from revision_r1.p2b_drive_matched.run_p2b import smoke_settings
        cfg, pc = smoke_settings(cfg, pc, a.smoke_ms)
    st_path = pc.output / "p2b_stage.json"
    if not st_path.exists():
        raise SystemExit(f"{st_path} not found: run run_p2b.py first")
    st = json.loads(st_path.read_text(encoding="utf-8"))
    if st.get("status") != "P2B_COMPLETE":
        raise SystemExit(f"P2b runs not complete (status {st.get('status')})")
    out = pc.output / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    D_target = float(st["target"]["D_target"])

    # per-trial metrics
    rows = trial_rows("full", st["full_frequency_paths"], cfg, a.smoke)
    items = {it["model"]: it for it in st["models"]}
    for m in CAL_MODELS:
        rows += trial_rows(m, items[m]["final_paths"], cfg, a.smoke)
    df = pd.DataFrame(rows)
    for m in CAL_MODELS:
        df.loc[df.model == m, "calibrated_param"] = items[m]["final_param"]
    df.to_csv(out / "per_trial.csv", index=False)

    # calibration summary
    cal_rows = []
    for m in CAL_MODELS:
        for h in items[m]["history"]:
            cal_rows.append(dict(model=m, wave=str(h["wave"]), param=h["param"], D=h["D"],
                                 rel_error=abs(h["D"] - D_target) / D_target, source=h["dir"]))
        cal_rows.append(dict(model=m, wave="final", param=items[m]["final_param"],
                             D=items[m]["final_D_calibration_seeds"], rel_error=items[m]["final_rel_error"],
                             source="calibration seeds at the final value", valid=items[m]["valid"]))
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(out / "calibration_summary.csv", index=False)

    # paired contrasts and verdict
    S = pc.statistics
    piv = df.pivot_table(index="seed", columns="model", values="frequency_hz")
    con = []
    for m in CAL_MODELS:
        d = (piv[m] - piv["full"]).dropna().to_numpy(float)
        mean, lo, hi = bootstrap_mean_ci(d, int(S["bootstrap_repeats"]), int(S["bootstrap_seed"]))
        con.append(dict(model=m, contrast=f"{m} - full", mean_diff_hz=mean, ci_low=lo, ci_high=hi,
                        abs_mean_diff_hz=abs(mean), n_pairs=int(d.size), calibration_valid=items[m]["valid"]))
    pd.DataFrame(con).to_csv(out / "contrasts.csv", index=False)
    mean_hz = {m: float(df[df.model == m].frequency_hz.mean()) for m in ORDER}
    theta = {m: float(df[df.model == m].theta.mean()) for m in ORDER}
    valid = {m: bool(items[m]["valid"]) for m in CAL_MODELS}
    v = p2b_verdict(mean_hz, theta, valid, pc.decision)
    summary = {m: dict(mean_frequency_hz=mean_hz[m], theta_fraction=theta[m],
                       mean_drive_D=float(df[df.model == m].drive_D.mean()),
                       drive_over_target=float(df[df.model == m].drive_D.mean() / D_target),
                       mean_c1_efficacy=float(df[df.model == m].c1_efficacy.mean()),
                       localization_failure_fraction=float(df[df.model == m].localization_failure.mean()),
                       mean_abs_shift_1_8_deg=float(df[df.model == m].shift_1_8_deg.abs().mean()),
                       active_fraction_per_cycle=float(df[df.model == m].active_fraction_per_cycle_mean.mean()),
                       n_trials=int((df.model == m).sum())) for m in ORDER}
    result = dict(verdict=v, summary=summary, D_target=D_target,
                  calibrated={m: dict(param=items[m]["final_param"], param_name=MODELS[m][1],
                                      rel_error=items[m]["final_rel_error"], valid=items[m]["valid"],
                                      converged=items[m]["converged"]) for m in CAL_MODELS},
                  c1_published_hz={k: v_[0] for k, v_ in C1_PUBLISHED_HZ.items()},
                  p1b_event_static_hz=P1B_EVENT_STATIC_HZ[0], smoke=bool(a.smoke),
                  note="continuous-gain convention; within-model matching-criterion check (v4 P2b)")
    (out / "p2b_verdict.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    make_figure(df, cal, v, D_target, out / "p2b_frequency.png")
    print(json.dumps({"verdict": v["verdict"], "static_label": v["static_label"],
                      "mean_frequency_hz": {k: round(x, 4) for k, x in mean_hz.items()},
                      "drive_over_target": {k: round(summary[k]["drive_over_target"], 4) for k in ORDER}},
                     indent=2))
    if a.smoke:
        print("SMOKE analysis only: these numbers are not results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
