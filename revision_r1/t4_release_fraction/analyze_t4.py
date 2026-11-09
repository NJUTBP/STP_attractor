"""R1-T4 analysis: F(rho, p; tau_r) curves, fixed-point predictions for the 21 Phase-10 settings, verdict.

  python revision_r1/t4_release_fraction/analyze_t4.py --out results/revision_r1/t4_release_fraction \
      [--t2-summary results/revision_r1/t2_efficacy/per_setting_summary.csv]
Primary prediction: (rho, p) interpolation with the row's own p_match; sensitivity: main curve only.
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

from revision_r1.t2_efficacy_stats.metrics import u_plus_periodic  # noqa: E402
from revision_r1.t4_release_fraction.fixed_point import make_F, solve_fixed_point  # noqa: E402
from revision_r1.t4_release_fraction.run_t4 import PHASE10, TR_MAIN, row_targets  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase10", default=str(PHASE10))
    ap.add_argument("--t2-summary", default=None, help="sensitivity only: spikes per active neuron per cycle from T2")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _t2spc for the sensitivity run")
    a = ap.parse_args(argv)
    out = Path(a.out)
    cells = pd.read_csv(out / "cells.csv")
    valid = cells[cells.calibration_valid & np.isfinite(cells.freq_hz)]
    p10 = pd.read_csv(a.phase10)
    spc = {}
    if a.t2_summary and Path(a.t2_summary).exists():
        t2 = pd.read_csv(a.t2_summary)
        t2 = t2[t2.model == "full"]
        spc = dict(zip(t2.combo_id, t2["spikes_per_active_neuron_per_cycle_mean"]))
    targets = row_targets(a.phase10)
    rows = []
    for tr in TR_MAIN:
        tab = valid[np.isclose(valid.tau_r, tr)]
        F = make_F(tab, targets[tr]["main"])
        F_main = make_F(tab[np.isclose(tab.p_target, targets[tr]["main"])], targets[tr]["main"])
        for _, s in p10[np.isclose(p10.tau_recovery_ms, tr)].iterrows():
            cid = s.combo_id
            n_sp = float(spc.get(cid, 1.0)) if np.isfinite(spc.get(cid, 1.0)) else 1.0
            f_pred, u_pred = solve_fixed_point(F, 0.4, s.tau_facilitation_ms, s.full_p_match_theta, n_sp)
            f_main, _ = solve_fixed_point(F_main, 0.4, s.tau_facilitation_ms, targets[tr]["main"], n_sp)
            rows.append({"combo_id": cid, "tau_f_ms": s.tau_facilitation_ms, "tau_r_ms": tr,
                         "p_match_full": s.full_p_match_theta, "spikes_per_cycle_used": n_sp,
                         "f_full_hz": s.full_frequency_hz, "f_pred_hz": f_pred, "abs_err_hz": abs(f_pred - s.full_frequency_hz),
                         "f_pred_main_curve_hz": f_main, "abs_err_main_hz": abs(f_main - s.full_frequency_hz),
                         "u_plus_pred": u_pred, "u_rel_full": s.full_mean_spike_release_u,
                         "F_at_U_pred_hz": F(0.4, s.full_p_match_theta),
                         "f_output_only_hz": s.u_output_only_frequency_hz, "f_neither_hz": s.neither_frequency_hz})
    pred = pd.DataFrame(rows)
    pred.to_csv(out / f"fixed_point_predictions{a.tag}.csv", index=False)
    err = pred.abs_err_hz.to_numpy(float)
    n_ok = int(np.sum(err <= 1.0)); med = float(np.nanmedian(err))
    verdict = {"n_settings": int(len(pred)), "n_predicted": int(np.isfinite(err).sum()),
               "median_abs_err_hz": med, "n_within_1hz": n_ok,
               "median_abs_err_main_curve_hz": float(np.nanmedian(pred.abs_err_main_hz)),
               "verdict": "FIXED_POINT_SUPPORTED" if (med <= 0.5 and n_ok >= 15) else "FIXED_POINT_NOT_SUPPORTED",
               "failures": pred[~(pred.abs_err_hz <= 1.0)][["combo_id", "f_full_hz", "f_pred_hz"]].to_dict("records")}
    (out / f"verdict{a.tag}.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
