"""P5: Fig. 1 off-bump firing diagnosis from EXISTING runs (Revision Plan v3 section 10).  No new simulation.

From the code root:
  python revision_r1/p5_fig1_diagnostics/run_p5.py ^
      --phase6 results/phase6_corrected_full ^
      --phase13 results/phase13_attractor_confirmation ^
      --module-c results/phase16/module_C/branches ^
      --fig1-csv make_figures/source_data/Fig1B_E_spikes.csv ^
      --out results/revision_r1/p5_fig1 --workers 8
Any source may be omitted; at least one is needed.  --plan lists the files without computing.

Sources and windows
  phase6   theta/{dynamic,static_gain}  (0 pA, seeds 0-9) and attractor/{dynamic,static_gain} (60 pA), 1-8 s
  phase13  combos/tf_1000_tr_20/noise_{000,030,060,090}/{dynamic,dynamic_static}, 16 trials per level, 1-8 s
  moduleC  branches/tf_1000_tr_20__{dynamic,static_gain}__seed*__no_shift__noise{30,60}__draw{0,1}.npz,
           10-s continuations from noise-free states; 0.5-10 s after the continuation start
  fig1     the E spikes of the Fig. 1 trial (source-data CSV), 1-8 s
Outputs: per_run_metrics.csv, summary_by_condition.csv, rate_hist.csv, p5_fig1_diagnostics.png
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

NE = 1024
MODEL_OF = {"dynamic": "full", "static_gain": "static"}
KEY_METRICS = ["all_frac_silent", "all_rate_q50", "all_rate_q90", "all_frac_ge_0p1hz", "off_median_rate_hz",
               "off_mean_rate_hz", "off_q90_rate_hz", "off_frac_any_spike", "off_frac_ge_0p1hz", "in_mean_rate_hz",
               "in_off_ratio", "active_frac_per_cycle_mean", "off_active_frac_per_cycle_mean",
               "bump_frac_cycles_0", "bump_frac_cycles_1", "bump_frac_cycles_2", "bump_frac_cycles_3plus",
               "bump_mean_spikes_per_cycle", "active_pairs_frac_1", "active_pairs_frac_2", "active_pairs_frac_3plus",
               "active_pairs_mean", "f0_pop_hz", "f_plfp_hz", "resultant_all_spikes", "center_change_deg"]


def collect(a) -> list[dict]:
    items = []
    if a.phase6:
        r = Path(a.phase6)
        for sub, ds in (("theta", "phase6_theta"), ("attractor", "phase6_attractor")):
            for cond in ("dynamic", "static_gain"):
                for f in sorted((r / sub / cond).glob("seed_*.npz")):
                    items.append(dict(kind="npz", path=str(f), dataset=ds, t0=1000.0, t1=8000.0))
    if a.phase13:
        base = Path(a.phase13) / "combos" / "tf_1000_tr_20"
        for nd in sorted(base.glob("noise_*")):
            for cond_dir in ("dynamic", "dynamic_static"):
                for f in sorted((nd / cond_dir).glob("seed_*.npz")):
                    items.append(dict(kind="npz", path=str(f), dataset="phase13", t0=1000.0, t1=8000.0))
    if a.module_c:
        for f in sorted(Path(a.module_c).glob("tf_1000_tr_20__*__no_shift__noise*__draw*.npz")):
            items.append(dict(kind="branch", path=str(f), dataset="moduleC", t0=500.0, t1=10000.0))
    if a.fig1_csv:
        items.append(dict(kind="csv", path=str(a.fig1_csv), dataset="fig1_source", t0=1000.0, t1=8000.0))
    return items


def _task(it: dict) -> tuple[dict, dict]:
    from revision_r1.p5_fig1_diagnostics.firing_metrics import firing_metrics, rate_histogram
    meta = {k: it[k] for k in ("dataset", "path")}
    st = pl = None
    if it["kind"] == "npz":
        with np.load(it["path"], allow_pickle=False) as z:
            t, ang = z["spike_times_e_ms"], z["spike_angles_e_deg"]
            cond = str(z["condition"].item())
            meta.update(condition=cond, seed=int(z["seed"]), noise_pa=float(z["noise_pa"]),
                        temporal_gain_scale=float(z["temporal_gain_scale"]) if "temporal_gain_scale" in z.files else 1.0)
            st, pl = z["sample_times_ms"], z["plfp_gaba_e"]
    elif it["kind"] == "branch":
        with np.load(it["path"], allow_pickle=False) as z:
            t, ang = z["spike_t"].astype(float), z["spike_a"].astype(float)
            m = json.loads(str(z["meta"]))
            cond = m["condition"]
            meta.update(condition=cond, seed=int(m["seed"]), noise_pa=float(m["noise"]), draw=int(m["draw"]))
    else:
        d = pd.read_csv(it["path"])
        d = d[d["population"].astype(str).str.lower() == "e"]
        t, ang = d["time_s"].to_numpy(float) * 1000.0, d["angle_deg"].to_numpy(float)
        cond = "dynamic"
        meta.update(condition=cond, seed=-1, noise_pa=0.0)
    meta["model"] = MODEL_OF.get(cond, cond)
    row, arr = firing_metrics(t, ang, NE, it["t0"], it["t1"], sample_times_ms=st, plfp=pl)
    row.update(meta)
    hist = {"all": rate_histogram(arr["rates"]), "off": rate_histogram(arr["off_rates"]),
            "bump_counts": arr.get("bump_counts", np.zeros(6, int))}
    return row, hist


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "model", "noise_pa"]
    cols = [c for c in KEY_METRICS if c in df.columns]
    g = df.replace([np.inf], np.nan).groupby(keys)
    s = g[cols].agg(["mean", "min", "max"])
    s.columns = ["_".join(c) for c in s.columns]
    s.insert(0, "n_runs", g.size())
    return s.reset_index()


def plot(df: pd.DataFrame, hists: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    from revision_r1.p5_fig1_diagnostics.firing_metrics import RATE_EDGES
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.5))
    full = df[df.model == "full"]
    groups = sorted({(r.dataset, r.noise_pa) for r in full.itertuples()}, key=lambda x: (x[1], x[0]))
    for k, (ds, nz) in enumerate(groups):
        h = hists.get((ds, "full", nz))
        if h is None:
            continue
        for axi, key in ((ax[0, 0], "all"), (ax[0, 1], "off")):
            hh = h[key].astype(float)
            if hh.sum() == 0:
                continue
            cdf = np.cumsum(hh) / hh.sum()
            axi.step(np.r_[0.005, RATE_EDGES[1:]], np.r_[0, cdf], where="post", label=f"{ds} {nz:g} pA", color=f"C{k}")
    for axi, title in ((ax[0, 0], "all E neurons"), (ax[0, 1], "off-bump neurons (> 45 deg)")):
        axi.set_xscale("log"); axi.set_xlabel("firing rate (Hz); silent plotted at 0.005"); axi.set_ylabel("CDF")
        axi.set_title(f"full model: {title}", fontsize=9); axi.legend(fontsize=6)
    w = 0.8 / max(len(groups), 1)
    for k, (ds, nz) in enumerate(groups):
        h = hists.get((ds, "full", nz))
        if h is None:
            continue
        bc = h["bump_counts"].astype(float)
        if bc.sum():
            ax[1, 0].bar(np.arange(6) + k * w - 0.4, bc / bc.sum(), width=w, label=f"{ds} {nz:g} pA", color=f"C{k}")
    ax[1, 0].set_xticks(range(6), ["0", "1", "2", "3", "4", "5+"])
    ax[1, 0].set_xlabel("spikes per cycle, bump-core neurons (<= 20 deg)"); ax[1, 0].set_ylabel("fraction")
    ax[1, 0].legend(fontsize=6)
    for model, mk in (("full", "o"), ("static", "s")):
        sub = df[df.model == model].groupby(["dataset", "noise_pa"])
        m = sub[["off_median_rate_hz", "off_frac_any_spike", "active_frac_per_cycle_mean"]].mean().reset_index()
        ax[1, 1].scatter(m.noise_pa, m.off_frac_any_spike, marker=mk, label=f"{model}: off-bump frac. with >= 1 spike")
        ax[1, 1].scatter(m.noise_pa, m.active_frac_per_cycle_mean, marker=mk, facecolors="none",
                         edgecolors="k", label=f"{model}: active fraction per cycle")
    ax[1, 1].set_xlabel("noise amplitude (pA)"); ax[1, 1].set_ylim(-0.02, 1.02); ax[1, 1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase6"); ap.add_argument("--phase13"); ap.add_argument("--module-c")
    ap.add_argument("--fig1-csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args(argv)
    items = collect(a)
    if not items:
        raise SystemExit("no input runs found; check the source paths")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(items).to_csv(out / "inputs.csv", index=False)
    print(pd.DataFrame(items).groupby("dataset").size().to_string())
    if a.plan:
        return 0
    rows, hists = [], {}
    if a.workers <= 1:
        res = [_task(it) for it in items]
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            res = [f.result() for f in as_completed([ex.submit(_task, it) for it in items])]
    for row, h in res:
        rows.append(row)
        key = (row["dataset"], row["model"], row["noise_pa"])
        acc = hists.setdefault(key, {k: np.zeros_like(v) for k, v in h.items()})
        for k, v in h.items():
            acc[k] = acc[k] + v
    df = pd.DataFrame(rows).sort_values(["dataset", "model", "noise_pa", "seed"])
    df.to_csv(out / "per_run_metrics.csv", index=False)
    summ = summarize(df)
    summ.to_csv(out / "summary_by_condition.csv", index=False)
    from revision_r1.p5_fig1_diagnostics.firing_metrics import RATE_EDGES
    hrows = []
    for (ds, model, nz), h in hists.items():
        for k in ("all", "off"):
            for lo, hi, n in zip(RATE_EDGES[:-1], RATE_EDGES[1:], h[k]):
                hrows.append(dict(dataset=ds, model=model, noise_pa=nz, population=k, rate_lo=lo, rate_hi=hi, n=int(n)))
    pd.DataFrame(hrows).to_csv(out / "rate_hist.csv", index=False)
    plot(df, hists, out / "p5_fig1_diagnostics.png")
    cols = ["dataset", "model", "noise_pa", "n_runs", "all_frac_silent_mean", "off_median_rate_hz_mean",
            "off_frac_any_spike_mean", "in_mean_rate_hz_mean", "bump_frac_cycles_1_mean",
            "active_frac_per_cycle_mean_mean"]
    print(summ[[c for c in cols if c in summ.columns]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
