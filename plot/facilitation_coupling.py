from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from config.phase8 import Phase8SweepConfig


def _read(path: Path)->list[dict]:
    with path.open("r",newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _grid(rows, sweep: Phase8SweepConfig, field: str):
    tf=list(map(float,sweep.tau_facilitation_ms)); tr=list(map(float,sweep.tau_recovery_ms))
    lookup={(float(r["tau_facilitation_ms"]),float(r["tau_recovery_ms"])):float(r[field]) for r in rows}
    return np.asarray([[lookup[(a,b)] for b in tr] for a in tf],float),tf,tr


def _heat(rows,sweep,field,title,path):
    z,tf,tr=_grid(rows,sweep,field)
    fig,ax=plt.subplots(figsize=(7.4,5.4))
    im=ax.imshow(z,origin="lower",aspect="auto")
    ax.set_xticks(range(len(tr)),[f"{x:g}" for x in tr])
    ax.set_yticks(range(len(tf)),[f"{x:g}" for x in tf])
    ax.set_xlabel(r"$\tau_{recovery}$ (ms)")
    ax.set_ylabel(r"$\tau_f$ (ms)")
    ax.set_title(title)
    fig.colorbar(im,ax=ax)
    fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def plot_phase10_facilitation_coupling(results_dir: str|Path, sweep: Phase8SweepConfig)->list[Path]:
    root=Path(results_dir); rows=_read(root/"analysis"/"combo_summary.csv")
    out=root/"figures"; out.mkdir(parents=True,exist_ok=True)
    specs=[
        ("u_output_only_abs_frequency_error_hz","Direct-u output only: |frequency error| (Hz)","01_output_only_frequency_error.png"),
        ("u_release_only_abs_frequency_error_hz","Dynamic u->R coupling only: |frequency error| (Hz)","02_release_only_frequency_error.png"),
        ("mean_u_release_abs_frequency_error_hz","Mean-u resource release: |frequency error| (Hz)","03_mean_release_frequency_error.png"),
        ("u_output_only_abs_noise_error_deg","Direct-u output only: |noise-response error| (deg)","04_output_only_noise_error.png"),
        ("u_release_only_abs_noise_error_deg","Dynamic u->R coupling only: |noise-response error| (deg)","05_release_only_noise_error.png"),
        ("mean_u_release_abs_noise_error_deg","Mean-u resource release: |noise-response error| (deg)","06_mean_release_noise_error.png"),
        ("full_mean_spike_release_u","Full STP mean spike-event release u","07_full_mean_release_u.png"),
    ]
    paths=[]
    for field,title,name in specs:
        p=out/name; _heat(rows,sweep,field,title,p); paths.append(p)

    # Reconstruction scatter: one panel per mechanism route.
    full=np.asarray([float(r["full_frequency_hz"]) for r in rows])
    fig,ax=plt.subplots(figsize=(6.2,5.6))
    for mode,label in [
        ("u_output_only","u output only"),
        ("u_release_only","u->R only"),
        ("mean_u_release","mean-u release"),
        ("neither","neither (Phase 9)"),
    ]:
        y=np.asarray([float(r[f"{mode}_frequency_hz"]) for r in rows])
        ax.scatter(full,y,s=25,label=label,alpha=.8)
    lo=min(np.min(full),3); hi=max(np.max(full),25)
    ax.plot([lo,hi],[lo,hi],"--",linewidth=1)
    ax.set_xlabel("Full STP frequency (Hz)")
    ax.set_ylabel("Mechanism-control frequency (Hz)")
    ax.set_title("Which facilitation route reconstructs Full-STP frequency?")
    ax.legend(fontsize=8)
    fig.tight_layout(); p=out/"08_frequency_reconstruction_scatter.png"; fig.savefig(p,dpi=180); plt.close(fig); paths.append(p)

    # Best reconstruction route as integer map; labels stored in a side text file.
    order=["neither","u_output_only","u_release_only","mean_u_release"]
    idx={m:i for i,m in enumerate(order)}
    tf=list(map(float,sweep.tau_facilitation_ms)); tr=list(map(float,sweep.tau_recovery_ms))
    lookup={(float(r["tau_facilitation_ms"]),float(r["tau_recovery_ms"])):idx[r["best_frequency_reconstruction_mode"]] for r in rows}
    z=np.asarray([[lookup[(a,b)] for b in tr] for a in tf],float)
    fig,ax=plt.subplots(figsize=(7.4,5.4))
    im=ax.imshow(z,origin="lower",aspect="auto",vmin=-.5,vmax=len(order)-.5)
    ax.set_xticks(range(len(tr)),[f"{x:g}" for x in tr]); ax.set_yticks(range(len(tf)),[f"{x:g}" for x in tf])
    ax.set_xlabel(r"$\tau_{recovery}$ (ms)"); ax.set_ylabel(r"$\tau_f$ (ms)")
    ax.set_title("Best single-route frequency reconstruction")
    cb=fig.colorbar(im,ax=ax,ticks=range(len(order))); cb.ax.set_yticklabels(order)
    fig.tight_layout(); p=out/"09_best_reconstruction_route.png"; fig.savefig(p,dpi=180); plt.close(fig); paths.append(p)
    return paths
