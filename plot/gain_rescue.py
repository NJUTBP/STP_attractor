from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from config.phase8 import Phase8SweepConfig


def _rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _matrix(rows: list[dict], sweep: Phase8SweepConfig, key: str) -> np.ndarray:
    out = np.full((len(sweep.tau_recovery_ms), len(sweep.tau_facilitation_ms)), np.nan)
    lookup = {(float(r["tau_facilitation_ms"]), float(r["tau_recovery_ms"])): r for r in rows}
    for j, tr in enumerate(sweep.tau_recovery_ms):
        for i, tf in enumerate(sweep.tau_facilitation_ms):
            r = lookup.get((float(tf), float(tr)))
            if r is not None:
                out[j, i] = float(r[key])
    return out


def _heat(mat: np.ndarray, sweep: Phase8SweepConfig, title: str, label: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    im = ax.imshow(mat, origin="lower", aspect="auto")
    ax.set_xticks(np.arange(len(sweep.tau_facilitation_ms)))
    ax.set_xticklabels([f"{x:g}" for x in sweep.tau_facilitation_ms])
    ax.set_yticks(np.arange(len(sweep.tau_recovery_ms)))
    ax.set_yticklabels([f"{x:g}" for x in sweep.tau_recovery_ms])
    ax.set_xlabel("tau_facilitation (ms)")
    ax.set_ylabel("tau_recovery (ms)")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax); cb.set_label(label)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=180); plt.close(fig)


def _scatter(x, y, xlabel: str, ylabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(x, y)
    lo = min(np.nanmin(x), np.nanmin(y)); hi = max(np.nanmax(x), np.nanmax(y))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=180); plt.close(fig)


def plot_phase9_gain_rescue(results_dir: str | Path, sweep: Phase8SweepConfig) -> list[Path]:
    root = Path(results_dir); rows = _rows(root / "analysis" / "combo_summary.csv")
    out = root / "figures"; out.mkdir(parents=True, exist_ok=True)
    specs = [
        ("rescue_frequency_hz", "Gain-rescued depletion: dominant frequency", "Hz", "01_rescue_frequency.png"),
        ("rescue_minus_full_frequency_hz", "Gain-rescued depletion minus Full STP frequency", "Hz", "02_frequency_error.png"),
        ("p_match_relative_error", "Final mean-gain matching relative error", "fraction", "03_pmatch_error.png"),
        ("rescue_minus_full_noise_abs_delta_deg", "Gain-rescued depletion minus Full STP noise perturbation", "deg", "04_noise_error.png"),
    ]
    written: list[Path] = []
    for key, title, label, fn in specs:
        p = out / fn; _heat(_matrix(rows, sweep, key), sweep, title, label, p); written.append(p)
    full_f = np.asarray([float(r["full_frequency_hz"]) for r in rows]); res_f = np.asarray([float(r["rescue_frequency_hz"]) for r in rows])
    p = out / "05_frequency_reconstruction_scatter.png"; _scatter(full_f, res_f, "Full STP frequency (Hz)", "Gain-rescued depletion frequency (Hz)", "Theta-frequency reconstruction", p); written.append(p)
    full_n = np.asarray([float(r["full_noise_mean_abs_delta_deg"]) for r in rows]); res_n = np.asarray([float(r["rescue_noise_mean_abs_delta_deg"]) for r in rows])
    p = out / "06_noise_reconstruction_scatter.png"; _scatter(full_n, res_n, "Full STP mean |q60-q0| (deg)", "Gain-rescued depletion mean |q60-q0| (deg)", "Noise-response reconstruction", p); written.append(p)
    return written
