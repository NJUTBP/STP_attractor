from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from config.phase8 import Phase8SweepConfig


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _matrix(rows: list[dict], sweep: Phase8SweepConfig, key: str) -> np.ndarray:
    tf = list(sweep.tau_facilitation_ms)
    tr = list(sweep.tau_recovery_ms)
    out = np.full((len(tr), len(tf)), np.nan)
    lookup = {(float(r["tau_facilitation_ms"]), float(r["tau_recovery_ms"])): r for r in rows}
    for j, rv in enumerate(tr):
        for i, fv in enumerate(tf):
            row = lookup.get((float(fv), float(rv)))
            if row is not None and row.get(key, "") != "":
                out[j, i] = float(row[key])
    return out


def _heatmap(matrix: np.ndarray, sweep: Phase8SweepConfig, title: str, label: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    im = ax.imshow(matrix, origin="lower", aspect="auto")
    ax.set_xticks(np.arange(len(sweep.tau_facilitation_ms)))
    ax.set_xticklabels([f"{x:g}" for x in sweep.tau_facilitation_ms])
    ax.set_yticks(np.arange(len(sweep.tau_recovery_ms)))
    ax.set_yticklabels([f"{x:g}" for x in sweep.tau_recovery_ms])
    ax.set_xlabel("tau_facilitation (ms)")
    ax.set_ylabel("tau_recovery (ms)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_phase8_mechanism_map(results_dir: str | Path, sweep: Phase8SweepConfig) -> list[Path]:
    root = Path(results_dir)
    rows = _read_csv(root / "analysis" / "combo_summary.csv")
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    specs = [
        ("dynamic_frequency_hz", "Full STP: dominant frequency", "Hz", "01_full_dynamic_frequency.png"),
        ("dynamic_temporal_slowing_hz", "Full STP: temporal slowing beyond matched mean gain", "Hz", "02_full_temporal_slowing.png"),
        ("facilitation_only_temporal_slowing_hz", "Facilitation-only: temporal slowing beyond matched mean gain", "Hz", "03_facilitation_temporal_slowing.png"),
        ("depletion_only_temporal_slowing_hz", "Depletion-only: temporal slowing beyond matched mean gain", "Hz", "04_depletion_temporal_slowing.png"),
        ("dynamic_frozen_frequency_hz", "Frozen full-STP spatial profile: dominant frequency", "Hz", "05_frozen_frequency.png"),
        ("dynamic_temporal_vs_frozen_slowing_hz", "Full STP: temporal contribution beyond frozen spatial profile", "Hz", "06_temporal_beyond_spatial.png"),
        ("dynamic_noise_resistance_gain_deg", "Full STP: 60-pA noise-resistance gain vs matched static", "deg", "07_full_noise_resistance.png"),
        ("facilitation_only_noise_resistance_gain_deg", "Facilitation-only: 60-pA noise-resistance gain", "deg", "08_facilitation_noise_resistance.png"),
        ("depletion_only_noise_resistance_gain_deg", "Depletion-only: 60-pA noise-resistance gain", "deg", "09_depletion_noise_resistance.png"),
        ("dynamic_theta_fraction", "Full STP: fraction of seeds in theta band", "fraction", "10_full_theta_fraction.png"),
        ("dynamic_stationary_fraction", "Full STP: stationary zero-noise seed fraction", "fraction", "11_full_stationary_fraction.png"),
        ("dynamic_p_match_theta", "Full STP: matched mean E->I gain (0 pA)", "p_match", "12_full_pmatch.png"),
    ]
    written = []
    for key, title, label, filename in specs:
        if any(key in r and r[key] != "" for r in rows):
            path = out / filename
            _heatmap(_matrix(rows, sweep, key), sweep, title, label, path)
            written.append(path)
    return written
