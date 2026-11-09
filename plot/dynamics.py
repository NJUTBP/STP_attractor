from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from analysis.flow import UnwrappedPAVTrace
from analysis.oscillation import normalized_autocorrelation
from model.result import SimulationResult


def plot_unwrapped_pav(
    traces: dict[str, list[UnwrappedPAVTrace]],
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for condition, cond_traces in traces.items():
        for tr in cond_traces:
            ax.plot(tr.times_ms / 1000.0, tr.displacement_deg, alpha=0.16, linewidth=0.8)
        if cond_traces:
            arr = np.stack([tr.displacement_deg for tr in cond_traces], axis=0)
            ax.plot(cond_traces[0].times_ms / 1000.0, np.mean(arr, axis=0), linewidth=2.2, label=condition)
    ax.axhline(0.0, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Unwrapped axial PAV displacement (deg)")
    ax.set_title("Zero-noise attractor motion (thin: trials; thick: condition mean)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_plfp_segment(
    results: dict[str, SimulationResult],
    path: str | Path,
    *,
    start_ms: float = 2000.0,
    end_ms: float = 3000.0,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for condition, result in results.items():
        mask = (result.sample_times_ms >= start_ms) & (result.sample_times_ms <= end_ms)
        x = result.plfp_gaba_e[mask].astype(float)
        if x.size == 0:
            continue
        sd = float(np.std(x))
        z = (x - np.mean(x)) / sd if sd > 0 else x - np.mean(x)
        ax.plot(result.sample_times_ms[mask] / 1000.0, z, linewidth=1.1, label=condition)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("pLFP (z-scored within segment)")
    ax.set_title("Zero-noise pLFP waveform, seed 0")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_plfp_acf(
    results: dict[str, SimulationResult],
    path: str | Path,
    *,
    analysis_start_ms: float = 1000.0,
    analysis_end_ms: float = 8000.0,
    max_lag_s: float = 0.25,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for condition, result in results.items():
        mask = (result.sample_times_ms >= analysis_start_ms) & (result.sample_times_ms <= analysis_end_ms)
        fs = 1000.0 / float(np.median(np.diff(result.sample_times_ms)))
        lags, acf = normalized_autocorrelation(result.plfp_gaba_e[mask], fs, max_lag_s=max_lag_s)
        ax.plot(lags * 1000.0, acf, linewidth=1.4, label=condition)
    ax.axhline(0.0, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Normalized autocorrelation")
    ax.set_title("Zero-noise pLFP autocorrelation, seed 0")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
