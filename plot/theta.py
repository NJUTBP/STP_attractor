from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from analysis.theta import PSDResult


def plot_mean_psd(psd_by_condition: dict[str, list[PSDResult]], output: str | Path, max_hz: float = 50.0) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for condition, psds in psd_by_condition.items():
        if not psds:
            continue
        freq = psds[0].frequency_hz
        stack = np.stack([p.psd for p in psds])
        mean = stack.mean(axis=0)
        mask = freq <= max_hz
        ax.plot(freq[mask], mean[mask], label=condition)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("pLFP PSD")
    ax.legend()
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
