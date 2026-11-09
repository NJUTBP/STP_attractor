from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from model.result import SimulationResult


def plot_efficacy_profiles(results_by_condition: dict[str, list[SimulationResult]], output: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for condition, results in results_by_condition.items():
        if not results:
            continue
        theta = results[0].theta_e_deg
        profiles = np.stack([r.mean_efficacy_by_e for r in results])
        ax.plot(theta, profiles.mean(axis=0), label=condition)
    ax.set_xlabel("Preferred angle (deg)")
    ax.set_ylabel("Mean E→I efficacy multiplier")
    ax.set_xlim(0, 360)
    ax.legend()
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
