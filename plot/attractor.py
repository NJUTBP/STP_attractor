from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from analysis.pav import AttractorSummary, circular_axis_mean_deg, circular_axis_sd_deg


def plot_endpoint_distributions(summaries: list[AttractorSummary], output: str | Path) -> None:
    conditions = ["dynamic", "baseline", "static_gain"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rng = np.random.default_rng(0)
    for x, condition in enumerate(conditions):
        rows = [r for r in summaries if r.condition == condition]
        valid = np.array([not r.localization_failure for r in rows])
        y = np.array([r.displacement_deg for r in rows], dtype=float)
        jitter = rng.normal(0, 0.035, len(rows))
        if np.any(valid):
            ax.scatter(np.full(valid.sum(), x) + jitter[valid], y[valid], s=24, label=condition if x == 0 else None)
            mean = circular_axis_mean_deg(y[valid])
            sd = circular_axis_sd_deg(y[valid]) if valid.sum() > 1 else 0.0
            ax.errorbar(x, mean, yerr=sd, fmt="o", capsize=5)
        if np.any(~valid):
            ax.scatter(np.full((~valid).sum(), x) + jitter[~valid], y[~valid], marker="x", s=36)
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(range(3), conditions)
    ax.set_ylabel("Axial displacement at 7 s (deg)")
    ax.set_title("Valid trials: dots; localization failures: ×")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
