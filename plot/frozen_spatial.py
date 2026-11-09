from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from analysis.theta import PSDResult


def plot_frozen_profile(
    theta_e_deg: np.ndarray,
    profile: np.ndarray,
    *,
    baseline: float,
    scalar_p_match: float | None,
    output: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(theta_e_deg, profile, label="frozen spatial profile")
    ax.axhline(baseline, linestyle="--", label=f"baseline U={baseline:.3f}")
    if scalar_p_match is not None:
        ax.axhline(float(scalar_p_match), linestyle=":", label=f"scalar p_match={float(scalar_p_match):.3f}")
    ax.set_xlabel("Preferred angle (deg)")
    ax.set_ylabel("E→I efficacy multiplier")
    ax.set_xlim(0, 360)
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_theta_psd_with_frozen(
    psd_by_condition: dict[str, list[PSDResult]],
    output: str | Path,
    *,
    max_hz: float = 50.0,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = ("dynamic", "frozen_profile", "static_gain", "baseline")
    for condition in order:
        rows = psd_by_condition.get(condition, [])
        if not rows:
            continue
        freq = rows[0].frequency_hz
        psd = np.stack([r.psd for r in rows])
        mean_psd = psd.mean(axis=0)
        mask = freq <= max_hz
        ax.plot(freq[mask], mean_psd[mask], label=condition)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean PSD")
    ax.legend()
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
