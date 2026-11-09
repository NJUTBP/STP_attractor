from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .pav import axial_displacement_deg, circular_axis_mean_deg, circular_axis_sd_deg


@dataclass(frozen=True)
class BootstrapDifference:
    estimate: float
    ci_low: float
    ci_high: float


def _paired_clean(x_a: np.ndarray, x_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(x_a, dtype=float)
    b = np.asarray(x_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("Paired arrays must be one-dimensional and have identical shapes.")
    valid = np.isfinite(a) & np.isfinite(b)
    return a[valid], b[valid]


def paired_bootstrap_sd_difference(
    x_a: np.ndarray,
    x_b: np.ndarray,
    *,
    repeats: int = 10_000,
    seed: int = 12345,
) -> BootstrapDifference:
    a, b = _paired_clean(x_a, x_b)
    if a.size < 2:
        return BootstrapDifference(float("nan"), float("nan"), float("nan"))

    estimate = circular_axis_sd_deg(a) - circular_axis_sd_deg(b)
    rng = np.random.default_rng(seed)
    diffs = np.empty(repeats)
    n = a.size
    for k in range(repeats):
        idx = rng.integers(0, n, n)
        diffs[k] = circular_axis_sd_deg(a[idx]) - circular_axis_sd_deg(b[idx])
    low, high = np.percentile(diffs, [2.5, 97.5])
    return BootstrapDifference(float(estimate), float(low), float(high))


def paired_bootstrap_mean_difference(
    x_a: np.ndarray,
    x_b: np.ndarray,
    *,
    repeats: int = 10_000,
    seed: int = 12345,
) -> BootstrapDifference:
    a, b = _paired_clean(x_a, x_b)
    if a.size < 1:
        return BootstrapDifference(float("nan"), float("nan"), float("nan"))
    paired = a - b
    estimate = float(np.mean(paired))
    rng = np.random.default_rng(seed)
    means = np.empty(repeats)
    n = paired.size
    for k in range(repeats):
        idx = rng.integers(0, n, n)
        means[k] = np.mean(paired[idx])
    low, high = np.percentile(means, [2.5, 97.5])
    return BootstrapDifference(estimate, float(low), float(high))


def paired_bootstrap_circular_mean_difference(
    x_a: np.ndarray,
    x_b: np.ndarray,
    *,
    repeats: int = 10_000,
    seed: int = 12345,
) -> BootstrapDifference:
    """Difference between axial circular means, preserving paired-seed resampling."""
    a, b = _paired_clean(x_a, x_b)
    if a.size < 1:
        return BootstrapDifference(float("nan"), float("nan"), float("nan"))

    def diff(aa: np.ndarray, bb: np.ndarray) -> float:
        ma = circular_axis_mean_deg(aa)
        mb = circular_axis_mean_deg(bb)
        return float(axial_displacement_deg(ma, mb))

    estimate = diff(a, b)
    rng = np.random.default_rng(seed)
    vals = np.empty(repeats)
    n = a.size
    for k in range(repeats):
        idx = rng.integers(0, n, n)
        vals[k] = diff(a[idx], b[idx])
    low, high = np.percentile(vals, [2.5, 97.5])
    return BootstrapDifference(float(estimate), float(low), float(high))


def paired_bootstrap_mean_absolute_displacement_difference(
    x_a: np.ndarray,
    x_b: np.ndarray,
    *,
    repeats: int = 10_000,
    seed: int = 12345,
) -> BootstrapDifference:
    """Exploratory endpoint-position error: mean |axial displacement| difference."""
    a, b = _paired_clean(x_a, x_b)
    return paired_bootstrap_mean_difference(np.abs(a), np.abs(b), repeats=repeats, seed=seed)
