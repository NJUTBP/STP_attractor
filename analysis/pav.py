from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from config.schema import AnalysisConfig, SimulationConfig
from model.result import SimulationResult


@dataclass(frozen=True)
class PAVTrace:
    times_ms: np.ndarray
    axis_deg: np.ndarray
    length: np.ndarray


@dataclass(frozen=True)
class AttractorSummary:
    seed: int
    condition: str
    noise_pa: float
    pav_deg_at_target: float
    displacement_deg: float
    localization_ratio: float
    localization_failure: bool


def compute_axial_pav(
    spike_times_ms: np.ndarray,
    spike_angles_deg: np.ndarray,
    sim_cfg: SimulationConfig,
    ana_cfg: AnalysisConfig,
) -> PAVTrace:
    """Compute doubled-angle PAV from spikes in a sliding window, updated every pav_step_ms."""
    step = ana_cfg.pav_step_ms
    window = ana_cfg.pav_window_ms
    n_bins = int(np.ceil(sim_cfg.duration_ms / step))
    edges = np.arange(n_bins + 1, dtype=float) * step

    doubled = np.deg2rad(2.0 * spike_angles_deg)
    cos_hist, _ = np.histogram(spike_times_ms, bins=edges, weights=np.cos(doubled))
    sin_hist, _ = np.histogram(spike_times_ms, bins=edges, weights=np.sin(doubled))
    count_hist, _ = np.histogram(spike_times_ms, bins=edges)

    ccos = np.concatenate([[0.0], np.cumsum(cos_hist)])
    csin = np.concatenate([[0.0], np.cumsum(sin_hist)])
    ccount = np.concatenate([[0.0], np.cumsum(count_hist)])
    w = max(1, int(round(window / step)))

    idx = np.arange(1, n_bins + 1)
    start = np.maximum(0, idx - w)
    vx = ccos[idx] - ccos[start]
    vy = csin[idx] - csin[start]
    counts = ccount[idx] - ccount[start]

    doubled_angle = np.mod(np.rad2deg(np.arctan2(vy, vx)), 360.0)
    axis = doubled_angle / 2.0
    length = np.divide(
        np.sqrt(vx * vx + vy * vy),
        counts,
        out=np.zeros_like(vx),
        where=counts > 0,
    )
    return PAVTrace(times_ms=edges[1:], axis_deg=axis, length=length)


def axial_displacement_deg(axis_deg: float | np.ndarray, target_axis_deg: float = 90.0) -> np.ndarray:
    delta = np.asarray(axis_deg, dtype=float) - target_axis_deg
    return 0.5 * np.rad2deg(np.arctan2(np.sin(np.deg2rad(2.0 * delta)), np.cos(np.deg2rad(2.0 * delta))))


def _window_median(times: np.ndarray, values: np.ndarray, start: float, end: float) -> float:
    mask = (times >= start) & (times <= end)
    if not np.any(mask):
        return float("nan")
    return float(np.nanmedian(values[mask]))


def summarize_attractor(
    result: SimulationResult,
    sim_cfg: SimulationConfig,
    ana_cfg: AnalysisConfig,
) -> tuple[AttractorSummary, PAVTrace]:
    trace = compute_axial_pav(
        result.spike_times_e_ms,
        result.spike_angles_e_deg,
        sim_cfg,
        ana_cfg,
    )
    target_idx = int(np.argmin(np.abs(trace.times_ms - ana_cfg.pav_target_time_ms)))
    q = float(trace.axis_deg[target_idx])
    displacement = float(axial_displacement_deg(q, ana_cfg.target_axis_deg))

    ref = _window_median(
        trace.times_ms,
        trace.length,
        ana_cfg.localization_reference_start_ms,
        ana_cfg.localization_reference_end_ms,
    )
    late = _window_median(
        trace.times_ms,
        trace.length,
        ana_cfg.localization_target_start_ms,
        ana_cfg.localization_target_end_ms,
    )
    ratio = late / ref if np.isfinite(ref) and ref > 0 else float("nan")
    failure = (not np.isfinite(ratio)) or ratio < ana_cfg.localization_ratio_cutoff

    summary = AttractorSummary(
        seed=result.seed,
        condition=result.condition,
        noise_pa=result.noise_pa,
        pav_deg_at_target=q,
        displacement_deg=displacement,
        localization_ratio=float(ratio),
        localization_failure=bool(failure),
    )
    return summary, trace


def circular_axis_mean_deg(displacements_deg: np.ndarray) -> float:
    """Axial circular mean in degrees, returned in [-90, 90).

    Endpoint displacements live on a 180-degree axis, so -90 and +90 are
    equivalent boundary representations.  Averaging them linearly would be
    wrong; doubled-angle circular averaging preserves that equivalence.
    """
    x = np.asarray(displacements_deg, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    phi = np.deg2rad(2.0 * x)
    mean_phi = np.arctan2(np.mean(np.sin(phi)), np.mean(np.cos(phi)))
    mean_axis = 0.5 * np.rad2deg(mean_phi)
    return float(axial_displacement_deg(mean_axis, 0.0))


def circular_axis_sd_deg(displacements_deg: np.ndarray) -> float:
    """Sample SD of shortest axial residuals around the axial circular mean.

    This reduces to the ordinary sample SD for a compact distribution away
    from the +/-90-degree representation boundary, but remains correct when a
    compact cluster straddles that boundary.
    """
    x = np.asarray(displacements_deg, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    center = circular_axis_mean_deg(x)
    residual = axial_displacement_deg(x, center)
    return float(np.std(residual, ddof=1))
