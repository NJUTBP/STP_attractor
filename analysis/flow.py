from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from analysis.pav import PAVTrace, axial_displacement_deg


@dataclass(frozen=True)
class UnwrappedPAVTrace:
    times_ms: np.ndarray
    displacement_deg: np.ndarray
    length: np.ndarray


@dataclass(frozen=True)
class FlowSummary:
    velocity_1_3_deg_s: float
    velocity_3_5_deg_s: float
    velocity_5_7_deg_s: float
    velocity_6_8_deg_s: float
    shift_1_7_deg: float
    shift_1_8_deg: float
    displacement_1s_deg: float
    displacement_7s_deg: float
    displacement_8s_deg: float


def unwrap_axial_trace(
    trace: PAVTrace,
    *,
    target_axis_deg: float = 90.0,
    start_ms: float = 1000.0,
) -> UnwrappedPAVTrace:
    """Unwrap a 180-degree axial PAV trajectory after a fixed start time.

    The doubled-angle representation is unwrapped first, then divided by two.
    This preserves continuous motion across the +/-90-degree representation
    boundary and can reveal travel exceeding one axial half-turn.
    """
    mask = np.asarray(trace.times_ms, float) >= float(start_ms)
    times = np.asarray(trace.times_ms, float)[mask]
    length = np.asarray(trace.length, float)[mask]
    if times.size < 2:
        raise ValueError("Not enough PAV samples after start_ms to unwrap trajectory.")

    wrapped = axial_displacement_deg(np.asarray(trace.axis_deg, float)[mask], target_axis_deg)
    doubled_phase = np.deg2rad(2.0 * wrapped)
    unwrapped = 0.5 * np.rad2deg(np.unwrap(doubled_phase))
    return UnwrappedPAVTrace(times_ms=times, displacement_deg=unwrapped, length=length)


def linear_velocity_deg_s(trace: UnwrappedPAVTrace, start_ms: float, end_ms: float) -> float:
    mask = (trace.times_ms >= start_ms) & (trace.times_ms <= end_ms)
    if np.sum(mask) < 3:
        return float("nan")
    t_s = trace.times_ms[mask] / 1000.0
    y = trace.displacement_deg[mask]
    slope, _ = np.polyfit(t_s, y, 1)
    return float(slope)


def _interp(trace: UnwrappedPAVTrace, time_ms: float) -> float:
    return float(np.interp(time_ms, trace.times_ms, trace.displacement_deg))


def summarize_flow(trace: UnwrappedPAVTrace) -> FlowSummary:
    d1 = _interp(trace, 1000.0)
    d7 = _interp(trace, 7000.0)
    d8 = _interp(trace, 8000.0)
    return FlowSummary(
        velocity_1_3_deg_s=linear_velocity_deg_s(trace, 1000.0, 3000.0),
        velocity_3_5_deg_s=linear_velocity_deg_s(trace, 3000.0, 5000.0),
        velocity_5_7_deg_s=linear_velocity_deg_s(trace, 5000.0, 7000.0),
        velocity_6_8_deg_s=linear_velocity_deg_s(trace, 6000.0, 8000.0),
        shift_1_7_deg=d7 - d1,
        shift_1_8_deg=d8 - d1,
        displacement_1s_deg=d1,
        displacement_7s_deg=d7,
        displacement_8s_deg=d8,
    )
