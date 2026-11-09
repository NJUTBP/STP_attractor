import numpy as np

from analysis.pav import axial_displacement_deg, compute_axial_pav
from config.schema import AnalysisConfig, SimulationConfig


def test_axial_displacement_wraps_at_180():
    x = axial_displacement_deg(np.array([179.0, 1.0]), target_axis_deg=0.0)
    assert np.allclose(x, [-1.0, 1.0], atol=1e-6)


def test_doubled_angle_pav_combines_symmetric_bumps():
    sim = SimulationConfig(dt_ms=0.1, duration_ms=100.0, cue_end_ms=20.0, sampling_rate_hz=200)
    ana = AnalysisConfig(
        pav_window_ms=20.0,
        pav_step_ms=5.0,
        pav_target_time_ms=80.0,
        localization_reference_start_ms=20.0,
        localization_reference_end_ms=40.0,
        localization_target_start_ms=60.0,
        localization_target_end_ms=90.0,
        theta_start_ms=20.0,
        theta_end_ms=100.0,
        gain_match_start_ms=20.0,
        gain_match_end_ms=80.0,
        welch_window_samples=8,
        welch_overlap_samples=4,
        welch_nfft=32,
    )
    times = np.repeat(np.arange(5.0, 100.0, 5.0), 2)
    angles = np.tile([90.0, 270.0], len(times) // 2)
    trace = compute_axial_pav(times, angles, sim, ana)
    valid = trace.length > 0
    assert np.allclose(trace.axis_deg[valid], 90.0, atol=1e-6)


def test_axial_sd_handles_boundary_cluster():
    from analysis.pav import circular_axis_mean_deg, circular_axis_sd_deg

    x = np.array([-89.0, 89.0, -88.0, 88.0])
    # These endpoints form one compact axial cluster around the +/-90 boundary,
    # not a ~178-degree-wide linear distribution.
    assert circular_axis_sd_deg(x) < 3.0
    assert abs(abs(circular_axis_mean_deg(x)) - 90.0) < 1e-9
