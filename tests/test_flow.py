import numpy as np

from analysis.flow import UnwrappedPAVTrace, linear_velocity_deg_s, summarize_flow, unwrap_axial_trace
from analysis.pav import PAVTrace


def test_unwrap_axial_trace_crosses_boundary_continuously():
    times = np.arange(1000.0, 8005.0, 5.0)
    true_disp = np.linspace(70.0, 130.0, times.size)
    wrapped_axis = np.mod(90.0 + true_disp, 180.0)
    pav = PAVTrace(times_ms=times, axis_deg=wrapped_axis, length=np.ones(times.size))
    out = unwrap_axial_trace(pav, target_axis_deg=90.0, start_ms=1000.0)
    assert np.allclose(out.displacement_deg, true_disp, atol=1e-8)


def test_linear_velocity_and_summary():
    times = np.arange(1000.0, 8005.0, 5.0)
    disp = 4.0 * (times / 1000.0 - 1.0) + 2.0
    tr = UnwrappedPAVTrace(times_ms=times, displacement_deg=disp, length=np.ones(times.size))
    assert np.isclose(linear_velocity_deg_s(tr, 6000.0, 8000.0), 4.0, atol=1e-10)
    s = summarize_flow(tr)
    assert np.isclose(s.velocity_1_3_deg_s, 4.0, atol=1e-10)
    assert np.isclose(s.velocity_6_8_deg_s, 4.0, atol=1e-10)
    assert np.isclose(s.shift_1_7_deg, 24.0, atol=1e-10)
