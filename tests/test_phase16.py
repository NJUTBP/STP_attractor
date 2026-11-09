from pathlib import Path

import numpy as np

from config import load_config
from model.simulator import run_segment
from phase16.reduced_model import axial_center, dax, drive_offset, msd_two_variable, restoration_curve
from phase16.snapshots import run_with_snapshots

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "smoke.toml")


def _checkpoint(condition="dynamic"):
    _res, state = run_segment(CFG, condition=condition, seed=0, noise_pa=0.0, end_ms=100.0)
    return state


def test_u_clamp_freezes_u_but_not_resources():
    st = _checkpoint()
    _r, clamped, _ = run_with_snapshots(CFG, condition="dynamic", seed=0, noise_pa=0.0, start_state=st.copy(),
                                        end_ms=160.0, clamp_u=True)
    _r, free, _ = run_with_snapshots(CFG, condition="dynamic", seed=0, noise_pa=0.0, start_state=st.copy(),
                                     end_ms=160.0, clamp_u=False)
    assert np.array_equal(clamped.u, st.u)
    assert not np.array_equal(free.u, st.u)
    assert not np.array_equal(clamped.resources, st.resources)


def test_clamp_is_restored_after_context():
    st = _checkpoint()
    run_with_snapshots(CFG, condition="dynamic", seed=0, noise_pa=0.0, start_state=st.copy(), end_ms=120.0, clamp_u=True)
    _r, after, _ = run_with_snapshots(CFG, condition="dynamic", seed=0, noise_pa=0.0, start_state=st.copy(), end_ms=160.0)
    assert not np.array_equal(after.u, st.u)


def test_snapshots_match_segmented_run():
    st = _checkpoint()
    _r, s1, snaps = run_with_snapshots(CFG, condition="dynamic", seed=0, noise_pa=0.0, start_state=st.copy(),
                                       end_ms=140.0, every_ms=10.0)
    _r, s2, _ = run_with_snapshots(CFG, condition="dynamic", seed=0, noise_pa=0.0, start_state=st.copy(), end_ms=140.0)
    assert snaps["u"].shape == (4, CFG.network.ne)
    np.testing.assert_allclose(s1.u, s2.u, atol=1e-12)
    np.testing.assert_allclose(s1.ve, s2.ve, atol=1e-9)


def test_reduced_model_formulas():
    t = np.linspace(0, 5, 501)
    F = restoration_curve(3.0, 7.0, t)
    assert abs(F[-1] - 0.3) < 1e-3
    msd = msd_two_variable(t, 1.0, 3.0, 7.0)
    assert abs(np.gradient(msd, t)[1] - 2.0) < 0.1          # free diffusion at short times
    assert abs(np.gradient(msd, t)[-1] - 2.0 * 0.49) < 0.02  # (1 - F_inf)^2 at long times


def test_drive_offset_sign():
    theta = np.arange(64) * 360.0 / 64
    s = np.exp(-dax(theta, 90.0) ** 2 / (2 * 10.0 ** 2))
    p = 0.5 + 0.3 * np.exp(-dax(theta, 80.0) ** 2 / (2 * 20.0 ** 2))   # efficacy higher toward 80 deg
    off, centre = drive_offset(s, p, theta)
    assert abs(dax(centre, 90.0)) < 1e-6 and off < 0
    assert abs(dax(axial_center(s, theta)[0], 90.0)) < 1e-6
