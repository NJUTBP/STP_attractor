"""P2b tests (python revision_r1/run_tests.py).  About 1-2 min on one core.

1. The calibrated continuous models run through run_segment_p1b exactly as through the frozen run_segment,
   including a static efficacy above 1 (the C3 static control needs p of about 1.8).
2. C3 drive: the accumulator of a continuous static run is p * sum_i w_i s_i per step; for temporal conditions the
   drive read from a frozen file (gain_match_numerator / window steps / N_I) equals the P2b accumulator, which
   justifies using the existing C1 runs as calibration wave 0.
3. Calibration core on synthetic drive curves; existing-file checks; verdict rules.
"""
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[3]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from config import load_config  # noqa: E402
from model.connectivity import build_connectivity  # noqa: E402
from model.simulator import run_segment  # noqa: E402
from revision_r1.common.simulator_event import run_segment_p1b  # noqa: E402
from revision_r1.p2b_drive_matched.rules_p2b import (  # noqa: E402
    calibrate_core, check_existing_run, drive_from_frozen_result, load_p2b_config, p2b_verdict, run_kwargs,
    window_steps)

T_END = 40.0
_CACHE = {}


def _cfg():
    if "cfg" not in _CACHE:
        cfg = load_config(CODE_ROOT / "config" / "phase4_causal.toml")
        cfg = replace(cfg, analysis=replace(cfg.analysis, gain_match_start_ms=10.0, gain_match_end_ms=35.0))
        _CACHE["cfg"], _CACHE["conn"] = cfg, build_connectivity(cfg.network)
    return _CACHE["cfg"], _CACHE["conn"]


def test_config_and_kwargs():
    pc = load_p2b_config()
    assert pc.seeds_calibration == (0, 1, 2, 3) and pc.seeds_frequency == tuple(range(10))
    assert abs(pc.initial["static_p0"] - 1.957016 * 0.902801) < 1e-3
    assert run_kwargs("output_only", 1.2) == dict(convention="continuous", condition="u_output_only",
                                                  temporal_gain_scale=1.2)
    assert run_kwargs("release_only", 2.2) == dict(convention="continuous", condition="u_release_only",
                                                   temporal_gain_scale=2.2)
    assert run_kwargs("static", 1.8) == dict(convention="continuous", condition="static_gain", static_gain=1.8)


def test_calibrated_models_equal_frozen_simulator():
    cfg, conn = _cfg()
    for model, param in (("output_only", 1.14), ("release_only", 2.16), ("static", 1.7668)):
        kw = run_kwargs(model, param)
        frozen_kw = {k: v for k, v in kw.items() if k != "convention"}
        ra, sa = run_segment(cfg, seed=0, noise_pa=0.0, connectivity=conn, end_ms=T_END, **frozen_kw)
        rb, sb, ex = run_segment_p1b(cfg, seed=0, noise_pa=0.0, connectivity=conn, end_ms=T_END, **kw)
        for k in ("spike_times_e_ms", "spike_angles_e_deg", "spike_times_i_ms", "plfp_gaba_e"):
            assert np.array_equal(getattr(ra, k), getattr(rb, k)), (model, k)
        for k in ("ve", "vi", "x_nmda", "s_nmda", "s_gaba"):
            assert np.array_equal(getattr(sa, k), getattr(sb, k)), (model, k)
        assert len(ra.spike_times_e_ms) > 0, model
        if model != "static":
            assert ex["drive_num"] == ra.gain_match_numerator, model


def test_static_drive_accumulator_definition():
    cfg, conn = _cfg()
    cfg = replace(cfg, analysis=replace(cfg.analysis, gain_match_start_ms=19.99, gain_match_end_ms=20.09))
    p = 1.7668
    _r, cur, _ = run_segment_p1b(cfg, condition="static_gain", static_gain=p, seed=0, noise_pa=0.0,
                                 connectivity=conn, end_ms=20.0)
    w = conn.ei_presynaptic_weight_sum
    for k in range(5):
        _r2, cur, ex = run_segment_p1b(cfg, condition="static_gain", static_gain=p, seed=0, noise_pa=0.0,
                                       connectivity=conn, start_state=cur, end_ms=round(20.0 + 0.02 * (k + 1), 6))
        ref = p * float(np.dot(w, cur.s_nmda))
        assert ex["drive_steps"] == 1 and abs(ex["drive_num"] - ref) <= 1e-12 * abs(ref)


def test_frozen_file_drive_equals_accumulator():
    """Wave 0 reads D of the existing C1 runs from gain_match_numerator; it must equal the P2b accumulator."""
    cfg, conn = _cfg()
    for cond, k in (("u_output_only", 0.8721143), ("u_release_only", 2.1329120)):
        r, _s = run_segment(cfg, condition=cond, seed=1, noise_pa=0.0, connectivity=conn, end_ms=T_END,
                            temporal_gain_scale=k)
        num, den = drive_from_frozen_result(r, cfg)
        _rb, _sb, ex = run_segment_p1b(cfg, condition=cond, seed=1, noise_pa=0.0, connectivity=conn,
                                       end_ms=T_END, temporal_gain_scale=k)
        assert ex["drive_steps"] == window_steps(cfg)
        D_file, D_acc = num / den, ex["drive_num"] / (ex["drive_steps"] * ex["ni"])
        assert D_file == D_acc, (cond, D_file, D_acc)


def test_existing_run_checks():
    ok = SimpleNamespace(condition="u_output_only", seed=2, noise_pa=0.0, temporal_gain_scale=0.87211431)
    check_existing_run(ok, "u_output_only", 2, k_expected=0.8721143)
    for bad, kw in ((dict(condition="dynamic"), {}), (dict(seed=3), {}), (dict(noise_pa=60.0), {}),
                    (dict(temporal_gain_scale=0.9), {})):
        r = SimpleNamespace(**{**vars(ok), **bad})
        try:
            check_existing_run(r, "u_output_only", 2, k_expected=0.8721143)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_calibration_core_synthetic():
    cal = dict(max_waves=3, tolerance=0.01, validity=0.02, max_step_factor=4.0)
    curves = {"output_only": lambda p: 50.0 * p ** 0.6, "release_only": lambda p: 26.0 * p,
              "static": lambda p: 40.0 * p ** 0.4}
    target = 57.68

    def run_wave(todo, wave):
        return [(curves[it["model"]](it["param"]), f"dir_{it['model']}_w{wave}") for it in todo]

    items = [dict(model="output_only", target=target, param=0.8721143,
                  history=[dict(wave=0, param=0.8721143, D=curves["output_only"](0.8721143),
                                rel_error=abs(curves["output_only"](0.8721143) - target) / target, dir="w0")]),
             dict(model="release_only", target=target, param=2.1329,
                  history=[dict(wave=0, param=2.1329, D=curves["release_only"](2.1329),
                                rel_error=abs(curves["release_only"](2.1329) - target) / target, dir="w0r")]),
             dict(model="static", target=target, param=1.7668, history=[])]
    items = calibrate_core(items, run_wave, cal)
    for it in items:
        assert it["converged"], it["model"]
        assert abs(curves[it["model"]](it["final_param"]) - target) / target < 0.01
        assert it["converged_dir"].startswith(f"dir_{it['model']}")
    w0 = [dict(model="release_only", target=target, param=2.2,
               history=[dict(wave=0, param=2.2, D=57.5, rel_error=abs(57.5 - target) / target, dir="existing")])]
    w0 = calibrate_core(w0, run_wave, cal)                      # within tolerance at wave 0: no new wave
    assert w0[0]["converged"] and w0[0]["converged_dir"] == "existing" and len(w0[0]["history"]) == 1
    flat = [dict(model="static", target=target, param=1.0, history=[])]
    flat = calibrate_core(flat, lambda todo, wave: [(10.0, "x") for _ in todo], cal)
    assert not flat[0]["converged"] and flat[0]["converged_dir"] is None and len(flat[0]["history"]) == 3


def test_verdict_rules():
    dec = dict(release_max_abs_error_hz=1.0, output_min_abs_error_hz=2.0, static_min_abs_difference_hz=2.0,
               min_theta_fraction=0.75)
    ok = dict(output_only=True, release_only=True, static=True)
    f = lambda full, rel, out, st: dict(full=full, release_only=rel, output_only=out, static=st)  # noqa: E731
    v = p2b_verdict(f(6.45, 6.4, 10.0, 14.0), dict(static=0.0), ok, dec)
    assert v["verdict"] == "ROBUST_TO_DRIVE_MATCHING" and v["static_label"] == "STATIC_DEPARTS_OUTSIDE_THETA"
    v = p2b_verdict(f(6.45, 6.4, 7.5, 8.2), dict(static=1.0), ok, dec)
    assert v["verdict"] == "DEPENDS_ON_MATCHING_CRITERION" and v["reasons"][0].startswith("C_output")
    assert v["static_label"] == "STATIC_WITHIN_2HZ"
    v = p2b_verdict(f(6.45, 7.6, 10.0, 9.0), dict(static=1.0), ok, dec)
    assert v["verdict"] == "DEPENDS_ON_MATCHING_CRITERION" and v["reasons"][0].startswith("C_release")
    assert v["static_label"] == "STATIC_DEPARTS_IN_THETA"
    v = p2b_verdict(f(6.45, 6.4, 10.0, 9.0), dict(static=1.0), dict(ok, output_only=False), dec)
    assert v["verdict"] == "NOT_EVALUABLE"
    v = p2b_verdict(f(6.45, 6.4, 10.0, 9.0), dict(static=1.0), dict(ok, static=False), dec)
    assert v["verdict"] == "ROBUST_TO_DRIVE_MATCHING" and v["static_label"] == "STATIC_NOT_EVALUABLE"
