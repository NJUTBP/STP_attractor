"""P3 unit tests (no network simulation)."""
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[3]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from experiment.phase17_physiological import combo_id, load_phase17_config  # noqa: E402
from revision_r1.p3_u_sensitivity.rules_p3 import (anchor_check, interpretable, load_p3_section,  # noqa: E402
                                                   overall_verdict, route_mapping)

CFG = Path(__file__).resolve().parents[1] / "p3_config.toml"


def test_config_is_preregistered():
    c = load_phase17_config(CFG)
    p3 = load_p3_section(CFG)
    assert c.u_baseline == (0.2, 0.3, 0.4, 0.5, 0.6)
    assert c.tau_facilitation_ms == (1000.0,) and c.tau_recovery_ms == (20.0,)
    assert c.seeds == (0, 1, 2, 3) and c.variants == ("raw",)
    assert tuple(c.restoration["seeds"]) == tuple(range(200, 208)) and c.restoration["deltas_deg"] == [11.25]
    assert p3["restoration_u"] == [0.3, 0.5] and p3["reference_u"] == 0.4 and not p3["smoke"]
    assert combo_id(0.4, 1000.0, 20.0) == "u_0p4_tf_1000_tr_20"


def _rows(n=4, loc=0.0, shift=2.0, q=0.3, f=6.45, th=1.0):
    return [dict(localization_failure=loc, shift_1_8_deg=shift, normalized_peak_power=q, theta=th, frequency_hz=f)
            for _ in range(n)]


def test_interpretable():
    p3 = load_p3_section(CFG)
    assert interpretable(_rows(), p3)["interpretable"]
    assert interpretable(_rows(f=14.0, th=0.0), p3)["interpretable"]          # theta not required
    assert not interpretable(_rows(loc=1.0), p3)["interpretable"]
    assert not interpretable(_rows(shift=30.0), p3)["interpretable"]
    assert not interpretable(_rows(q=0.02), p3)["interpretable"]


def test_anchor_check():
    ref = json.loads((CODE_ROOT / "config" / "anchor_reference.json").read_text())
    f = {s: ref["per_seed_theta"][str(s)]["frequency_hz"] for s in range(4)}
    sh = {s: ref["per_seed_theta"][str(s)]["zero_noise_abs_shift_1_8_deg"] for s in range(4)}
    ok = anchor_check(f, sh, ref["full_p_match_theta"], ref["full_mean_spike_release_u"], ref)
    assert ok["status"] == "PASS"
    bad = anchor_check({s: v + 1.0 for s, v in f.items()}, sh, ref["full_p_match_theta"],
                       ref["full_mean_spike_release_u"], ref)
    assert bad["status"] == "FAIL"


def test_route_mapping_and_verdict():
    D = dict(release_route_max_abs_error_hz=1.0, output_route_min_abs_error_hz=2.0)
    assert route_mapping(6.45, 6.35, 11.2, D)["holds"]
    assert not route_mapping(6.45, 6.35, 7.0, D)["holds"]
    p3 = load_p3_section(CFG)
    rows = [dict(u=u, interpretable=True, calibration_valid=True, mapping_holds=True) for u in (0.2, 0.3, 0.4, 0.5, 0.6)]
    assert overall_verdict(rows, p3)["verdict"] == "U_ROBUST"
    rows[0]["mapping_holds"] = False
    assert overall_verdict(rows, p3)["verdict"] == "PARTIAL"
    rows = [dict(u=u, interpretable=(u in (0.3, 0.4)), calibration_valid=True, mapping_holds=(True if u in (0.3, 0.4) else None))
            for u in (0.2, 0.3, 0.4, 0.5, 0.6)]
    assert overall_verdict(rows, p3)["verdict"] == "HOLDS_WHERE_EVALUABLE"
    rows = [dict(u=u, interpretable=True, calibration_valid=True, mapping_holds=False) for u in (0.2, 0.4)]
    assert overall_verdict(rows, p3)["verdict"] == "FAILS"
