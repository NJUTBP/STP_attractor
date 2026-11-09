from config import load_phase13_confirmatory
from experiment.mechanism_map import combo_id
from experiment.phase13_confirmation import ANCHOR_P_MATCH, load_frozen_controls
from config import load_config


def test_anchor_count_is_128():
    confirm = load_phase13_confirmatory("config/phase13_anchor.toml")
    assert confirm.expected_run_count() == 128
    assert confirm.combos == ((1000.0, 20.0),)
    assert confirm.temporal_modes == ("dynamic",)
    assert confirm.seeds[0] == 100 and confirm.seeds[-1] == 115


def test_full_grid_count_is_4608():
    confirm = load_phase13_confirmatory("config/phase13_full_grid.toml")
    assert len(confirm.combos) == 12
    assert confirm.expected_run_count() == 4608


def test_anchor_p_match_fallback_does_not_need_phase8():
    base = load_config("config/smoke.toml")
    confirm = load_phase13_confirmatory("config/phase13_anchor.toml")
    controls, ctx = load_frozen_controls(confirm, base, None)
    cid = combo_id(1000.0, 20.0)
    assert abs(controls[cid]["dynamic"] - ANCHOR_P_MATCH[cid]["dynamic"]) < 1e-12
    assert ctx[cid]["source"].startswith("phase10")
