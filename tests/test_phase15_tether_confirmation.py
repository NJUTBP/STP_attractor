from __future__ import annotations

from experiment.phase15_tether_confirmation import (
    PHASE15_SEEDS, PHASE15_CONDITIONS, PHASE15_DELTAS_DEG, PHASE15_SURGERIES,
    iter_phase15_branches,
)


def test_phase15_uses_independent_confirmation_seeds():
    assert PHASE15_SEEDS == tuple(range(200, 216))
    assert set(PHASE15_SEEDS).isdisjoint(set(range(8)))


def test_phase15_branch_matrix_is_focused_and_unique():
    branches = iter_phase15_branches()
    ids = [b.branch_id for b in branches]
    assert len(ids) == len(set(ids))
    # Per seed: 4 no-shift + 4 conditions*2 magnitudes*2 directions +
    # dynamic 3 surgeries*2 magnitudes*2 directions = 32.
    assert len(branches) == 16 * 32
    assert set(PHASE15_CONDITIONS) == {"dynamic", "static_gain", "u_output_only", "u_release_only"}
    assert PHASE15_DELTAS_DEG == (11.25, 22.5)
    assert PHASE15_SURGERIES == ("activity_u_shift", "activity_r_shift", "co_shift")


def test_phase15_primary_activity_shift_exists_for_all_conditions():
    branches = iter_phase15_branches()
    for cond in PHASE15_CONDITIONS:
        xs = [b for b in branches if b.condition == cond and b.shift_kind == "activity_shift" and abs(b.delta_deg) == 11.25]
        assert len(xs) == 16 * 2


def test_phase15_surgeries_apply_only_to_dynamic():
    branches = iter_phase15_branches()
    surgery = [b for b in branches if b.shift_kind in PHASE15_SURGERIES]
    assert surgery
    assert {b.condition for b in surgery} == {"dynamic"}
