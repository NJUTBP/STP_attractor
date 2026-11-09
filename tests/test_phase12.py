from __future__ import annotations

import numpy as np

from analysis.phase11_pilot import recovery_fraction
from experiment.phase12_attractor import iter_module_a_jobs, iter_module_b_jobs
from model.simulator import paired_post_checkpoint_rng
from model.state import aligned_roll_bins


def test_module_counts():
    a = iter_module_a_jobs()
    b = iter_module_b_jobs()
    assert len(a) == 800
    assert len(b) == 120
    assert sum(j.noise_pa == 0.0 for j in a) == 32
    assert sum(j.shift_kind == "no_shift" for j in b) == 24


def test_small_delta_is_lattice_aligned():
    assert aligned_roll_bins(11.25, 1024, 256) == (32, 8)


def test_paired_noise_streams_match_across_calls():
    a_e, a_i = paired_post_checkpoint_rng(3, 5)
    b_e, b_i = paired_post_checkpoint_rng(3, 5)
    c_e, c_i = paired_post_checkpoint_rng(3, 6)
    x = a_e.normal(size=8)
    y = b_e.normal(size=8)
    z = c_e.normal(size=8)
    assert np.array_equal(x, y)
    assert not np.array_equal(x, z)
    assert np.array_equal(a_i.normal(size=4), b_i.normal(size=4))


def test_recovery_fraction_still_uses_abs_delta():
    assert abs(float(recovery_fraction(11.25, -11.25))) < 1e-12
    assert float(recovery_fraction(0.0, -11.25)) == 1.0
