import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[3]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from revision_r1.t2_efficacy_stats.metrics import u_plus_periodic  # noqa: E402
from revision_r1.t4_release_fraction.fixed_point import make_F, solve_fixed_point  # noqa: E402
from revision_r1.t4_release_fraction.run_t4 import build_cells, row_targets  # noqa: E402


def test_u_plus_increases_with_frequency():
    f = np.linspace(2, 20, 50)
    u = [u_plus_periodic(x, 0.4, 1000.0) for x in f]
    assert np.all(np.diff(u) > 0) and u[0] > 0.64 and u[-1] < 1.0


def test_fixed_point_on_synthetic_linear_F():
    rho = np.linspace(0.3, 0.95, 8)
    tab = [{"rho": r, "p_target": 0.7, "freq_hz": 16.0 - 10.0 * r} for r in rho]
    F = make_F(tab, 0.7)
    f, u = solve_fixed_point(F, 0.4, 1000.0, 0.7)
    assert np.isfinite(f) and abs(F(u, 0.7) - f) < 1e-3


def test_p_interpolation():
    rows = [{"rho": r, "p_target": 0.7, "freq_hz": 10.0} for r in (0.3, 0.6, 0.9)]
    rows += [{"rho": r, "p_target": 0.5, "freq_hz": 12.0} for r in (0.4, 0.8)]
    rows += [{"rho": r, "p_target": 0.9, "freq_hz": 8.0} for r in (0.4, 0.8)]
    F = make_F(rows, 0.7)
    assert np.isclose(F(0.6, 0.6), 11.0) and np.isclose(F(0.6, 0.8), 9.0) and np.isnan(F(0.6, 0.95))


def test_cells_from_phase10_targets():
    t = row_targets()
    assert abs(t[20.0]["main"] - 0.7042313) < 1e-6
    cells = build_cells(t)
    assert len(cells) == 52 and len({c["cell_id"] for c in cells}) == 52
