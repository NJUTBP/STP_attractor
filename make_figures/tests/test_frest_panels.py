"""Fig. 3(d) and Fig. 4(a) (CPL revision 6): one drawing specification, data drawn exactly as archived."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PathCollection

MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

import fig3_attractor as f3          # noqa: E402
import fig4_synthesis as f4          # noqa: E402
import visual_grammar as vg          # noqa: E402

SRC = MAIN / 'source_data'


def _panels():
    with vg.style():
        out = []
        for mod, table, groups, pair in ((f3, 'Fig3D_confirmation_paired.csv', f3.GROUPS, (0, 1)),
                                         (f4, 'Fig4A_recovery_paired.csv', f4.RECOVERY_MODES, (1, 2))):
            fig = mod.draw_figure(mod.load_source_data(SRC))
            ax = [a for a in fig.axes if 'rest}(2' in a.get_ylabel()][0]
            out.append((fig, ax, pd.read_csv(SRC / table).set_index('seed')[groups], pair))
        return out


def test_same_specification_and_exact_data():
    (fig3, ax3, t3, p3), (fig4, ax4, t4, p4) = _panels()
    assert ax3.get_ylim() == ax4.get_ylim() == vg.FREST_AXIS['ylim']
    assert list(ax3.get_yticks()) == list(ax4.get_yticks()) == vg.FREST_AXIS['yticks']
    assert ax3.get_ylabel() == ax4.get_ylabel()
    w3 = ax3.get_position().width * fig3.get_figwidth() * 25.4 / t3.shape[1]
    w4 = ax4.get_position().width * fig4.get_figwidth() * 25.4 / t4.shape[1]
    assert abs(w3 - vg.FREST_GROUP_MM) < 1e-6 and abs(w4 - vg.FREST_GROUP_MM) < 1e-6
    off = vg.trial_offsets(16, vg.FREST_PANEL['jitter'])
    for ax, t, pair in ((ax3, t3, p3), (ax4, t4, p4)):
        assert len(t) == 16
        pts = [c for c in ax.collections if isinstance(c, PathCollection)][:t.shape[1]]
        for k, (c, g) in enumerate(zip(pts, t.columns)):
            assert np.array_equal(c.get_offsets()[:, 1], t[g].to_numpy())        # y = archived values, untouched
            assert np.allclose(c.get_offsets()[:, 0], k + off)                     # jitter: x only, by trial order
            assert c.get_sizes()[0] == vg.MS['scatter']
        seg = [c for c in ax.collections if isinstance(c, LineCollection)][0].get_segments()
        assert len(seg) == 16 and all(np.allclose(s[:, 1], t.iloc[i, list(pair)]) for i, s in enumerate(seg))
        means = [ln for ln in ax.lines if ln.get_linewidth() == 2.0]
        assert np.allclose([m.get_ydata()[0] for m in means], t.mean().to_numpy())


def test_full_column_of_fig4a_is_activity_only_of_fig3d():
    t3 = pd.read_csv(SRC / 'Fig3D_confirmation_paired.csv').set_index('seed')
    t4 = pd.read_csv(SRC / 'Fig4A_recovery_paired.csv').set_index('seed')
    assert list(t3.index) == list(t4.index) and np.array_equal(t3.activity_shift, t4.dynamic)
