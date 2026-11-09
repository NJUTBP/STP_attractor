"""Manuscript Supplementary Fig. S7 (legacy internal module S15 / former ED12).

Phase-16 module B, the profile-snapshot set.  Plotting-layer revision only:
the archived Phase-16 tables and every legacy ``FigS15*`` source-data export
are preserved.  The rendered manuscript figure is renumbered to S7 and uses
the manuscript-wide visual grammar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from runtime import context
from dynamics import seed_mean_band
import phase16_data as P16
import visual_grammar as vg
from supp_captions_cpl import SUPP_CAPTIONS

STEM = 'FigS7_profile_snapshot'
LEGACY_PREFIX = 'FigS15'
BOUND_LO, BOUND_HI = 1e-6, 200.
INK, RAW, MID, LIGHT = (vg.NEUTRAL[k] for k in ['ink', 'raw', 'mid', 'light'])

NAME = {'dynamic': 'Full', 'u_output_only': 'Output-only', 'u_release_only': 'Release-only'}
SHORT = {'dynamic': 'Full', 'u_output_only': 'Output', 'u_release_only': 'Release'}
MODEL_KEY = {'dynamic': 'full', 'u_output_only': 'output', 'u_release_only': 'release'}
KIN_STYLE = [
    ('x', 'activity', r'Activity center $\varphi_s$'),
    ('psi_eff', 'drive', 'Effective synaptic reference ' + vg.psi(r'\mathrm{eff}')),
    ('psi_u', 'u', 'Facilitation trace ' + vg.psi('u')),
    ('psi_D', 'resource', 'Depletion trace ' + vg.psi('D')),
]

CAPTION = SUPP_CAPTIONS[7]


def compute_panels(ds):
    p16 = P16.attach(ds)
    return {'kin': p16.table('B', 'kinematics')}


def _kinematic_bands(kin):
    bands = []
    for cond, delta in [('dynamic', 22.5), ('u_output_only', 11.25), ('u_release_only', 11.25)]:
        kl = P16.kinematics_long(kin, cond, 'activity_shift', delta).copy()
        kl['delta_eff'] = -kl['drive_offset']
        for col in [c for c, _, _ in KIN_STYLE] + ['delta_eff']:
            t, m, lo, hi, n = seed_mean_band(kl, col, group='t_ms')
            bands.append(pd.DataFrame({
                'condition': cond, 'abs_delta': delta, 'trace': col, 't_ms': t,
                'mean_deg': m, 'ci_low': lo, 'ci_high': hi, 'n_trials': n,
            }))
    return pd.concat(bands, ignore_index=True)


def export_source_data(ex, d):
    """v10 draws a-c only; their bands keep the legacy file name."""
    ex.csv(f'{LEGACY_PREFIX}ABC_kinematics_bands', _kinematic_bands(d['kin']))


def _trace_handles():
    out = []
    for col, key, label in KIN_STYLE:
        enc = vg.TRACES[key]
        kw = dict(color=enc['color'], ls=enc['ls'], lw=enc.get('lw', 1.3), alpha=enc.get('alpha', 1.0), label=label)
        if col == 'x':
            kw['solid_capstyle'] = 'butt'
        out.append(Line2D([], [], **kw))
    return out


def _kinematics(ax, bands, cond, delta, ylabel=False, ax_sub=None):
    tops = []
    for col, trace_key, _ in KIN_STYLE:
        g = bands[(bands.condition == cond) & np.isclose(bands.abs_delta, delta) & (bands.trace == col)].sort_values('t_ms')
        enc = vg.TRACES[trace_key]
        t = g.t_ms.to_numpy(float) / 1000
        if col in ('x', 'psi_eff'):
            ax.fill_between(t, g.ci_low, g.ci_high, color=enc['color'], alpha=.12, lw=0, zorder=1)
            tops.append(float(g.ci_high.max()))
        tops.append(float(g.mean_deg.max()))
        kw = dict(color=enc['color'], ls=enc['ls'], lw=enc.get('lw', 1.3), alpha=enc.get('alpha', 1.0))
        if col == 'x':
            kw['solid_capstyle'] = 'butt'
        z = 2 if col == 'psi_D' else (3 if col == 'x' else (4 if col == 'psi_u' else 5))
        ax.plot(t, g.mean_deg.to_numpy(float), zorder=z, **kw)
    ax.axhline(delta, color=MID, ls=':', lw=vg.LW['ref'], zorder=0)
    vg.right_tag(ax, delta, r'$|\Delta\theta|$')
    ax.set(xlim=(0, 2.5), xticks=[0, .5, 1, 1.5, 2, 2.5],
           ylim=(-.06 * delta, max(1.12 * delta, max(tops) + .04 * delta)),
           yticks=[0, 5, 10, 15, 20, 25] if delta > 15 else [0, 3, 6, 9, 12], xlabel='Time after shift (s)')
    if ylabel:
        ax.set_ylabel('Displacement from\nold position (°)' if ax_sub is not None else 'Displacement from old position (°)')
    vg.g_ticks(ax, 'xy')
    if ax_sub is not None:
        vg.hide_x(ax)
        g_de = bands[(bands.condition == cond) & np.isclose(bands.abs_delta, delta) & (bands.trace == 'delta_eff')].sort_values('t_ms')
        t_s = g_de.t_ms.to_numpy(float) / 1000
        dcol = vg.VARIABLES['drive']
        ax_sub.axhspan(-.5, .5, color='#E7E7E7', lw=0, zorder=0)
        vg.zero_line(ax_sub)
        ax_sub.fill_between(t_s, g_de.ci_low, g_de.ci_high, color=dcol, alpha=.16, lw=0, zorder=1)
        ax_sub.plot(t_s, g_de.mean_deg.to_numpy(float), color=dcol, ls='-', lw=1.2, zorder=3)
        ax_sub.set(xlim=(0, 2.5), xticks=[0, .5, 1, 1.5, 2, 2.5],
                   ylim=(-.7, max(2.8, 1.1 * float(g_de.ci_high.max()))),
                   yticks=[0, 2, 4] if delta > 15 else [0, 2], xlabel='Time after shift (s)')
        if ylabel:
            ax_sub.set_ylabel(r'$\delta_\mathrm{eff}$ (°)')
        vg.g_ticks(ax_sub, 'xy')


def draw_figure(d):
    """Manuscript S7 (CPL revision 2): same column grid as Fig. S6; one trace key on top (names as in the SI);
    model and shift size in each label row; lower strip showing delta_eff."""
    bands = d['bands'] if 'bands' in d else _kinematic_bands(d['kin'])
    L = vg.Layout(77)
    fig = L.fig
    X, CW, Y, H = (18., 74., 130.), 37., 11., 47.
    vg.fig_legend(L, _trace_handles(), 87., 71.8, loc='center', ncol=4, columnspacing=1.5, handlelength=2.6)
    top = Y + H
    for letter, x, cond, delta in zip('abc', X, ['dynamic', 'u_output_only', 'u_release_only'], [22.5, 11.25, 11.25]):
        ax_sub = L.axes(x, Y, CW, 11.)
        ax = L.axes(x, Y + 13.5, CW, H - 13.5, sharex=ax_sub)
        _kinematics(ax, bands, cond, delta, ylabel=(letter == 'a'), ax_sub=ax_sub)
        L.letter(x - 14., top + 3.2, letter, va='center')
        L.text(x + 1.0, top + 3.2, f'{NAME[cond]}, ' + r'$|\Delta\theta|$' + f' = {delta:g}°', ha='left', va='center',
               color=INK)
    return fig


def build_figure(ds=None, ex=None):
    ds, ex = context(ds, ex)
    d = compute_panels(ds)
    export_source_data(ex, d)
    with vg.style():
        fig = draw_figure(d)
        saved = ex.save(fig, STEM, CAPTION)
    return saved


if __name__ == '__main__':
    build_figure()
