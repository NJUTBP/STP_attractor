"""Manuscript Supplementary Fig. S5 (legacy internal module S12 / ED7).

Plotting-layer revision only. Phase-15 confirmation tables, statistical roles,
and every legacy ``FigS12*`` source-data export are preserved. The rendered
manuscript figure is renumbered to S5 and uses the manuscript-wide visual
language.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from runtime import context
from data_io import paired_matrix, unique
import visual_grammar as vg
from supp_captions_cpl import SUPP_CAPTIONS

STEM = 'FigS5_confirmation_set'
LEGACY_PREFIX = 'FigS12'

CAPTION = SUPP_CAPTIONS[5]

CONTRAST_LABELS = {
    'PRIMARY_carrier_u': 'Activity only − Activity + $u$',
    'PRIMARY_route_output_vs_release': 'Output-only − Release-only',
    'SUPPORT_co_shift': 'Activity only − All states',
    'SUPPORT_output_vs_static': 'Output-only − Static',
    'SUPPORT_dynamic_vs_r_shift': 'Activity only − Activity + $R$',
    'SUPPORT_dynamic_vs_output': 'Full − Output-only',
}
CONTRAST_ORDER = list(CONTRAST_LABELS)

MODEL_CONDITIONS = ['dynamic', 'u_output_only', 'u_release_only', 'static_gain']
MODEL_ENCS = [vg.MODELS[k] for k in ['full', 'output', 'release', 'static']]
MODEL_LABELS = ['Full', 'Output-\nonly', 'Release-\nonly', 'Static']

SHIFT_KINDS = ['activity_shift', 'activity_u_shift', 'activity_r_shift', 'co_shift']
SHIFT_ENCS = [vg.SHIFTS[k] for k in SHIFT_KINDS]
SHIFT_LABELS = ['Activity\nonly', 'Activity\n+ $u$', 'Activity\n+ $R$', 'All\nstates']

INK, MID, LIGHT = vg.NEUTRAL['ink'], vg.NEUTRAL['mid'], vg.NEUTRAL['light']


def compute_panels(ds):
    """Archived Phase-15 confirmation tables of the v10 panels (retention tables are no longer drawn)."""
    return {
        'seed': ds.csv(15, 'analysis/phase15_seed_metrics.csv'),
        'contrasts': ds.csv(15, 'analysis/phase15_confirmatory_contrasts.csv'),
    }


def export_source_data(ex, d):
    """Preserve all legacy S12 source-data filenames and schemas."""
    ex.csv(f'{LEGACY_PREFIX}_seed_metrics', d['seed'])
    ex.csv(f'{LEGACY_PREFIX}_contrasts', d['contrasts'])


def _paired_values(seed, delta, conditions=None, shifts=None):
    if conditions is not None:
        q = seed[np.isclose(seed.abs_delta_deg, delta) & seed.shift_kind.eq('activity_shift')]
        return paired_matrix(q, 'condition', 'R_2000ms', conditions, 16)
    q = seed[np.isclose(seed.abs_delta_deg, delta) & seed.condition.eq('dynamic')]
    return paired_matrix(q, 'shift_kind', 'R_2000ms', shifts, 16)


def _contrast_forest(ax, contrasts):
    names = CONTRAST_ORDER
    yy = np.arange(len(names))[::-1].astype(float)
    # Same data as v6: 11.25 deg and 22.5 deg are plotted on each named contrast row.
    for delta, offset in [(11.25, +.15), (22.5, -.15)]:
        q = (contrasts[np.isclose(contrasts.abs_delta_deg, delta)].set_index('contrast').reindex(names))
        if q.R2s_difference.isna().any():
            raise ValueError(f'Missing confirmation contrast at {delta}°')
        for k, name in enumerate(names):
            r = q.loc[name]
            y = yy[k] + offset
            primary = delta == 11.25 and name in {'PRIMARY_carrier_u', 'PRIMARY_route_output_vs_release'}
            if primary:
                color, marker, face, lw, ms = vg.MODELS['full']['color'], 'D', vg.MODELS['full']['color'], 1.4, 5.0
            elif delta == 11.25:
                color, marker, face, lw, ms = '#4D4D4D', 'o', '#4D4D4D', 1.0, 4.6
            else:
                color, marker, face, lw, ms = '#7A7A7A', 's', 'white', 1.0, 4.6
            m = float(r.R2s_difference)
            ax.errorbar([m], [y], xerr=[[m - float(r.ci_low)], [float(r.ci_high) - m]], fmt=marker, color=color,
                        mfc=face, mec=color, mew=.85, ms=ms, elinewidth=lw, capsize=2.2, capthick=lw * .8, zorder=3)
    vg.zero_line(ax, vertical=True)
    ax.set_yticks(yy, [CONTRAST_LABELS[n] for n in names])
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(-.6, len(names) - .4)
    lo, hi = float(contrasts.ci_low.min()), float(contrasts.ci_high.max())
    lim = 1.08 * max(abs(lo), abs(hi))
    ax.set_xlim(-lim, lim)
    vg.g_ticks(ax, 'x')
    vg.closed(ax)
    handles = [
        Line2D([], [], color=vg.MODELS['full']['color'], marker='D', ls='', ms=5.0, label='11.25° primary'),
        Line2D([], [], color='#4D4D4D', marker='o', ls='', ms=4.6, label='11.25° supporting'),
        Line2D([], [], color='#7A7A7A', marker='s', mfc='white', ls='', ms=4.6, label='22.5° supporting'),
    ]
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0., 1.02), ncol=3, handlelength=1.0,
              columnspacing=1.6, borderaxespad=0.)


def draw_figure(d):
    """Manuscript S5 (CPL revision 2): (a) closed forest plot; (b)-(e) 2 x 2 strips with tick labels on every
    strip, letters on one column grid and one row grid."""
    L = vg.Layout(146)
    fig = L.fig
    axa = L.axes(60., 106., 110., 31.)
    _contrast_forest(axa, d['contrasts'])
    axa.set_xlabel(r'Paired contrast $\Delta F_\mathrm{rest}$(2 s)')
    CX, CW, RH = (18., 104.), 64., 33.

    def strip(ax, delta, show_ylabel):
        vg.zero_line(ax)
        ax.set_ylim(-.47, .76)
        ax.set_yticks([-.4, -.2, 0, .2, .4, .6])
        vg.g_ticks(ax, 'y')
        if show_ylabel:
            ax.set_ylabel(r'$F_\mathrm{rest}$(2 s)')
        ax.text(.98, .96, r'$|\Delta\theta|$ = ' + f'{delta:g}°', transform=ax.transAxes, ha='right', va='top',
                fontsize=vg.FS['note'], color=MID)
        vg.categorical(ax)

    paired_route, paired_surgery = [], []
    for letter, x, delta, connect in [('b', CX[0], 11.25, [(1, 2)]), ('c', CX[1], 22.5, [(0, 1)])]:
        ax = L.axes(x, 56., CW, RH)
        pv = _paired_values(d['seed'], delta, conditions=MODEL_CONDITIONS)
        paired_route.append((letter, pv))
        vg.paired_dots(ax, pv.to_numpy(float), MODEL_ENCS, MODEL_LABELS, connect=connect)
        strip(ax, delta, letter == 'b')
    for letter, x, delta in [('d', CX[0], 11.25), ('e', CX[1], 22.5)]:
        ax = L.axes(x, 10., CW, RH)
        pv = _paired_values(d['seed'], delta, shifts=SHIFT_KINDS)
        paired_surgery.append((letter, pv))
        vg.paired_dots(ax, pv.to_numpy(float), SHIFT_ENCS, SHIFT_LABELS, connect=[(0, 1)])
        strip(ax, delta, letter == 'd')
    L.letter(1.5, 139.6, 'a', va='center')
    for k, x, top in (('b', 1.5, 89.), ('c', 89., 89.), ('d', 1.5, 43.), ('e', 89., 43.)):
        L.letter(x, top + 2.6, k, va='center')
    return fig, paired_route, paired_surgery


def build_figure(ds=None, ex=None):
    ds, ex = context(ds, ex)
    d = compute_panels(ds)
    export_source_data(ex, d)
    # Preserve the four additional legacy paired-matrix exports exactly.
    for letter, delta in [('B', 11.25), ('C', 22.5)]:
        pv = _paired_values(d['seed'], delta, conditions=MODEL_CONDITIONS)
        ex.csv(f'{LEGACY_PREFIX}{letter}_paired_routes', pv.reset_index())
    for letter, delta in [('D', 11.25), ('E', 22.5)]:
        pv = _paired_values(d['seed'], delta, shifts=SHIFT_KINDS)
        ex.csv(f'{LEGACY_PREFIX}{letter}_paired_surgery', pv.reset_index())

    with vg.style():
        fig, _, _ = draw_figure(d)
        return ex.save(fig, STEM, CAPTION)


if __name__ == '__main__':
    build_figure()
