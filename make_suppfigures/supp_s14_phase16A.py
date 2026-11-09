"""Manuscript Supplementary Fig. S6 (legacy internal module S14 / former ED11).

Phase-16 module A, the facilitation-clamp causal test and restoring field.
Plotting-layer revision only: the Phase-16 archived tables and every legacy
``FigS14*`` source-data export are preserved byte-for-byte.  The rendered
manuscript figure is renumbered to S6 and uses the manuscript-wide visual
language.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from runtime import context
import phase16_data as P16
import visual_grammar as vg
from supp_captions_cpl import SUPP_CAPTIONS

STEM = 'FigS6_clamp_field'
LEGACY_PREFIX = 'FigS14'

CAPTION = SUPP_CAPTIONS[6]

ROUTES = [
    ('dynamic', 'full'),
    ('u_output_only', 'output'),
    ('u_release_only', 'release'),
]
INK, RAW, MID, LIGHT = (vg.NEUTRAL[k] for k in ['ink', 'raw', 'mid', 'light'])


def compute_panels(ds):
    """Read the identical eight archived Phase-16 module-A tables as v6."""
    p16 = P16.attach(ds)
    return {
        'rest_ci': p16.table('A', 'restoration_mean_ci'),
        'rest': p16.table('A', 'restoration_by_trial'),
        'vel': p16.table('A', 'velocity_by_trial'),
        'force': p16.table('A', 'force_curve_mean_ci'),
        'p1': p16.table('A', 'P1_paired_contrasts'),
        'p2t': p16.table('A', 'P2_small_delta_slopes_by_trial'),
        'p2s': p16.table('A', 'P2_small_delta_slopes_summary'),
        'route': p16.table('A', 'P2_route_contrasts'),
    }


def export_source_data(ex, d):
    """Preserve every legacy FigS14 source-data export name and schema."""
    ex.csv(f'{LEGACY_PREFIX}ABC_time_courses_all_shifts', d['rest_ci'])

    s = d['rest'][np.isclose(d['rest'].abs_delta, 11.25) & np.isclose(d['rest'].t_ms, 600.)]
    pv = s.pivot(index='seed', columns=['condition', 'mode'], values='F')
    rows = []
    for cond, _ in ROUTES:
        a, b = pv[(cond, 'normal')], pv[(cond, 'clamp')]
        rows.append(pd.DataFrame({
            'condition': cond,
            'seed': pv.index,
            'F_normal_600ms': a.to_numpy(float),
            'F_clamp_600ms': b.to_numpy(float),
        }))
    ex.csv(f'{LEGACY_PREFIX}D_paired_F600_by_trial', pd.concat(rows, ignore_index=True))
    ex.csv(f'{LEGACY_PREFIX}D_P1_paired_contrasts', d['p1'])
    ex.csv(f'{LEGACY_PREFIX}E_small_offset_gain_by_trial', d['p2t'])
    ex.csv(f'{LEGACY_PREFIX}E_small_offset_gain_summary', d['p2s'])
    ex.csv(f'{LEGACY_PREFIX}F_route_contrasts', d['route'])
    ex.csv(f'{LEGACY_PREFIX}GH_clamp_velocity_by_trial', d['vel'][d['vel']['mode'] == 'clamp'])
    ex.csv(f'{LEGACY_PREFIX}I_force_curve_mean_ci', d['force'])


def _zero(ax):
    vg.zero_line(ax)


def _model_key(ax, x, y, key, text, transform=None, gap=.05):
    vg.marker_key(ax, x, y, vg.MODELS[key], text, transform=transform or ax.transAxes, gap=gap,
                  fontsize=vg.FS['note'])


FREE_CLAMP = [('normal', '-', 'white', 'Free'), ('clamp', '--', None, r'$u$ clamped')]   # mode, ls, face, label


def _time_courses(ax, rest_ci, cond, deltas, colors, ylim, show_ylabel=False):
    for dl, c in zip(deltas, colors):
        for mode, ls, alpha, lw in [('normal', '-', .85, .95), ('clamp', '--', 1.0, 1.15)]:
            g = rest_ci[(rest_ci.condition == cond) & (rest_ci['mode'] == mode)
                        & np.isclose(rest_ci.abs_delta, dl)].sort_values('t_ms')
            ax.plot(g.t_ms, g['mean'], ls=ls, color=c, alpha=alpha, lw=lw)
    _zero(ax)
    ax.set(xlim=(135, 615), xticks=[200, 400, 600], ylim=ylim, yticks=[-.2, 0, .2, .4, .6, .8])
    if show_ylabel:
        ax.set_ylabel(r'Restoration fraction $F_\mathrm{rest}(t)$')
    vg.g_ticks(ax, 'y')


def _endpoint_pairs(ax, rest, p1):
    s = rest[np.isclose(rest.abs_delta, 11.25) & np.isclose(rest.t_ms, 600.)]
    pv = s.pivot(index='seed', columns=['condition', 'mode'], values='F')
    xbase = np.array([0., 2.75, 5.5])
    for (cond, key), xb in zip(ROUTES, xbase):
        enc = vg.MODELS[key]
        a, b = pv[(cond, 'normal')], pv[(cond, 'clamp')]
        both = a.notna() & b.notna()
        off = np.linspace(-.08, .08, len(pv))
        segs = [np.array([[xb + o, a.loc[sd]], [xb + 1 + o, b.loc[sd]]]) for o, sd in zip(off, pv.index) if both.loc[sd]]
        ax.add_collection(LineCollection(segs, colors=LIGHT, linewidths=.6, zorder=1))
        for j, (v, mode) in enumerate([(a, 'normal'), (b, 'clamp')]):
            x = xb + j
            ok = v.notna().to_numpy()
            ax.scatter(x + off[ok], v.to_numpy(float)[ok], s=vg.MS['scatter'], marker=enc['marker'],
                       facecolors='white' if mode == 'normal' else enc['color'], edgecolors=enc['color'],
                       linewidths=.8, alpha=.92, zorder=3)
            ax.plot([x - .2, x + .2], [np.nanmean(v)] * 2, color=enc['color'], lw=2.0, solid_capstyle='butt', zorder=4)
    _zero(ax)
    ax.set_xticks(xbase + .5, ['Full', 'Output-\nonly', 'Release-\nonly'])
    ax.tick_params(axis='x', length=0)
    ax.set(xlim=(-.6, 7.1), ylim=(-.35, .95), yticks=[-.2, 0, .2, .4, .6, .8], ylabel=r'$F_\mathrm{rest}(600\,\mathrm{ms})$')
    vg.g_ticks(ax, 'y')


def _gains(ax, p2t, p2s):
    allv = []
    for k, (cond, key) in enumerate(ROUTES):
        enc = vg.MODELS[key]
        v = p2t[p2t.condition == cond].small_delta_slope_per_s.to_numpy(float)
        allv.append(v)
        off = np.linspace(-.10, .10, len(v))
        ax.scatter(k + off, v, s=vg.MS['scatter'], marker=enc['marker'], facecolors=vg.marker_face(enc),
                   edgecolors='white', linewidths=.4, alpha=.92, zorder=3)
        r = p2s[p2s.condition == cond].iloc[0]
        m = float(r.mean_slope_per_s)
        ax.errorbar([k + .27], [m], yerr=[[m - float(r.ci_low)], [float(r.ci_high) - m]], fmt='none',
                    color=enc['color'], capsize=2.2, elinewidth=1.0, capthick=1.0, zorder=4)
        ax.plot([k - .17, k + .17], [m, m], color=enc['color'], lw=2.0, solid_capstyle='butt', zorder=5)
    _zero(ax)
    ax.set_xticks(range(3), ['Full', 'Output-\nonly', 'Release-\nonly'])
    ax.tick_params(axis='x', length=0)
    ax.set(xlim=(-.55, 2.6), ylim=vg.pad(min(map(np.min, allv)), max(map(np.max, allv)), .1, hi_frac=.2),
           ylabel=r'Small-offset gain (s$^{-1}$)')
    ax.text(.97, .96, r'$|\Delta\theta|\leq$ 16.875°', transform=ax.transAxes, ha='right', va='top',
            fontsize=vg.FS['note'], color=MID)
    vg.g_ticks(ax, 'y')


def _route_contrast(ax, route):
    r = route.sort_values('abs_delta')
    m = r['mean'].to_numpy(float)
    lo, hi = r.ci_low.to_numpy(float), r.ci_high.to_numpy(float)
    ax.errorbar(r.abs_delta, m, yerr=[m - lo, hi - m], color=INK, marker='D', mfc='white', mec=INK, mew=.85,
                ms=4.6, lw=1.1, capsize=2.0, elinewidth=.8, capthick=.8)
    _zero(ax)
    ax.set(xlim=(0, 48), xticks=[0, 15, 30, 45], xlabel=r'Shift magnitude $|\Delta\theta|$ (°)',
           ylabel='Full − output-only\n' + r'velocity (° s$^{-1}$)')
    vg.g_ticks(ax, 'xy')


def _individual_field(ax, clamp, force, cond, key, ylim, show_ylabel=False):
    from matplotlib.ticker import MultipleLocator
    enc = vg.MODELS[key]
    for _, g in clamp[clamp.condition == cond].groupby('seed'):
        g = g.sort_values('abs_delta')
        ax.plot(g.abs_delta, g.v_toward_old_deg_s, color='#D0D0D0', lw=.55, zorder=1)
    m = force[(force.condition == cond) & (force['mode'] == 'clamp')].sort_values('abs_delta')
    ax.plot(m.abs_delta, m['mean'], color=enc['color'], ls=enc['ls'], lw=1.6, marker=enc['marker'],
            mfc=vg.marker_face(enc), mec=enc['color'], mew=.8, ms=vg.MS['line'] + .3,
            path_effects=vg.HALO, zorder=3)
    _zero(ax)
    ax.set(xlim=(0, 48), xticks=[0, 15, 30, 45], ylim=ylim, xlabel=r'Shift magnitude $|\Delta\theta|$ (°)')
    ax.yaxis.set_major_locator(MultipleLocator(50))
    if show_ylabel:
        ax.set_ylabel(r'Restoring velocity (° s$^{-1}$)')
    vg.g_ticks(ax, 'xy')


def _release_detail(ax, force):
    enc = vg.MODELS['release']
    handles = []
    for mode, ls, face, lab in FREE_CLAMP:
        g = force[(force.condition == 'u_release_only') & (force['mode'] == mode)].sort_values('abs_delta')
        m, lo, hi = g['mean'].to_numpy(float), g.ci_low.to_numpy(float), g.ci_high.to_numpy(float)
        fc = face or enc['color']
        ax.errorbar(g.abs_delta, m, yerr=[m - lo, hi - m], color=enc['color'], ls=ls, marker=enc['marker'],
                    mfc=fc, mec=enc['color'], mew=.8, ms=vg.MS['line'], lw=1.1, capsize=2.0, elinewidth=.8, capthick=.8)
        handles.append(Line2D([], [], color=enc['color'], ls=ls, marker=enc['marker'], mfc=fc, mec=enc['color'],
                              mew=.8, ms=vg.MS['key'], label=lab))
    _zero(ax)
    ax.set(xlim=(0, 48), xticks=[0, 15, 30, 45], xlabel=r'Shift magnitude $|\Delta\theta|$ (°)',
           ylabel=r'Restoring velocity (° s$^{-1}$)')
    vg.g_ticks(ax, 'xy')


def _header(L, fig, cmap, norm, y0):
    """Figure-level keys in one row above the panels (nothing inside the axes):
    left  - facilitation profile free / clamped: line style for (a)-(c), (i); marker fill for (d), (i);
    right - horizontal colour scale of the shift magnitude |Delta theta| used in (a)-(c)."""
    handles = [Line2D([], [], color=RAW, ls='-', lw=1.15, marker='o', ms=vg.MS['key'], mfc='white', mec=RAW,
                      mew=.8, label='Free'),
               Line2D([], [], color=RAW, ls='--', lw=1.15, marker='o', ms=vg.MS['key'], mfc=RAW, mec=RAW,
                      mew=.8, label=r'$u$ clamped')]
    vg.fig_legend(L, handles, 18.0, y0, loc='center left', ncol=2, handlelength=3.0, columnspacing=2.2)
    BAR_X, BAR_W, BAR_H = 112.0, 58.0, 2.2                     # bar ends at the right edge of column 3 (170 mm)
    cax = L.axes(BAR_X, y0 - BAR_H / 2, BAR_W, BAR_H)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
    cb.set_ticks([2.8125, 11.25, 22.5, 33.75, 45])
    cb.set_ticklabels(['2.8125', '11.25', '22.5', '33.75', '45'])
    cax.tick_params(length=1.8, width=vg.LW['axis'], pad=1.2, labelsize=vg.FS['tick'])
    cb.outline.set_linewidth(vg.LW['axis'])
    L.text(BAR_X - 7.0, y0, r'Shift magnitude $|\Delta\theta|$ (°)', ha='right', va='center', fontsize=vg.FS['legend'])
    return cb


def draw_figure(d):
    """Manuscript S6 (CPL revision 5): 3 x 3 panels on one column grid with open L-shaped axes; one key row
    at the top (free/clamped legend on the left, |Delta theta| colour bar on the right); no legends inside panels;
    letters and model names share one label row per panel row."""
    L = vg.Layout(177)
    fig = L.fig
    X, CW, RH = (18., 75., 132.), 38., 39.
    R3, R2, R1 = 12., 67., 121.
    tops = {1: R1 + RH, 2: R2 + RH, 3: R3 + RH}
    deltas = np.array(sorted(d['rest_ci'].abs_delta.unique()), dtype=float)
    base = plt.get_cmap('Blues')
    cmap = LinearSegmentedColormap.from_list('Blues_s5', [base(.28), base(.95)])
    norm = Normalize(vmin=float(deltas.min()), vmax=float(deltas.max()))
    colors = [cmap(norm(x)) for x in deltas]
    rc = d['rest_ci']
    ylim = vg.pad(float(rc['mean'].min()), float(rc['mean'].max()), .08)
    for letter, x, (cond, key) in zip('abc', X, ROUTES):
        ax = L.axes(x, R1, CW, RH)
        _time_courses(ax, rc, cond, deltas, colors, ylim, show_ylabel=(letter == 'a'))
        if letter == 'b':
            ax.set_xlabel('Time after shift (ms)')
        L.text(x + 1.0, tops[1] + 3.2, vg.MODELS[key]['label'], ha='left', va='center', color=vg.NEUTRAL['ink'])
    _header(L, fig, cmap, norm, tops[1] + 12.3)

    axd = L.axes(X[0], R2, CW, RH); _endpoint_pairs(axd, d['rest'], d['p1']); vg.categorical(axd)
    axe = L.axes(X[1], R2, CW, RH); _gains(axe, d['p2t'], d['p2s']); vg.categorical(axe)
    axf = L.axes(X[2], R2, CW, RH); _route_contrast(axf, d['route'])

    clamp = d['vel'][d['vel']['mode'] == 'clamp']
    sel = clamp[clamp.condition.isin(['dynamic', 'u_output_only'])].v_toward_old_deg_s
    lim = vg.pad(float(sel.min()), float(sel.max()), .06)
    _individual_field(L.axes(X[0], R3, CW, RH), clamp, d['force'], 'dynamic', 'full', lim, True)
    _individual_field(L.axes(X[1], R3, CW, RH), clamp, d['force'], 'u_output_only', 'output', lim, False)
    _release_detail(L.axes(X[2], R3, CW, RH), d['force'])
    for x, key in zip(X, ('full', 'output', 'release')):
        L.key(x + 1.8, tops[3] + 3.2, vg.MODELS[key], vg.MODELS[key]['label'])

    for row, letters in ((1, 'abc'), (2, 'def'), (3, 'ghi')):
        for x, k in zip(X, letters):
            L.letter(x - 14., tops[row] + 3.2, k, va='center')
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
