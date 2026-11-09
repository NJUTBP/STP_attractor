"""Manuscript Supplementary Fig. S9: prespecified noise-robustness test (legacy module S9).

Panels a-c: mean absolute displacement of the full model and the static control at 0-90 pA,
the noise-integrated displacement B per trial, and the paired static-minus-full contrasts
(primary integrated contrast and secondary single-amplitude contrasts).

v9 (plotting layer): the exploratory static-minus-model parameter maps (v7 panels d-f) and their
``FigS9_combo_mode_summary`` export are removed; a-c keep their v7 geometry (the drawing is shifted
down with ``y_shift_mm``).  Phase-13 archived inputs, the integrated-burden definition, the archived
bootstrap summaries and the remaining legacy ``FigS9*`` source-data names are unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from runtime import context
from data_io import integrated_anchor, summary_sem
from plot_style import Export
import visual_grammar as vg
from supp_captions_cpl import SUPP_CAPTIONS

STEM = 'FigS9_noise_robustness'
LEGACY_PREFIX = 'FigS9'
MISSING = '#D9D9D9'
INK, RAW, MID, LIGHT = (vg.NEUTRAL[k] for k in ['ink', 'raw', 'mid', 'light'])

CAPTION = SUPP_CAPTIONS[9]


def compute_panels(ds):
    """Read and derive the identical Phase-13 quantities used by the v6 figure."""
    seed = ds.csv(13, 'analysis/per_seed_noise_response.csv')
    noise = ds.csv(13, 'analysis/noise_level_summary.csv')
    primary = ds.json(13, 'analysis/phase13_summary.json')['primary_confirmatory_anchor']

    anchor = seed[(seed.combo_id == 'tf_1000_tr_20') & (seed['mode'] == 'dynamic')]
    burdens = integrated_anchor(seed)

    nn = noise[(noise.combo_id == 'tf_1000_tr_20') &
               (noise['mode'] == 'dynamic') & (noise.noise_pa > 0)].sort_values('noise_pa')
    labels = ['Integrated\n(primary)'] + [f'{v:g} pA' for v in nn.noise_pa]
    means = [primary['integrated_resistance_gain_deg']] + nn.mean_abs_resistance_gain_deg.tolist()
    low = [primary['integrated_resistance_gain_ci_low']] + nn.mean_abs_resistance_gain_ci_low.tolist()
    high = [primary['integrated_resistance_gain_ci_high']] + nn.mean_abs_resistance_gain_ci_high.tolist()
    forest = pd.DataFrame({
        'comparison': labels,
        'mean_deg': means,
        'ci_low': low,
        'ci_high': high,
        'n_paired_seeds': 16,
    })

    forest = forest.iloc[:1].copy()   # v10 (SI-8): panel c shows the prespecified primary contrast only

    return {
        'seed': seed, 'noise': noise, 'primary': primary,
        'anchor': anchor, 'burdens': burdens, 'forest': forest,
    }


def export_source_data(ex, d):
    """Legacy FigS9 source-data exports (schema/order unchanged) for the panels drawn in v9."""
    for name, frame in [
        ('seed_noise_response', d['seed']),
        ('noise_level_summary', d['noise']),
        ('anchor_integrated_seed', d['burdens']),
    ]:
        ex.csv(f'{LEGACY_PREFIX}_{name}', frame)
    ex.csv(f'{LEGACY_PREFIX}C_anchor_forest', d['forest'])


def _model_handle(key):
    return vg.model_handle(key)


def _panel_a(ax, anchor):
    tops = []
    for col, key in [('temporal_noise_abs_delta_deg', 'full'), ('static_noise_abs_delta_deg', 'static')]:
        enc = vg.MODELS[key]
        s = summary_sem(anchor, 'noise_pa', col)
        ax.errorbar(s.noise_pa, s['mean'], yerr=s['sem'], color=enc['color'], ls=enc['ls'], marker=enc['marker'],
                    mfc=vg.marker_face(enc), mec=enc['color'], mew=.8, ms=vg.MS['line'] + .4, capsize=2.2,
                    lw=1.2, elinewidth=.9, capthick=.9, zorder=3)
        tops.append(float((s['mean'] + s['sem']).max()))
    ax.set(xlim=(-5, 95), xticks=[0, 30, 60, 90], ylim=(-.04 * max(tops), 1.1 * max(tops)),
           xlabel=r'Noise amplitude $\sigma_\eta$ (pA)', ylabel='Mean absolute\n' + r'displacement $|\Delta\varphi|$ (°)')
    ax.legend(handles=[_model_handle('full'), _model_handle('static')], loc='upper left', handlelength=2.4)
    vg.g_ticks(ax, 'xy')


def _panel_b(ax, burdens):
    pv = burdens.set_index('seed')[['Full', 'Static']]
    if pv.isna().any().any():
        raise ValueError('Confirmatory burden pairing cannot omit seeds')
    vg.paired_dots(ax, pv.to_numpy(float), [vg.MODELS['full'], vg.MODELS['static']], ['Full', 'Static'],
                   connect=[(0, 1)], jitter=.07)
    ax.set(xlim=(-.45, 1.45), ylim=(0, 1.1 * float(pv.to_numpy().max())),
           ylabel='Noise-integrated\n' + r'displacement $B$ (°)')
    vg.g_ticks(ax, 'y')


def _panel_c(ax, forest):
    """Primary paired contrast as a vertical estimate + 95% CI (estimation-plot style)."""
    r = forest.iloc[0]
    m, lo, hi = float(r.mean_deg), float(r.ci_low), float(r.ci_high)
    color = vg.MODELS['full']['color']
    ax.errorbar([0], [m], yerr=[[m - lo], [hi - m]], fmt='D', color=color, mfc=color, mec=color, ms=5.6,
                elinewidth=1.5, capsize=3.0, capthick=1.3, zorder=3)
    vg.zero_line(ax)
    span = hi - lo
    ax.set(xlim=(-.7, .7), ylim=(lo - .18 * span, hi + .18 * span), ylabel=r'$B_\mathrm{static}-B_\mathrm{full}$ (°)')
    ax.set_xticks([0], ['Primary\ncontrast'])
    ax.tick_params(axis='x', length=0)
    vg.g_ticks(ax, 'y')


def draw_figure(d):
    """S8 (CPL revision 2): one row -- (a) displacement vs noise, (b) paired B, (c) the primary contrast as a
    narrow estimation panel with the same height as (a) and (b)."""
    L = vg.Layout(58)
    fig = L.fig
    y, h = 12., 36.
    axa = L.axes(19., y, 56., h); _panel_a(axa, d['anchor'])
    axb = L.axes(95., y, 38., h); _panel_b(axb, d['burdens']); vg.categorical(axb)
    axc = L.axes(153., y, 17., h); _panel_c(axc, d['forest']); vg.categorical(axc)
    for k, x in (('a', 1.5), ('b', 80.), ('c', 138.)):
        L.letter(x, y + h + 3.0, k, va='center')
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
