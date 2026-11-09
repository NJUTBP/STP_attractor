"""Manuscript Supplementary Fig. S1 (legacy internal module S2 / ED1).

Plotting-layer revision only.  The archived source-data exports retain their
legacy ``FigS2*`` names so downstream data interfaces are unchanged; only the
rendered manuscript figure is renumbered to S1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from runtime import context
from data_io import paired_matrix
from dynamics import welch_proxy
import visual_grammar as vg
from supp_captions_cpl import SUPP_CAPTIONS

STEM = 'FigS1_oscillation_diagnostics'
LEGACY_SOURCE_PREFIX = 'FigS2'

COND_ORDER = ['dynamic', 'static_gain', 'baseline']
ENC = {
    'dynamic': vg.MODELS['full'],
    'static_gain': vg.MODELS['static'],
    'baseline': vg.MODELS['baseline'],
}

CAPTION = SUPP_CAPTIONS[1]


def _current_table(raw, static):
    rows = []
    for c, r in [('dynamic', raw), ('static_gain', static)]:
        t = r['sample_times_ms'] / 1000
        x = r['plfp_gaba_e']
        m = (t >= 5) & (t <= 6.2)
        center = x[(t >= 1) & (t <= 8)].mean()
        rows.append(pd.DataFrame({
            'condition': c,
            'time_s': t[m],
            'raw_current_pA': x[m],
            'centering_mean_pA': center,
            'centered_current_nA': (x[m] - center) / 1000,
        }))
    return pd.concat(rows, ignore_index=True)


def _psd_table(raw, static):
    rows = []
    for c, r in [('dynamic', raw), ('static_gain', static)]:
        f, p, peak = welch_proxy(r)
        rows.append(pd.DataFrame({
            'condition': c,
            'frequency_hz': f,
            'psd_nA2_per_Hz': p / 1e6,
            'dominant_frequency_hz': peak,
        }))
    return pd.concat(rows, ignore_index=True)


def compute_panels(ds):
    """Read exactly the same archived inputs as the v6 figure."""
    theta = ds.csv(6, 'analysis/theta_summary.csv')
    diag = ds.csv(6, 'analysis_phase6/oscillation_diagnostics.csv')
    raw = ds.theta('dynamic')
    static = ds.theta('static_gain')
    return {
        'theta': theta,
        'diag': diag,
        'raw': raw,
        'static': static,
        'current': _current_table(raw, static),
        'psd': _psd_table(raw, static),
    }


def export_source_data(ex, d):
    """Preserve every v6 source-data export, including now-unplotted rasters."""
    ex.csv(f'{LEGACY_SOURCE_PREFIX}_theta', d['theta'])
    ex.csv(f'{LEGACY_SOURCE_PREFIX}_oscillation_diagnostics', d['diag'])

    for letter, pop in [('A', 'e'), ('B', 'i')]:
        r = d['raw']
        st = r[f'spike_times_{pop}_ms'] / 1000
        ang = r[f'spike_angles_{pop}_deg']
        m = (st >= 5) & (st <= 6.2)
        ex.csv(f'{LEGACY_SOURCE_PREFIX}{letter}_spikes',
               pd.DataFrame({'time_s': st[m], 'angle_deg': ang[m]}))
    ex.csv(f'{LEGACY_SOURCE_PREFIX}C_current', d['current'])
    # Match the v6 export schema exactly: do not add the helper peak column.
    ex.csv(f'{LEGACY_SOURCE_PREFIX}D_PSD', d['psd'][['condition', 'frequency_hz', 'psd_nA2_per_Hz']])


def _summary_points(ax, df, value, n):
    p = paired_matrix(df, 'condition', value, COND_ORDER, n)
    vg.paired_dots(ax, p.to_numpy(float), [ENC[c] for c in COND_ORDER],
                   [ENC[c]['label'] for c in COND_ORDER], connect=(), jitter=.09)
    return p


def draw_figure(d):
    """Manuscript S1 (CPL revision 2): 2 x 2 grid of equal panels, one shared condition key, closed axes,
    PSD units as in Fig. 1(d) and 0-50 Hz (the dominant-frequency search range)."""
    L = vg.Layout(97)
    fig = L.fig
    MID = vg.NEUTRAL['mid']
    LX, RX, W = 19., 103., 64.
    TOP_Y, TOP_H = 57., 29.
    BOT_Y, BOT_H = 11., 32.
    # CPL revision 4 (C): the baseline control is drawn only as markers (c, d), so its key has no line.
    vg.fig_legend(L, [vg.model_handle(ENC[c], line=(c != 'baseline')) for c in COND_ORDER], 87., 93.7, loc='center',
                  columnspacing=2.4, handlelength=2.6)

    ax_a = L.axes(LX, TOP_Y, W, TOP_H)
    vals = []
    for c in ['static_gain', 'dynamic']:
        q = d['current'][d['current'].condition == c]
        e = ENC[c]
        ax_a.plot(q.time_s, q.centered_current_nA, color=e['color'], ls=e['ls'],
                  lw=1.15 if c == 'dynamic' else 1.05, zorder=3 if c == 'dynamic' else 2)
        vals.append(q.centered_current_nA.to_numpy(float))
    v = np.concatenate(vals)
    ax_a.set(xlim=(5, 6.2), ylim=vg.pad(v.min(), v.max(), .08), xticks=np.round(np.arange(5, 6.21, .2), 1),
             xlabel='Time (s)', ylabel=r'$I_\mathrm{inh}-\langle I_\mathrm{inh}\rangle$ (nA)')
    vg.g_ticks(ax_a, 'y')

    ax_b = L.axes(RX, TOP_Y, W, TOP_H)
    pmax = 0.
    for c in ['static_gain', 'dynamic']:
        q = d['psd'][d['psd'].condition == c]
        e = ENC[c]
        y = q.psd_nA2_per_Hz.to_numpy(float) * 1e3                  # nA^2/Hz -> 10^-3 nA^2/Hz (unit only)
        ax_b.plot(q.frequency_hz, y, color=e['color'], ls=e['ls'], lw=1.15 if c == 'dynamic' else 1.05,
                  zorder=3 if c == 'dynamic' else 2)
        pmax = max(pmax, float(np.nanmax(y[q.frequency_hz.to_numpy() <= vg.PSD_FMAX_HZ])))
    ax_b.set(xlim=(0, vg.PSD_FMAX_HZ), xticks=np.arange(0, vg.PSD_FMAX_HZ + 1, 10), ylim=(-.03 * pmax, 1.1 * pmax),
             xlabel='Frequency (Hz)', ylabel=r'PSD (10$^{-3}$ nA$^2$ Hz$^{-1}$)')
    vg.interval_bar(ax_b, 4, 12, '4–12 Hz', color=vg.NEUTRAL['band'])
    vg.g_ticks(ax_b, 'y')

    ax_c = L.axes(LX, BOT_Y, W, BOT_H)
    _summary_points(ax_c, d['theta'], 'dominant_frequency_hz', 10)
    ax_c.set(ylim=(0, 35), yticks=[0, 10, 20, 30], ylabel='Dominant frequency (Hz)')
    vg.side_bar(ax_c, 4., 12., '4–12 Hz')
    vg.categorical(ax_c)

    ax_d = L.axes(RX, BOT_Y, W, BOT_H)
    for c in COND_ORDER:
        q = d['diag'][d['diag'].condition == c]
        e = ENC[c]
        open_ = e.get('mfc') == 'none'
        ax_d.scatter(q.dominant_frequency_hz, q.acf_best_frequency_hz, s=vg.MS['scatter'] + 6, marker=e['marker'],
                     facecolors=vg.marker_face(e), edgecolors=e['color'] if open_ else 'white',
                     linewidths=.8 if open_ else .4, alpha=.92, zorder=3)
    ax_d.plot([0, 36], [0, 36], color=MID, lw=vg.LW['ref'], ls=':', zorder=0)
    ax_d.plot([0, 36], [0, 18], color=MID, lw=vg.LW['ref'], ls='--', zorder=0)
    ax_d.set(xlim=(0, 36), ylim=(0, 36), xticks=[0, 10, 20, 30], yticks=[0, 10, 20, 30],
             xlabel='PSD dominant frequency (Hz)', ylabel='ACF frequency (Hz)')
    ax_d.text(22., 22., '1:1', rotation=vg.line_angle(ax_d, 1.), rotation_mode='anchor', ha='center', va='bottom',
              fontsize=vg.FS['note'], color=MID, transform_rotates_text=False)
    ax_d.text(24., 12., '1:2', rotation=vg.line_angle(ax_d, .5), rotation_mode='anchor', ha='center', va='bottom',
              fontsize=vg.FS['note'], color=MID)

    for k, (x, y) in {'a': (1.5, TOP_Y + TOP_H + 2.4), 'b': (88.5, TOP_Y + TOP_H + 2.4),
                      'c': (1.5, BOT_Y + BOT_H + 2.4), 'd': (88.5, BOT_Y + BOT_H + 2.4)}.items():
        L.letter(x, y, k, va='center')
    return fig


def build_figure(ds=None, ex=None):
    ds, ex = context(ds, ex)
    d = compute_panels(ds)
    export_source_data(ex, d)
    with vg.style():
        fig = draw_figure(d)
        return ex.save(fig, STEM, CAPTION)


if __name__ == '__main__':
    build_figure()
