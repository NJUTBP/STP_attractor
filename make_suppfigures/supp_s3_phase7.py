"""Manuscript Supplementary Fig. S2 (legacy internal module S3 / ED2).

Plotting-layer revision only.  All archived input paths and legacy ``FigS3*``
source-data export names are preserved; only the rendered manuscript figure is
renumbered to S2 and adopts the manuscript-wide visual grammar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patheffects

from runtime import context
from data_io import paired_matrix
from dynamics import welch_proxy, pav
import visual_grammar as vg
from supp_captions_cpl import SUPP_CAPTIONS

STEM = 'FigS2_frozen_profile_control'
LEGACY_SOURCE_PREFIX = 'FigS3'

CAPTION = SUPP_CAPTIONS[2]

ENC = {
    'dynamic': vg.MODELS['full'],
    'frozen_profile': vg.MODELS['frozen'],
    'static_gain': vg.MODELS['static'],
}


def _profile_table(profile):
    return pd.DataFrame({
        'angle_deg': profile['theta_e_deg'],
        'efficacy': profile['efficacy_by_e'],
    })


def _example_spikes_and_pav(raw):
    t = raw['spike_times_e_ms'] / 1000
    a = raw['spike_angles_e_deg']
    m = (t >= 1) & (t <= 8)
    pt = np.arange(1000, 8001, 25.)
    pv = pav(raw, pt)
    return (
        pd.DataFrame({'time_s': t[m], 'angle_deg': a[m]}),
        pd.DataFrame({'time_ms': pt, 'axial_PAV_deg': pv}),
    )


def _example_current(raw):
    t = raw['sample_times_ms'] / 1000
    x = raw['plfp_gaba_e'] / 1000
    m = (t >= 5) & (t <= 6.2)
    return pd.DataFrame({'time_s': t[m], 'current_nA': x[m]})


def _spectra_table(ds):
    rows = []
    for seed in range(10):
        f, p, peak = welch_proxy(ds.theta('frozen_profile', seed))
        rows.append(pd.DataFrame({
            'seed': seed,
            'frequency_hz': f,
            'psd_nA2_per_Hz': p / 1e6,
            'peak_frequency_hz': peak,
        }))
    return pd.concat(rows, ignore_index=True)


def compute_panels(ds):
    """Archived inputs of the v10 panels (former a, d, f).  The stationarity table is still exported:
    it backs the legend statement that all 10 frozen-profile trials passed the localization criterion."""
    profile = ds.npz(ds.path(7, 'frozen_profile.npz'), 'archived frozen spatial efficacy')
    return {
        'profile': _profile_table(profile),
        'stat': ds.csv(7, 'analysis/frozen_stationarity.csv'),
        'theta': ds.csv(7, 'analysis/theta_per_seed.csv'),
        'spectra': _spectra_table(ds),
    }


def export_source_data(ex, d):
    """v6 file names for the panels still drawn (former B/C example-trial files are not drawn in v10)."""
    ex.csv(f'{LEGACY_SOURCE_PREFIX}_stationarity', d['stat'])
    ex.csv(f'{LEGACY_SOURCE_PREFIX}_theta', d['theta'])
    ex.csv(f'{LEGACY_SOURCE_PREFIX}_profile', d['profile'])
    ex.csv(f'{LEGACY_SOURCE_PREFIX}D_PSD_by_seed', d['spectra'])


def draw_figure(d):
    """Manuscript S2 (CPL revision 2): three panels on one row, closed axes, PSD units as in Fig. 1(d),
    theta band as a side bar outside the plotting area."""
    from matplotlib.lines import Line2D
    L = vg.Layout(59)          # CPL revision 4 (C): top blank band 8.6 -> 3.6 mm
    fig = L.fig
    y, h = 13., 38.
    ax_a = L.axes(17., y, 40., h)
    prof = d['profile']
    ax_a.plot(prof.angle_deg, prof.efficacy, color=vg.MODELS['frozen']['color'], lw=1.2)
    ax_a.set(xlim=(0, 360), xticks=[0, 90, 180, 270, 360],
             ylim=vg.pad(float(prof.efficacy.min()), float(prof.efficacy.max()), .1),
             xlabel='E-cell angle (°)', ylabel='Time-averaged efficacy')
    vg.g_ticks(ax_a, 'y')

    ax_b = L.axes(75., y, 40., h)
    sp = d['spectra']
    for _, q in sp.groupby('seed', sort=True):
        ax_b.plot(q.frequency_hz, q.psd_nA2_per_Hz * 1e3, color=vg.NEUTRAL['light'], lw=.6, alpha=.85)
    mean = sp.groupby('frequency_hz', sort=True, as_index=False).psd_nA2_per_Hz.mean()
    e = vg.MODELS['frozen']
    ax_b.plot(mean.frequency_hz, mean.psd_nA2_per_Hz * 1e3, color=e['color'], ls=e['ls'], lw=1.2)
    pmax = float((sp.psd_nA2_per_Hz[sp.frequency_hz <= vg.PSD_FMAX_HZ] * 1e3).max())
    ax_b.set(xlim=(0, vg.PSD_FMAX_HZ), xticks=np.arange(0, vg.PSD_FMAX_HZ + 1, 10), ylim=(-.03 * pmax, 1.1 * pmax),
             xlabel='Frequency (Hz)', ylabel=r'PSD (10$^{-3}$ nA$^2$ Hz$^{-1}$)')
    ax_b.legend(handles=[Line2D([], [], color=vg.NEUTRAL['light'], lw=1.0, label='Single trials'),
                         Line2D([], [], color=e['color'], ls=e['ls'], lw=1.2, label='Mean')], loc='upper right')
    vg.interval_bar(ax_b, 4, 12, '4–12 Hz', color=vg.NEUTRAL['band'])
    vg.g_ticks(ax_b, 'y')

    ax_c = L.axes(133., y, 33., h)
    conds = ['dynamic', 'frozen_profile', 'static_gain']
    p = paired_matrix(d['theta'], 'condition', 'dominant_frequency_hz', conds, 10)
    vg.paired_dots(ax_c, p.to_numpy(float), [ENC[c] for c in conds], ['Full', 'Frozen\nprofile', 'Static'],
                   connect=(), jitter=.1, mean_width=.62)   # rev. 4 (C): mean marks visible beside the points
    ax_c.set(ylim=(0, 16.5), yticks=[0, 4, 8, 12, 16], ylabel='Dominant frequency (Hz)')
    vg.side_bar(ax_c, 4., 12., '4–12 Hz')
    vg.categorical(ax_c)

    for k, x in (('a', 1.5), ('b', 60.), ('c', 118.)):
        L.letter(x, y + h + 2.6, k, va='center')
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
