"""Manuscript Supplementary Fig. S3 (legacy internal module S6 / ED5).

Plotting-layer revision only. Archived Phase-10 inputs and legacy ``FigS6*``
source-data export names are preserved. The rendered manuscript figure is
renumbered to S5 and uses the manuscript-wide visual grammar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import NullLocator
from matplotlib.patches import Rectangle

from runtime import context
from plot_style import norm_for
from data_io import map_matrix
import visual_grammar as vg
from supp_captions_cpl import SUPP_CAPTIONS

STEM = 'FigS3_route_decomposition_frequency'
LEGACY_PREFIX = 'FigS6'

MODES = ['u_output_only', 'u_release_only', 'mean_u_release']
MODE_LABELS = ['Output-only', 'Release-only', 'Mean-release control']
MODE_ENCS = [vg.MODELS['output'], vg.MODELS['release'], vg.MODELS['mean_release']]

CAPTION = SUPP_CAPTIONS[3]


def gain_sensitivity(df):
    """Descriptive paired sensitivity audit; definitions unchanged from Phase 10."""
    rows = []
    for limit in [None, 1., 2., 5.]:
        keep = np.ones(len(df), bool) if limit is None else (
            (100*df.u_output_only_p_match_relative_error.abs() <= limit) &
            (100*df.u_release_only_p_match_relative_error.abs() <= limit)
        )
        q = df.loc[keep]
        rows.append({
            'limit_percent': 'all' if limit is None else str(limit),
            'n_paired_settings': len(q),
            'output_median_abs_frequency_error_hz': q.u_output_only_abs_frequency_error_hz.median(),
            'release_median_abs_frequency_error_hz': q.u_release_only_abs_frequency_error_hz.median(),
        })
    return pd.DataFrame(rows)


def _ecdf_table(df):
    rows = []
    for mode in MODES:
        vals = np.sort(df[f'{mode}_abs_frequency_error_hz'].to_numpy(float))
        cdf = np.arange(1, len(vals)+1) / len(vals)
        rows.append(pd.DataFrame({
            'mode': mode,
            'abs_error_hz': vals,
            'empirical_fraction': cdf,
        }))
    return pd.concat(rows, ignore_index=True)


FREQ_MAPS = [('full', 'Full'), ('u_output_only', 'Output-only'), ('u_release_only', 'Release-only')]


def compute_panels(ds):
    """Phase-10 route table (unchanged derivations) + the Phase-8 full-model theta fraction (former S3g)."""
    df = ds.csv(10).copy()
    for mode in MODES:
        df[f'{mode}_gain_error_percent'] = 100 * df[f'{mode}_p_match_relative_error']
    p8 = ds.csv(8)
    return {
        'summary': df,
        'gain_sensitivity': gain_sensitivity(df),
        'ecdf': _ecdf_table(df),
        'theta8': p8[['tau_facilitation_ms', 'tau_recovery_ms', 'dynamic_theta_fraction']].copy(),
    }


def export_source_data(ex, d):
    """Legacy source-data names kept (FigS6* = Phase 10, FigS4* = Phase 8; see SOURCE_DATA_MAP.md)."""
    ex.csv(f'{LEGACY_PREFIX}_phase10_combo_summary', d['summary'])
    ex.csv(f'{LEGACY_PREFIX}_gain_sensitivity_descriptive', d['gain_sensitivity'])
    ex.csv(f'{LEGACY_PREFIX}K_error_ecdf', d['ecdf'])
    ex.csv('FigS4_phase8_full_theta_fraction', d['theta8'])


def _anchor_outline(ax, mat):
    if 20. not in mat.columns or 1000. not in mat.index:
        return
    ix = list(mat.columns).index(20.)
    iy = list(mat.index).index(1000.)
    ax.add_patch(Rectangle((ix - .5, iy - .5), 1, 1, fill=False, lw=1.3, ec=vg.NEUTRAL['ink'], zorder=6))
    ax.add_patch(Rectangle((ix - .5, iy - .5), 1, 1, fill=False, lw=.65, ls=(0, (2.2, 1.5)), ec='white', zorder=7))


def _map(ax, df, value, norm, cmap, show_x, show_y, label_cells=False, signed_min=None):
    """One tau_f x tau_r map (no interpolation); closed frame, ticks at cell centres."""
    mat = map_matrix(df, value, 'tau_recovery_ms', 'tau_facilitation_ms')
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad('#E5E5E5')
    im = ax.imshow(np.ma.masked_invalid(mat.to_numpy(float)), origin='lower', aspect='auto', norm=norm,
                   cmap=cm, interpolation='none', rasterized=True)
    ax.set_xticks(range(len(mat.columns)))
    if show_x:
        ax.set_xticklabels([f'{v:g}' for v in mat.columns])
    else:
        ax.tick_params(axis='x', labelbottom=False)
    ax.set_yticks(range(len(mat.index)))
    if show_y:
        ax.set_yticklabels([f'{v:g}' for v in mat.index])
    else:
        ax.tick_params(axis='y', labelleft=False)
    vg.closed(ax)
    ax.tick_params(length=2.0, width=vg.LW['axis'], direction='out')
    _anchor_outline(ax, mat)
    if label_cells:
        # Four trials -> discrete quarters; label only heterogeneous settings.
        for iy in range(len(mat.index)):
            for ix in range(len(mat.columns)):
                val = mat.iloc[iy, ix]
                if not np.isfinite(val) or np.isclose(val, 0.) or np.isclose(val, 1.):
                    continue
                rgba = cm(norm(val))
                lum = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                ax.text(ix, iy, f'{val:g}', ha='center', va='center', fontsize=vg.FS['note'],
                        color=vg.NEUTRAL['ink'] if lum > .53 else 'white', zorder=5)
    if signed_min is not None:
        # Signed values on a diverging scale: print the rounded value in cells with |value| >= signed_min,
        # so the sign stays readable in grey-scale print (both ends of RdBu turn the same grey).
        for iy in range(len(mat.index)):
            for ix in range(len(mat.columns)):
                val = mat.iloc[iy, ix]
                if not np.isfinite(val) or abs(val) < signed_min:
                    continue
                rgba = cm(norm(val))
                lum = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                ax.text(ix, iy, f'{val:+.0f}'.replace('-', '\u2212'), ha='center', va='center',
                        fontsize=vg.FS['note'], color=vg.NEUTRAL['ink'] if lum > .45 else 'white', zorder=5)
    return im


def _colorbar(fig, im, cax, label, ticks=None):
    cb = fig.colorbar(im, cax=cax)
    if ticks is not None:
        cb.set_ticks(ticks)
    cb.set_label(label, fontsize=vg.FS['cbar'], labelpad=3)
    cb.ax.tick_params(labelsize=vg.FS['tick'], length=2.0, width=vg.LW['axis'])
    cb.outline.set_linewidth(vg.LW['axis'])
    return cb


MODE_HEADS = ['Output-only', 'Release-only', 'Mean-release']


def draw_figure(d):
    """CPL Fig. S3 (revision 2): twelve panels lettered in reading order.

    Row 1: (a)-(c) dominant-frequency maps, (d) theta-band fraction.   Row 2: (e)-(g) absolute frequency
    errors, (h) their cumulative distributions.   Row 3: (i)-(k) calibration residuals, (l) mean release
    probability u_rel.  Each letter sits on the left edge of its plotting area; plot data and colour scales
    are those of the archived figure.
    """
    df = d['summary']
    freq_vals = np.concatenate([df[f'{m}_abs_frequency_error_hz'].to_numpy(float) for m in MODES])
    gain_vals = np.concatenate([df[f'{m}_gain_error_percent'].to_numpy(float) for m in MODES])
    L = vg.Layout(154)
    fig = L.fig
    ink = vg.NEUTRAL['ink']
    pw, ph = 28.5, 33.0
    xs = [14.0, 45.5, 77.0]
    cbx, cbw = 107.5, 2.2
    xr, right_w, cbx_r = 132.0, 26.0, 160.0
    y1, y2, y3 = 112.0, 63.0, 14.0

    def head(x, top, letter, text=None):
        if text:
            L.letter_title(x, top + 3.6, letter, text, color=ink)
        else:
            L.letter(x, top + 3.6, letter, va='center')

    lognorm = LogNorm(vmin=3.5, vmax=32.)
    im = None
    for col, ((mode, label), letter) in enumerate(zip(FREQ_MAPS, 'abc')):
        ax = L.axes(xs[col], y1, pw, ph)
        im = _map(ax, df, f'{mode}_frequency_hz', lognorm, vg.CMAPS['frequency'], False, col == 0)
        head(xs[col], y1 + ph, letter, label)
    cb = _colorbar(fig, im, L.axes(cbx, y1, cbw, ph), 'Dominant frequency (Hz)', ticks=[4, 8, 16, 32])
    cb.set_ticklabels(['4', '8', '16', '32'])
    cb.ax.yaxis.set_minor_locator(NullLocator())

    axd = L.axes(xr, y1, right_w, ph)
    imd = _map(axd, d['theta8'], 'dynamic_theta_fraction', Normalize(0., 1.), vg.CMAPS['frequency'],
               True, True, label_cells=True)
    axd.set_xlabel(r'$\tau_\mathrm{r}$ (ms)')
    head(xr, y1 + ph, 'd', 'Full')
    _colorbar(fig, imd, L.axes(cbx_r, y1, cbw, ph), 'Theta-band fraction', ticks=[0, .5, 1])

    rows = [(y2, 'abs_frequency_error_hz', '|Frequency error| (Hz)', vg.CMAPS['frequency'],
             norm_for(freq_vals, signed=False), 'efg', False),
            (y3, 'gain_error_percent', 'Calibration residual (%)', vg.CMAPS['difference'],
             norm_for(gain_vals, signed=True), 'ijk', True)]
    for y, metric, cbar_label, cmap, norm, letters, show_x in rows:
        last = None
        for col, (mode, letter, text) in enumerate(zip(MODES, letters, MODE_HEADS)):
            ax = L.axes(xs[col], y, pw, ph)
            last = _map(ax, df, f'{mode}_{metric}', norm, cmap, show_x, col == 0,
                        signed_min=2. if metric == 'gain_error_percent' else None)
            head(xs[col], y + ph, letter, text)
        _colorbar(fig, last, L.axes(cbx, y, cbw, ph), cbar_label)
    L.text(4.0, (y3 + y1 + ph) / 2, r'$\tau_\mathrm{f}$ (ms)', ha='center', va='center', rotation=90, color=ink)
    L.text((xs[0] + xs[-1] + pw) / 2, 3.6, r'$\tau_\mathrm{r}$ (ms)', ha='center', va='bottom', color=ink)

    # (h) cumulative distributions of the errors mapped in (e)-(g)
    axh = L.axes(xr, y2, cbx_r + cbw - xr, ph)
    for mode, enc in zip(MODES, MODE_ENCS):
        q = d['ecdf'][d['ecdf']['mode'].eq(mode)]
        vals = q.abs_error_hz.to_numpy(float)
        cdf = q.empirical_fraction.to_numpy(float)
        axh.step(np.r_[0, vals], np.r_[0, cdf], where='post', color=enc['color'], ls=enc['ls'], lw=1.25,
                 label=enc['label'])
    emax = float(d['ecdf'].abs_error_hz.max())
    axh.set(xlim=(-.03 * emax, 1.04 * emax), ylim=(-.04, 1.04), yticks=[0, .5, 1], xticks=[0, 10, 20],
            xlabel='|Frequency error| (Hz)', ylabel='Fraction of settings')
    vg.g_ticks(axh, 'y')
    axh.legend(loc='lower right', handlelength=2.2, borderaxespad=.3)
    head(xr, y2 + ph, 'h')

    # (l) spike-weighted mean release fraction of the full model
    axl = L.axes(xr, y3, right_w, ph)
    iml = _map(axl, df, 'full_mean_spike_release_u', Normalize(0., 1.), vg.CMAPS['frequency'], True, True)
    axl.set_xlabel(r'$\tau_\mathrm{r}$ (ms)')
    head(xr, y3 + ph, 'l', 'Full')
    _colorbar(fig, iml, L.axes(cbx_r, y3, cbw, ph), r'Mean release fraction $\bar{u}_\mathrm{rel}$', ticks=[0, .5, 1])
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
