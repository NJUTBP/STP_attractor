"""Manuscript Supplementary Fig. S8 (legacy internal module S16 / former ED13).

Phase-16 module C, the relative-coordinate noise set. Plotting-layer revision only:
the archived Module-C tables and all legacy ``FigS16*`` source-data exports are
preserved byte-for-byte. The manuscript-facing figure uses S11 numbering and the
shared v7 visual grammar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from runtime import context
import phase16_data as P16
import visual_grammar as vg
from supp_captions_cpl import SUPP_CAPTIONS
from matplotlib import patheffects

HALO = [patheffects.Stroke(linewidth=2.2, foreground='white', alpha=.9), patheffects.Normal()]

STEM = 'FigS8_relative_coordinate_noise'
LEGACY_PREFIX = 'FigS16'
WINDOWS = [(.5, 2.), (2., 4.), (4., 8.), (8., 10.)]
INK, RAW, MID, LIGHT = (vg.NEUTRAL[k] for k in ['ink', 'raw', 'mid', 'light'])
FULL = vg.MODELS['full']
STATIC = vg.MODELS['static']
U_TRACE = vg.TRACES['u']

# Noise amplitude is an ordered experimental variable, not a model identity.
# It is therefore encoded redundantly by fill/line style while keeping the model colour fixed.
NOISE = {
    30.: {'ls': '--', 'marker': 'o', 'mfc': 'none', 'label': '30 pA'},
    60.: {'ls': '-',  'marker': 'o', 'mfc': None,   'label': '60 pA'},
}

CAPTION = SUPP_CAPTIONS[8]


def pick(cb, cond, noise, parameter):
    r = cb[(cb.condition == cond) & np.isclose(cb.noise, noise) & (cb.parameter == parameter)]
    if len(r) != 1:
        raise ValueError(f'Missing archived C estimate {cond}/{noise}/{parameter}')
    return r.iloc[0]


def compute_panels(ds):
    p16 = P16.attach(ds)
    return {'curves': p16.table('C', 'curves')}


def _window_means(curves):
    rows = []
    for noise in (30., 60.):
        g = curves[(curves.condition == 'dynamic') & np.isclose(curves.noise, noise)]
        vals = [float(g.var_drive_offset[(g.t_s >= lo) & (g.t_s <= hi)].mean()) for lo, hi in WINDOWS]
        rows += [{'noise_pa': noise, 'window_s': f'{lo:g}-{hi:g}', 'mean_var_delta_eff_deg2': v}
                 for (lo, hi), v in zip(WINDOWS, vals)]
    return pd.DataFrame(rows)


def export_source_data(ex, d):
    """v10 draws a-d only."""
    ex.csv(f'{LEGACY_PREFIX}ABCD_curves', d['curves'])


def _noise_marker(ax, x, y, noise, color, transform=None, ms=4.0):
    enc = NOISE[float(noise)]
    face = 'white' if enc['mfc'] == 'none' else color
    ax.plot([x], [y], ls='none', marker=enc['marker'], ms=ms, mfc=face, mec=color,
            mew=.8, transform=transform or ax.transData, clip_on=False)


def _noise_key(color, label_prefix=''):
    out = []
    for noise in (30., 60.):
        enc = NOISE[noise]
        out.append(Line2D([], [], color=color, ls=enc['ls'], lw=1.1, marker=enc['marker'], ms=3.8,
                          mfc='white' if enc['mfc'] == 'none' else color, mec=color, mew=.7,
                          label=f'{label_prefix}{enc["label"]}'))
    return out


def _noise_title(ax, noise):
    """Kept for compatibility; CPL revision 2 writes sigma_eta in the figure-level label row."""
    return None


def _delta_eff(ax, curves, noise, ylabel=False):
    """Var(delta_eff): light = 20-ms samples, dark = 0.5-s running mean.  Logarithmic y axis so that every
    20-ms sample (up to 0.53 / 1.61 deg^2) is inside the axes without compressing the running mean."""
    g = curves[(curves.condition == 'dynamic') & np.isclose(curves.noise, noise)].sort_values('t_s')
    raw = g.var_drive_offset.to_numpy(float)
    ax.plot(g.t_s, raw, color=FULL['color'], lw=.5, alpha=.28, zorder=1)
    ax.plot(g.t_s, P16.running_mean(g.var_drive_offset, 25), color=FULL['color'], lw=1.4, zorder=3)
    allv = curves[curves.condition == 'dynamic'].var_drive_offset.to_numpy(float)
    allv = allv[np.isfinite(allv) & (allv > 0)]          # one log range for (a) and (b): every sample shown
    ax.set_yscale('log')
    ax.set(xlim=(0, 10), xticks=[0, 2, 4, 6, 8, 10], ylim=(allv.min() / 1.5, allv.max() * 1.5),
           xlabel='Elapsed time (s)')
    if ylabel:
        ax.set_ylabel(r'Var($\delta_\mathrm{eff}$) (deg$^2$)')
    vg.g_ticks(ax, 'xy')


def _absolute_msd(ax, curves, noise, ylabel=False, legend=False):
    handles, top = [], 0.
    for cond, enc in [('dynamic', FULL), ('static_gain', STATIC)]:
        g = curves[(curves.condition == cond) & np.isclose(curves.noise, noise)].sort_values('t_s')
        t = g.t_s.to_numpy(float)
        # CPL revision 4 (C): plain lines; colour and line style already separate the two models.
        kw = dict(color=enc['color'], ls=enc['ls'], lw=1.25)
        ax.plot(t, g.msd_phi, **kw)
        handles.append(Line2D([], [], label=enc['label'], **kw))
        top = max(top, float(g.msd_phi.max()))
    ax.set(xlim=(-.2, 10), xticks=[0, 2, 4, 6, 8, 10], ylim=(-.04 * top, 1.08 * top), xlabel='Elapsed time (s)')
    if ylabel:
        ax.set_ylabel(r'MSD of $\varphi_s$ (deg$^2$)')
    if legend:
        ax.legend(handles=handles, loc='lower right', handlelength=2.4, borderaxespad=.35)
    vg.g_ticks(ax, 'xy')


def _late_slopes(ax, cb):
    # Pair Full and Static estimates at each noise level visually, without implying this set was a
    # prespecified Full-vs-Static inferential comparison.
    xnoise = np.array([30., 60.])
    offsets = {'dynamic': -2.0, 'static_gain': 2.0}
    for cond, enc, label in [('dynamic', FULL, 'Full'), ('static_gain', STATIC, 'Static (context)')]:
        for noise in xnoise:
            r = pick(cb, cond, noise, 'late_msd_slope')
            x = noise + offsets[cond]
            face = vg.marker_face(enc)
            ax.errorbar([x], [r.point_estimate],
                        yerr=[[r.point_estimate-r.ci_low], [r.ci_high-r.point_estimate]],
                        color=enc['color'], marker=enc['marker'], mfc=face, mec=enc['color'], mew=.7,
                        ms=4.0, ls='none', capsize=2.0, elinewidth=.85, capthick=.75, zorder=3)
    ax.axhline(0, color=MID, ls=':', lw=vg.LW['ref'], zorder=0)
    ax.set_xticks([30, 60], ['30', '60'])
    ax.set_xlim(21, 69)
    ax.set_xlabel('Noise amplitude (pA)')
    ax.set_ylabel(r'Late MSD slope (deg$^2$ s$^{-1}$)')
    handles = [
        Line2D([], [], ls='none', marker=FULL['marker'], ms=4, mfc=vg.marker_face(FULL),
               mec=FULL['color'], color=FULL['color'], label='Full'),
        Line2D([], [], ls='none', marker=STATIC['marker'], ms=4, mfc=vg.marker_face(STATIC),
               mec=STATIC['color'], color=STATIC['color'], label='Static (context)'),
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=vg.FS['legend'], handletextpad=.35,
              labelspacing=.2, borderaxespad=.1)
    vg.g_ticks(ax, 'y')


def _ratio_test(ax, test):
    t = test.sort_values('noise')
    for _, r in t.iterrows():
        face = 'white' if float(r.noise) == 30. else FULL['color']
        ax.errorbar([r.noise], [r.empirical_minus_predicted],
                    yerr=[[r.empirical_minus_predicted-r.ci_low], [r.ci_high-r.empirical_minus_predicted]],
                    color=FULL['color'], marker='D', mfc=face, mec=FULL['color'], mew=.75,
                    ms=4.0, ls='none', capsize=2, elinewidth=.85, capthick=.75)
    ax.axhline(0, color=MID, ls=':', lw=vg.LW['ref'])
    ax.set_xticks([30, 60], ['30', '60'])
    ax.set_xlim(20, 70)
    ax.set_ylim(-1.2, 4.2)
    ax.set_xlabel('Noise amplitude (pA)')
    ax.set_ylabel('Empirical − predicted ratio')
    vg.g_ticks(ax, 'y')


def _delta_u(ax, curves):
    for noise in (30., 60.):
        g = curves[(curves.condition == 'dynamic') & np.isclose(curves.noise, noise)].sort_values('t_s')
        encn = NOISE[noise]
        ax.plot(g.t_s, g.var_phi_minus_psi_u, color=U_TRACE['color'], ls=encn['ls'], lw=1.05,
                label=encn['label'])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, None)
    ax.set_xlabel('Elapsed time (s)')
    ax.set_ylabel(r'Variance of $\delta_u$ (deg$^2$)')
    ax.legend(handles=_noise_key(U_TRACE['color']), loc='upper right', fontsize=vg.FS['legend'],
              handlelength=1.8, handletextpad=.4, borderaxespad=.1)
    vg.g_ticks(ax, 'xy')


def _window_panel(ax, curves):
    rows = _window_means(curves)
    x = np.arange(len(WINDOWS), dtype=float)
    for noise in (30., 60.):
        g = rows[np.isclose(rows.noise_pa, noise)]
        encn = NOISE[noise]
        face = 'white' if encn['mfc'] == 'none' else FULL['color']
        ax.plot(x, g.mean_var_delta_eff_deg2, color=FULL['color'], ls=encn['ls'], lw=1.05,
                marker=encn['marker'], ms=3.8, mfc=face, mec=FULL['color'], mew=.7,
                label=encn['label'])
    ax.set_xticks(x, ['0.5-2', '2-4', '4-8', '8-10'])
    ax.set_ylim(0, .30)
    ax.set_xlabel('Window (s)')
    ax.set_ylabel(r'Mean Var($\delta_\mathrm{eff}$) (deg$^2$)')
    ax.legend(loc='upper right', fontsize=vg.FS['legend'], handlelength=1.8,
              handletextpad=.4, borderaxespad=.1)
    vg.g_ticks(ax, 'y')


def draw_figure(d):
    """S7 (CPL revision 2): 2 x 2 grid, noise amplitude sigma_eta in each label row, closed axes."""
    L = vg.Layout(104)
    fig = L.fig
    X, W, H = (18., 101.), 64., 33.
    YT, YB = 61., 12.
    axa = L.axes(X[0], YT, W, H); _delta_eff(axa, d['curves'], 30., ylabel=True)
    axb = L.axes(X[1], YT, W, H); _delta_eff(axb, d['curves'], 60.)
    axc = L.axes(X[0], YB, W, H); _absolute_msd(axc, d['curves'], 30., ylabel=True, legend=True)
    axd = L.axes(X[1], YB, W, H); _absolute_msd(axd, d['curves'], 60.)
    for (k, x, y0, noise) in (('a', X[0], YT, 30), ('b', X[1], YT, 60), ('c', X[0], YB, 30), ('d', X[1], YB, 60)):
        L.letter(x - 16.5 if k in 'ac' else x - 15., y0 + H + 3.2, k, va='center')
        L.text(x + .3, y0 + H + 3.2, r'$\sigma_\eta$ = ' + f'{noise} pA', ha='left', va='center')
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
