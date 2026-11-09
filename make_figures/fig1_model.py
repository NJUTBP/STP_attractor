"""Manuscript v10 (GC_STP-text-14) layout; see CHANGES_v10.md.

Figure 1: persistent bump activity, a theta-band inhibitory current and latent STP states.

The module is split into two layers.

Data layer  -- ``compute_panels`` / ``export_source_data``.  The computations and the
               exported source-data files (names, columns, arrays) are exactly those of
               manuscript v6; nothing is smoothed, filtered, selected or re-normalised.
Plot layer  -- ``draw_figure``.  It receives only the arrays that are exported as source
               data, so the identical drawing code also re-renders the figure from its own
               source data on a machine without the canonical raw tree.

Manuscript v8 representation revision (plot layer only): panel b uses the field-standard
angle-sorted raster plus an unsmoothed per-neuron spatial firing-rate marginal (1-8 s); the
recorded PAV source file is preserved but no longer overplotted in b. Optional A/B renderings
for panel f and panels g/h are exposed as draw_figure keyword arguments. All source-data files
and schemas remain unchanged.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.patches import Circle, FancyArrowPatch

from dynamics import welch_proxy, theta_band, cycle_waveforms, replay_stp, pav
from plot_style import panel_letter
from captions_cpl import CAPTIONS
import visual_grammar as vg

STEM = 'Fig1_Real_network_dynamics'
CAPTION_KEY = 'Fig1'
# Source-data interface of Fig. 1 (unchanged since v6).
SOURCE_FILES = ['Fig1B_E_spikes.csv', 'Fig1B_PAV.csv', 'Fig1C_EI_zoom_spikes.csv', 'Fig1D_PSD.csv',
                'Fig1EF_STP_replay.npz']
# Reference-point baseline release probability: the lowest value u can take.  Used only as
# the floor of the u colour scale (no data are clipped: u >= U by construction).
U_REFERENCE = .4
# CPL revision 2 geometry (mm from the lower-left corner).  Panel (a) is the author's Fig1a.pdf (1951 x 2907 pt,
# aspect 0.6711), inserted by assemble_fig1_vector.py in this rectangle: same size as before, moved left/up.
FIG_H_MM = 134.
PANEL_A_RECT_MM = (3.5, 57.0, 48.66, 72.5)


# ================================================================== data layer (unchanged)
def _spike_table(raw, pop, start, end):
    t = raw[f'spike_times_{pop}_ms']/1000; a = raw[f'spike_angles_{pop}_deg']
    m = (t >= start) & (t <= end)
    return pd.DataFrame({'time_s': t[m], 'angle_deg': a[m], 'population': pop})


def compute_panels(raw, window=(5., 6.2), *, psd_override=None):
    """All quantities drawn in Fig. 1 (manuscript v10: panels a-f), computed exactly as in v6-v9."""
    start, end = window
    d = {'window': (float(start), float(end))}
    d['b_spikes'] = _spike_table(raw, 'e', 0, 8)
    tq = np.arange(900., 8000., 25.); p = pav(raw, tq)       # recorded activity position
    d['b_pav'] = pd.DataFrame({'time_ms': tq, 'axis_deg': p})
    d['c_spikes'] = pd.concat([_spike_table(raw, 'e', start, end), _spike_table(raw, 'i', start, end)],
                              ignore_index=True)
    # The historical 0-pA path still uses the unchanged 1–8 s Welch implementation.
    # A Phase-13 illustrative trial may have a shorter archived current trace.
    # In that case, its PSD is explicitly computed from *its actual recorded interval*
    # by the dedicated Fig. 1 exporter and passed here, never extrapolated.
    f, power, peak = welch_proxy(raw) if psd_override is None else psd_override
    d['d_psd'] = pd.DataFrame({'frequency_hz': f, 'psd_pA2_per_Hz': power})
    d['d_peak_hz'] = peak
    time = np.arange(start*1000, end*1000+.1, 5.)
    u, r = replay_stp(raw, time)
    d['ef_stp'] = {'time_ms': time, 'theta_e_deg': raw['theta_e_deg'], 'u': u, 'resources': r}
    return d


def export_source_data(ex, d):
    """Write the Fig. 1 source data.  v10: the former E (current) and F (cycles) files are no longer
    drawn; the u/D replay is now panels e,f."""
    ex.csv('Fig1B_E_spikes', d['b_spikes'])
    ex.csv('Fig1B_PAV', d['b_pav'])
    ex.csv('Fig1C_EI_zoom_spikes', d['c_spikes'])
    ex.csv('Fig1D_PSD', d['d_psd'])
    ex.npz('Fig1EF_STP_replay', **d['ef_stp'])


def load_source_data(source, window=(5., 6.2)):
    """Rebuild the panel dictionary from exported source data (replay; no recomputation).
    v9 folders also work: their ``Fig1GH_STP_replay.npz`` holds the same arrays."""
    source = Path(source)
    stp_file = 'Fig1EF_STP_replay.npz' if (source/'Fig1EF_STP_replay.npz').is_file() else 'Fig1GH_STP_replay.npz'
    missing = [n for n in SOURCE_FILES[:-1] + [stp_file] if not (source/n).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing Fig. 1 source data in {source}: {missing}')
    d = {'window': (float(window[0]), float(window[1]))}
    d['b_spikes'] = pd.read_csv(source/'Fig1B_E_spikes.csv')
    d['b_pav'] = pd.read_csv(source/'Fig1B_PAV.csv')
    d['c_spikes'] = pd.read_csv(source/'Fig1C_EI_zoom_spikes.csv')
    d['d_psd'] = pd.read_csv(source/'Fig1D_PSD.csv')
    f = d['d_psd'].frequency_hz.to_numpy(); p = d['d_psd'].psd_pA2_per_Hz.to_numpy()
    m = (f > 0) & (f <= 50)                                    # same peak rule as welch_proxy
    d['d_peak_hz'] = float(f[m][np.argmax(p[m])])
    # An optional *separate* provenance label is supplied only for the new
    # Fig. 1 noise90 export. Unlabelled legacy 0-pA sources remain untouched.
    provenance = source / 'Fig1_representative_trial.json'
    if provenance.is_file():
        d['representative_metadata'] = json.loads(provenance.read_text(encoding='utf-8'))
    with np.load(source/stp_file) as z:
        d['ef_stp'] = {k: z[k] for k in ('time_ms', 'theta_e_deg', 'u', 'resources')}
    start, end = d['window']
    t = d['ef_stp']['time_ms']/1000
    if not (np.isclose(t.min(), start) and np.isclose(t.max(), end)):
        raise ValueError(f'Fig1EF source data cover {t.min():.3f}-{t.max():.3f} s, not the '
                         f'requested window {start}-{end} s.')
    return d


def caption_for_source(d):
    """Return the manuscript Fig. 1 caption."""
    return CAPTIONS[CAPTION_KEY]


# ================================================================== plot layer (v7)
INK, RAW, MID, LIGHT, BAND = (vg.NEUTRAL[k] for k in ('ink', 'raw', 'mid', 'light', 'band'))
FULL = vg.MODELS['full']['color']


def draw_circuit(ax):
    """Placeholder schematic (panel a is to be redrawn by the authors; see the redesign notes)."""
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    for x, label, n in [(.21, 'E', 1024), (.78, 'I', 256)]:
        ax.add_patch(Circle((x, .71), .12, fill=False, linewidth=.95, color=RAW))
        ax.text(x, .72, label, ha='center', va='center', fontsize=9, fontweight='bold', color=RAW)
        ax.text(x, .49, f'{n} cells', ha='center', fontsize=vg.FS['note'], color=RAW)
    ax.add_patch(FancyArrowPatch((.32, .78), (.65, .78), arrowstyle='-|>', mutation_scale=8,
                                 linewidth=.9, color=RAW))
    ax.text(.49, .90, 'NMDA × $uR$', ha='center', fontsize=vg.FS['note'], color=RAW)
    ax.plot([.66, .35], [.64, .64], linewidth=.9, color=RAW, clip_on=False)
    ax.plot([.35, .35], [.595, .685], linewidth=.9, color=RAW, clip_on=False)
    ax.text(.50, .55, 'GABA', ha='center', fontsize=vg.FS['note'], color=RAW)
    ax.plot([.90, .98], [.71, .71], linewidth=.8, color=RAW, clip_on=False)
    ax.plot([.98, .98], [.665, .755], linewidth=.8, color=RAW, clip_on=False)
    ax.text(1.0, .90, 'global I–I', ha='right', fontsize=vg.FS['note'], color=RAW)
    ax.text(.03, .28, 'Two-bump ring · 180° symmetry', fontsize=vg.FS['note'], color=RAW)


def _angle_axis(ax):
    ax.set(ylim=(0, 360), yticks=[0, 180, 360])


def _persistent_rate_profile(d, t0=1.0, t1=8.0):
    """Per-neuron mean firing rate in the persistent interval; no binning or smoothing.

    The complete E-cell angle grid is already archived in Fig1EF_STP_replay.npz. Spike counts
    are obtained from Fig1B_E_spikes.csv and zero-filled for neurons with no spikes. This is a
    deterministic plotting summary of existing source data, not a new analysis interface.
    """
    theta = np.asarray(d['ef_stp']['theta_e_deg'], dtype=float)
    sp = d['b_spikes']
    sel = sp[(sp.time_s >= t0) & (sp.time_s <= t1)]
    counts = sel.groupby('angle_deg').size()
    rate = np.array([float(counts.get(float(a), 0)) / (t1 - t0) for a in theta], dtype=float)
    return theta, rate


def draw_figure(d, stp_marginals=False, layout_variant='A', schematic='pending'):
    """Fig. 1, CPL revision 2 (plot layer only; recorded data and plotting operations unchanged).

    Left column: (a) the author's ring schematic, placed by assemble_fig1_vector.py in PANEL_A_RECT_MM, above
    (d) the PSD, whose plotting area spans exactly the rows of (e) and (f).  Right column: (b) raster and rate
    marginal, (c) E/I rasters, (e) u and (f) D on one time axis.  All right-column letters share one x left of
    the shared axis labels; letters of one row share one height.  Closed axes throughout.
    ``schematic='pending'`` draws a geometry-only placeholder that must never be delivered.
    """
    if layout_variant != 'A':
        raise ValueError("CPL revision 2 maintains layout 'A' only.")
    if schematic not in ('pending', 'external'):
        raise ValueError('Fig. 1(a) requires the supplied external ring-network schematic')
    if stp_marginals:
        raise ValueError('The STP-marginal variant is retired in CPL revision 2.')
    start, end = d['window']
    L = vg.Layout(height_mm=FIG_H_MM)
    fig = L.fig
    AX, AW = 75., 82.                                   # time-aligned stack of the right column
    rows = {'cE': (75.5, 15.5), 'cI': (56.5, 15.), 'e': (33., 17.), 'f': (10., 17.)}
    raster_rect, marginal_rect = (AX, 102., 74., 23.5), (153., 102., 17., 23.5)
    psd_rect = (13.5, rows['f'][0], 42.5, rows['e'][0] + rows['e'][1] - rows['f'][0])
    YL = AX - 9.0                                       # shared rotated y labels of (c) and (e, f)
    LX_LEFT, LX_RIGHT = 1.5, 57.0                       # letter columns

    # ---- (a) placeholder rectangle for the vector schematic ------------------------------------------
    ax_a = L.axes(*PANEL_A_RECT_MM)
    ax_a.set(xlim=(0, 1), ylim=(0, 1)); ax_a.axis('off')
    if schematic == 'pending':
        from matplotlib.patches import Rectangle
        ax_a.add_patch(Rectangle((.01, .01), .98, .98, facecolor='#F5F5F5', edgecolor='#B0B0B0',
                                 linewidth=.6, linestyle='--'))
        ax_a.text(.5, .52, 'Fig. 1(a)\nschematic\nnot inserted', ha='center', va='center', fontsize=8., color='#666666')

    # ---- (b) raster + unsmoothed spatial rate marginal ----------------------------------------------
    ax_b = L.axes(*raster_rect)
    sp = d['b_spikes']
    ax_b.scatter(sp.time_s, sp.angle_deg, s=1.0, marker='o', linewidths=0, color=INK, rasterized=True, zorder=2)
    ax_b.set(xlim=(-.08, 8.08), xticks=range(0, 9), xlabel='Time (s)', ylabel='E-cell angle (°)')
    _angle_axis(ax_b)
    rep = d.get('representative_metadata')
    if rep is None or rep.get('cue_0_to_0p5_verified', False):
        vg.interval_bar(ax_b, 0., .5, 'cue', color=INK)
    vg.interval_bar(ax_b, start, end, f'{panel_letter("c")}, {panel_letter("e")}, {panel_letter("f")}', color=BAND)
    axp = L.axes(*marginal_rect, sharey=ax_b)
    theta, rate = _persistent_rate_profile(d, 1.0, 8.0)
    axp.plot(rate, theta, color=INK, lw=.85)
    rate_max = max(7.2, float(np.nanmax(rate)) * 1.07)
    ticks = [0, 3, 6] if rate_max <= 7.2 else [0, round(rate_max/2), round(rate_max)]
    axp.set(xlim=(-.4, rate_max), xticks=ticks, xlabel='Rate (Hz)')
    _angle_axis(axp)
    axp.tick_params(axis='y', labelleft=False)
    vg.g_ticks(axp, 'x')

    # ---- (c) E and I rasters in the expanded interval ---------------------------------------------------
    sc = d['c_spikes']
    axE = L.axes(AX, rows['cE'][0], AW, rows['cE'][1])
    axI = L.axes(AX, rows['cI'][0], AW, rows['cI'][1], sharex=axE)
    ev, iv = sc[sc.population == 'e'], sc[sc.population == 'i']
    axE.scatter(ev.time_s, ev.angle_deg, s=1.7, marker='o', linewidths=0, color=INK, rasterized=True)
    axI.scatter(iv.time_s, iv.angle_deg, s=.45, marker='o', linewidths=0, color=RAW, alpha=.6, rasterized=True)
    for ax, tag in ((axE, 'E'), (axI, 'I')):
        _angle_axis(ax); vg.hide_x(ax)
        ax.text(1.012, .5, tag, transform=ax.transAxes, ha='left', va='center', fontweight='bold',
                fontsize=vg.FS['label'], color=RAW)
    L.text(YL, rows['cI'][0] + (rows['cE'][0] + rows['cE'][1] - rows['cI'][0]) / 2, 'Cell angle (°)',
           rotation=90, ha='center', va='center')

    # ---- (e), (f) reconstructed facilitation u and depletion D = 1 - R ----------------------------------
    stp = d['ef_stp']
    maps = [
        ('e', stp['u'], vg.CMAPS['u'], Normalize(vmin=U_REFERENCE, vmax=1.), '$u$', [U_REFERENCE, 1.]),
        ('f', 1 - stp['resources'], vg.CMAPS['depletion'], PowerNorm(gamma=0.5, vmin=0., vmax=1.), '$D$', [0., 0.25, 1.]),
    ]
    for key, matrix, cmap, norm, label, ticks in maps:
        bottom, height = rows[key]
        ax = L.axes(AX, bottom, AW, height, sharex=axE)
        im = ax.imshow(matrix.T, origin='lower', aspect='auto', extent=[start, end, 0, 360], cmap=cmap,
                       norm=norm, interpolation='none', rasterized=True)
        _angle_axis(ax)
        cax = L.axes(AX + AW + 1.6, bottom, 1.8, height)
        cb = fig.colorbar(im, cax=cax, ticks=ticks)
        cb.outline.set_linewidth(vg.LW['axis'])
        cax.tick_params(length=1.8, width=vg.LW['axis'], pad=1.2, labelsize=vg.FS['tick'])
        cb.ax.set_yticklabels([f'{t:g}' for t in ticks])
        cax.set_title(label, fontsize=vg.FS['label'], pad=2.0, loc='left')
        if key == 'e':
            vg.hide_x(ax)
        else:
            ax.set(xlim=(start, end), xlabel='Time (s)')
            ax.set_xticks(np.round(np.arange(start, end + 1e-9, .2), 1))
    L.text(YL, rows['f'][0] + (rows['e'][0] + rows['e'][1] - rows['f'][0]) / 2, 'E-cell angle (°)',
           rotation=90, ha='center', va='center')

    # ---- (d) PSD of the inhibitory current (1-8 s); 0-50 Hz = the dominant-frequency search range -------
    ax_d = L.axes(*psd_rect)
    psd = d['d_psd']
    freq = psd.frequency_hz.to_numpy(float)
    power = psd.psd_pA2_per_Hz.to_numpy(float) / 1e3          # pA^2/Hz -> 10^-3 nA^2/Hz (unit change only)
    ax_d.plot(freq, power, color=FULL, lw=1.1)
    pmax = float(np.nanmax(power))
    f0 = float(d['d_peak_hz'])
    for k, h_label in ((1, '$f_0$'), (2, '$2f_0$'), (3, '$3f_0$')):
        hm = (freq >= k * f0 - 1.5) & (freq <= k * f0 + 1.5)
        if np.any(hm):
            idx = int(np.where(hm)[0][np.argmax(power[hm])])
            ax_d.text(float(freq[idx]), float(power[idx]) + .022 * pmax, h_label,
                      ha='center', va='bottom', fontsize=vg.FS['note'], color=INK)
    ax_d.set(xlim=(0, vg.PSD_FMAX_HZ), xticks=np.arange(0, vg.PSD_FMAX_HZ + 1, 10),
             ylim=(-.03 * pmax, 1.20 * pmax), yticks=[0, 2, 4, 6, 8, 10],
             xlabel='Frequency (Hz)', ylabel=r'PSD (10$^{-3}$ nA$^2$ Hz$^{-1}$)')
    vg.interval_bar(ax_d, 4., 12., '4–12 Hz', color=BAND)

    # ---- letters: one x per column, one y per row -----------------------------------------------------
    top_row = 129.0
    for k, (x, y) in {'a': (LX_LEFT, top_row), 'b': (LX_RIGHT, top_row),
                      'c': (LX_RIGHT, rows['cE'][0] + rows['cE'][1] + 2.0),
                      'd': (LX_LEFT, psd_rect[1] + psd_rect[3] + 2.0),
                      'e': (LX_RIGHT, rows['e'][0] + rows['e'][1] + 2.0),
                      'f': (LX_RIGHT, rows['f'][0] + rows['f'][1] + 2.0)}.items():
        L.letter(x, y, k, va='center')
    return fig


def make_figure(ds, ex, window=(5., 6.2)):
    raw = ds.theta('dynamic')
    d = compute_panels(raw, window)
    export_source_data(ex, d)
    with vg.style():
        fig = draw_figure(d)
        ex.save(fig, STEM, CAPTIONS[CAPTION_KEY])
