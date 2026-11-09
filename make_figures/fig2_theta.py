"""Manuscript v10 (GC_STP-text-14) layout; see CHANGES_v10.md.

CPL revision 3 (2026-10-27): the three panels of the former Supplementary Fig. S11 form the bottom row of
Fig. 2 -- (d) E->I drive of one neuron, (e) fixed-release networks F(rho), (f) self-consistent fixed point --
drawn by fig2_mechanism.py from the frozen RevS11* tables ('mechanism' layout, 145 mm). Panels (a)-(c) keep
the exact CPL revision-2 geometry, shifted up by ROW_SHIFT_MM; '--layout legacy-short' redraws the old figure.

Figure 2: route isolation -- example currents, spectra, frequency reconstruction and maps.

Data layer  -- ``compute_panels`` / ``export_source_data``: computations and exported source
               data (file names, columns) are exactly those of manuscript v6.
Plot layer  -- ``draw_figure``: manuscript v7 visual revision, drawn only from the exported
               arrays, so ``tools/replay_v7.py`` can re-render the figure from its source data.

v7 visual revision (plot layer only): conditions carry the manuscript-wide encoding
(colour + marker + line style from ``visual_grammar``) so the full/release similarity and the
output-only deviation are visible without text; no in-panel titles or numbers; neutral labels;
theta band as a bar; frequency maps on a shared logarithmic colour scale.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator

from data_io import LABELS
from dynamics import welch_proxy
from plot_style import panel_letter
from captions_cpl import CAPTIONS
import visual_grammar as vg
import fig2_mechanism as mech

STEM = 'Fig2_Theta_route_mechanism'
CAPTION_KEY = 'Fig2'
MODES = ['dynamic', 'u_output_only', 'u_release_only', 'static_gain']
SOURCE_FILES = ['Fig2B_waveforms.csv', 'Fig2C_reconstruction.csv']
MECH_FILES = list(mech.FILES)          # panels (d)-(f): frozen RevS11* tables in make_suppfigures/source_data
RECON_COLUMNS = ['combo_id', 'tau_facilitation_ms', 'tau_recovery_ms', 'full_frequency_hz',
                 'u_output_only_frequency_hz', 'u_release_only_frequency_hz']


# ================================================================== data layer (unchanged)
def _anchor_row(tab):
    row = tab[(tab.tau_facilitation_ms == 1000.) & (tab.tau_recovery_ms == 20.)]
    if len(row) != 1:
        raise ValueError('Expected exactly one predefined anchor row at tau_f=1000 ms, tau_r=20 ms.')
    return row.iloc[0]


def compute_panels(ds, window=(5., 6.2)):
    """All quantities drawn in Fig. 2 (manuscript v10: panels a-c), computed exactly as in v6-v9."""
    raw = {c: ds.theta(c) for c in MODES}
    tab = ds.table('routes')
    start, end = window
    d = {'window': (float(start), float(end))}
    waves = []
    for c in MODES:
        r = raw[c]
        t = r['sample_times_ms']/1000; x = r['plfp_gaba_e']
        center = x[(t >= 1) & (t <= 8)].mean(); m = (t >= start) & (t <= end)
        waves.append(pd.DataFrame({'condition': c, 'time_s': t[m], 'raw_pA': x[m], 'centered_pA': x[m]-center}))
    d['b_waves'] = pd.concat(waves, ignore_index=True)
    d['c_tab'] = tab[RECON_COLUMNS]
    d['anchor'] = _anchor_row(tab)
    d['mech'] = mech.load_mechanism_data()          # (d)-(f): frozen former-S11 tables, verified
    return d


def export_source_data(ex, d):
    """Write the Fig. 2 source data.  v10: the former PSD panel is not drawn and the frequency maps
    are Supplementary Fig. S3a-c (built by make_suppfigures)."""
    ex.csv('Fig2B_waveforms', d['b_waves'])
    ex.csv('Fig2C_reconstruction', d['c_tab'])


def load_source_data(source, window=(5., 6.2), mech_source='auto'):
    """Rebuild the panel dictionary from exported source data (replay; v9 folders also work).

    mech_source: folder with the frozen RevS11* tables of panels (d)-(f); 'auto' = ../make_suppfigures/source_data,
    None = do not load them (legacy-short layout only).
    """
    source = Path(source)
    recon = 'Fig2C_reconstruction.csv' if (source/'Fig2C_reconstruction.csv').is_file() else 'Fig2D_reconstruction.csv'
    missing = [n for n in ('Fig2B_waveforms.csv', recon) if not (source/n).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing Fig. 2 source data in {source}: {missing}')
    d = {'window': (float(window[0]), float(window[1]))}
    d['b_waves'] = pd.read_csv(source/'Fig2B_waveforms.csv')
    tab = pd.read_csv(source/recon)
    d['c_tab'] = tab[RECON_COLUMNS]
    d['anchor'] = _anchor_row(tab)
    t = d['b_waves'].time_s.to_numpy()
    if not (np.isclose(t.min(), window[0]) and np.isclose(t.max(), window[1])):
        raise ValueError(f'Fig2B source data cover {t.min():.3f}-{t.max():.3f} s, not {window}.')
    if mech_source is not None:
        d['mech'] = mech.load_mechanism_data(None if mech_source == 'auto' else mech_source)
    return d


# ================================================================== plot layer (v7)
# All coordinates are in millimetres from the BOTTOM-LEFT of the figure.
# Keep the geometry in one place so the PDF assembler and data-panel renderer
# cannot accidentally disagree about the page height or Fig2a placement.
# The short layout is selected automatically by assemble_fig2_vector.py when
# the author replaces the original portrait Fig2a with the low-profile PDF.
ROW_SHIFT_MM = 60.0     # CPL revision 3: height of the added bottom row (d)-(f)

FIG2_LAYOUTS = {
    # Legacy tall-artwork layouts (kept for provenance; not drawn in CPL revision 2).
    'compact': {'height_mm': 109.0, 'a_box': (6., 48., 62., 58.)},
    'balanced': {'height_mm': 112.0, 'a_box': (6., 48., 70., 62.5)},
    # CPL revision 2.  a_box is shifted up by 3.5 mm and c_box height is reduced from 38 mm to 33.5 mm
    # so that the 2-row legend above panel (c) has generous breathing room below schematic (a).
    'short': {
        'height_mm': 85.0,
        'a_box': (6.0, 53.5, 62.0, 28.41),
        'c_box': (17.0, 9.0, 38.0, 33.5),
        'b_box': (74.0, 9.0, 87.0, 69.41),
    },
    # CPL revision 3.  (a)-(c): the 'short' boxes shifted up by ROW_SHIFT_MM.
    # (d)-(f): the geometry of the former S11 (three 36 x 38 mm axes from x = 16 mm with 23-mm gaps), reordered
    # along the mechanism chain: d = S11c (drive kernel), e = S11a (F(rho)), f = S11b (fixed point).
    'mechanism': {
        'height_mm': 85.0 + ROW_SHIFT_MM,
        'a_box': (6.0, 53.5 + ROW_SHIFT_MM, 62.0, 28.41),
        'c_box': (17.0, 9.0 + ROW_SHIFT_MM, 38.0, 33.5),
        'b_box': (74.0, 9.0 + ROW_SHIFT_MM, 87.0, 69.41),
        'd_box': (16.0, 11.0, 36.0, 38.0),
        'e_box': (75.0, 11.0, 36.0, 38.0),
        'f_box': (134.0, 11.0, 36.0, 38.0),
        'top_letter_y': 82.0 + ROW_SHIFT_MM,
        'bottom_letter_y': 55.0,
        'letter_x': {'d': 1.5, 'e': 75.0 - 14.5, 'f': 134.0 - 14.5},
    },
}

KEY = {'dynamic': 'full', 'full': 'full', 'u_output_only': 'output', 'u_release_only': 'release',
       'static_gain': 'static'}
INK, MID = vg.NEUTRAL['ink'], vg.NEUTRAL['mid']


def _enc(condition):
    return vg.MODELS[KEY[condition]]


def _marker(ax, x, y, condition, size=4.2, transform=None, **kw):
    e = _enc(condition)
    face = 'white' if e['mfc'] == 'none' else e['color']
    ax.plot([x], [y], linestyle='none', marker=e['marker'], markersize=size, markerfacecolor=face,
            markeredgecolor=e['color'], markeredgewidth=.8, clip_on=False,
            transform=transform if transform is not None else ax.transData, **kw)


def route_table(ax):
    """Placeholder for panel a (to be redrawn by the authors; see the redesign notes)."""
    ax.axis('off')
    rows = [['Full', r'$uR$', r'$u^+$'], ['Output-only', r'$k\,uR$', r'$U$'],
            ['Release-only', r'$k\,UR$', r'$u^+$'], ['Static', r'$\bar{p}$', '—']]
    table = ax.table(cellText=rows, colLabels=['Model', 'Synaptic\noutput', 'Depletion\nuses'],
                     loc='upper center', cellLoc='center', colWidths=[.40, .29, .31], bbox=[0, .22, 1, .76])
    table.auto_set_font_size(False)
    table.set_fontsize(vg.FS['note'])
    for (i, j), cell in table.get_celld().items():
        cell.set_linewidth(vg.LW['min']); cell.set_edgecolor('#A8A8A8')
        if i == 0:
            cell.set_text_props(fontweight='bold', color=INK); cell.set_facecolor('#F4F4F4')
    ax.text(0, .15, r'$u^+$: post-spike facilitation; $k$: calibrated scale factor', fontsize=vg.FS['note'],
            color=MID, transform=ax.transAxes)


def draw_figure(d, layout_variant='mechanism'):
    """Data panels (b)-(f); panel (a) is left free for the author's vector Fig2a.pdf.

    CPL revision 3 ('mechanism', default): (b)-(c) as in revision 2 plus the bottom row (d)-(f) of the release-fraction
    mechanism (former Supp. Fig. S11, see fig2_mechanism.py). 'short' redraws the revision-2 figure.

    (b): the four mean-subtracted currents share one vertical scale, given by the 0.5-nA bar; they are offset
    vertically only so that they do not overlap, so there is no y axis -- only the time axis. A single-row
    model legend sits above the top trace.
    (c): x-axis narrowed to 3.0-11.5 Hz (matching panel (f) so there is no wasted margin on the left or right)
    while keeping y-axis at 0-33 Hz; panel height is 33.5 mm and schematic (a) is shifted up so the legend
    above (c) sits cleanly without overlapping (a).
    """
    from matplotlib.transforms import ScaledTranslation, blended_transform_factory
    if layout_variant not in ('short', 'mechanism'):
        raise ValueError("Fig. 2 is drawn with the 'mechanism' layout (CPL revision 3) or the legacy 'short' layout.")
    if layout_variant == 'mechanism' and 'mech' not in d:
        raise ValueError("The 'mechanism' layout needs the panel (d)-(f) tables: load_source_data(..., mech_source=...).")
    start, end = d['window']
    lay = FIG2_LAYOUTS[layout_variant]
    top = lay.get('top_letter_y', 82.0)
    L = vg.Layout(height_mm=lay['height_mm'])
    fig = L.fig
    ax_a = L.axes(*lay['a_box'])
    ax_a.axis('off')

    # ---- (b) example currents ------------------------------------------------------------------------
    ax_b = L.axes(*lay['b_box'])
    waves = d['b_waves']
    series = {c: waves[waves.condition == c].sort_values('time_s') for c in MODES}
    yv = {c: s.centered_pA.to_numpy() / 1000 for c, s in series.items()}     # pA -> nA (unit change only)
    LABEL_BAND, GAP = .55, .15                                                # nA
    offsets, cursor = {}, 0.
    for c in reversed(MODES):
        offsets[c] = cursor - float(np.nanmin(yv[c]))
        cursor = offsets[c] + float(np.nanmax(yv[c])) + LABEL_BAND + GAP
    for c in MODES:
        e = _enc(c)
        ax_b.plot(series[c].time_s, yv[c] + offsets[c], color=e['color'], lw=1.0, zorder=3)
    y_label = offsets[MODES[0]] + float(np.nanmax(yv[MODES[0]])) + .5 * LABEL_BAND
    # CPL revision 4: the four traces are unmarked solid lines (lw 1.0), so the key shows short solid lines in
    # the trace colours instead of the model marker glyphs (those stay in the scatter panel (c)).
    handles = [Line2D([], [], color=_enc(c)['color'], ls='-', lw=1.0, solid_capstyle='butt') for c in MODES]
    ax_b.legend(handles=handles, labels=[LABELS[c] for c in MODES], ncol=4, mode='expand',
                loc='center left', bbox_to_anchor=(.015, y_label, .985, 0.),
                bbox_transform=blended_transform_factory(ax_b.transAxes, ax_b.transData),
                frameon=False, borderaxespad=0., borderpad=0., handlelength=1.2,
                handletextpad=.35, columnspacing=.8, fontsize=vg.FS['label'])
    ax_b.set(xlim=(start, end), ylim=(-.15, cursor - GAP + .05), xlabel='Time (s)')
    ax_b.set_xticks(np.round(np.arange(start, end + 1e-9, .2), 1))
    ax_b.set_yticks([])
    ax_b.spines['left'].set_visible(False)
    base = blended_transform_factory(ax_b.transAxes, ax_b.transData)
    y0 = offsets[MODES[-1]]                       # zero level of the bottom trace
    ax_b.plot([1, 1], [y0, y0 + .5], transform=base + ScaledTranslation(1.6 * vg.MM, 0, fig.dpi_scale_trans),
              color=INK, lw=1.3, solid_capstyle='butt', clip_on=False)
    ax_b.text(1, y0 + .25, '0.5 nA', transform=base + ScaledTranslation(2.6 * vg.MM, 0, fig.dpi_scale_trans),
              ha='left', va='center', fontsize=vg.FS['note'], color=INK, clip_on=False)

    # ---- (c) frequency reconstruction across the 42 settings ---------------------------------------------
    ax_c = L.axes(*lay['c_box'])
    tab, anchor = d['c_tab'], d['anchor']
    xlim_min, xlim_max = 3.0, 11.5
    ylim_min, ylim_max = 0., 33.
    ax_c.plot([xlim_min, xlim_max], [xlim_min, xlim_max], ls=':', lw=vg.LW['ref'], color=MID, zorder=0)
    _, _, c_w, c_h = lay['c_box']
    angle_deg = np.degrees(np.arctan((c_h / (ylim_max - ylim_min)) / (c_w / (xlim_max - xlim_min))))
    ax_c.text(12.8, 13.0, 'Identity', rotation=angle_deg, ha='center', va='center', fontsize=vg.FS['note'], color=MID)
    for c in ('u_output_only', 'u_release_only'):
        e = _enc(c)
        ax_c.scatter(tab.full_frequency_hz, tab[c + '_frequency_hz'], s=vg.MS['scatter'] + 4, marker=e['marker'],
                     color=e['color'], edgecolors='white', linewidths=.4, alpha=.92, zorder=2)
    for y in (anchor.u_output_only_frequency_hz, anchor.u_release_only_frequency_hz):
        ax_c.scatter(anchor.full_frequency_hz, y, s=80, facecolors='none', edgecolors=INK, linewidths=.9, zorder=4)
    handles = [vg.model_handle('output', line=False), vg.model_handle('release', line=False),
               Line2D([], [], ls='none', marker='o', markersize=6.4, markerfacecolor='none', markeredgecolor=INK,
                      markeredgewidth=.9, label='Reference point')]
    ax_c.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, 1.03), ncol=2,
                frameon=False, handletextpad=.25, columnspacing=.7, labelspacing=.18, borderaxespad=0.)
    ax_c.set(xlim=(xlim_min, xlim_max), ylim=(ylim_min, ylim_max),
             xticks=[4, 6, 8, 10], yticks=[0, 10, 20, 30],
             xlabel='Full-model frequency (Hz)', ylabel='Isolated-route frequency (Hz)')
    vg.g_ticks(ax_c, 'xy')

    # ---- letters ---------------------------------------------------------------------------------------
    L.letter(1.5, top, 'a', va='center')
    L.letter(lay['b_box'][0] - 2.5, top, 'b', va='center')
    L.letter(1.5, lay['c_box'][1] + lay['c_box'][3] + 2.0, 'c', va='center')

    # ---- (d)-(f) release-fraction mechanism (former Supp. Fig. S11; CPL revision 3) -------------------------
    if layout_variant == 'mechanism':
        mech.draw_row(L, d['mech'], {k: lay[k + '_box'] for k in 'def'}, lay['bottom_letter_y'], lay['letter_x'])
    return fig


def make_figure(ds, ex, window=(5., 6.2)):
    d = compute_panels(ds, window)
    export_source_data(ex, d)
    with vg.style():
        fig = draw_figure(d)
        ex.save(fig, STEM, CAPTIONS[CAPTION_KEY])