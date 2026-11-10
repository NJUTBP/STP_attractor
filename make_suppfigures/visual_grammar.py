"""Manuscript-wide visual grammar (plotting layer only).

CPL revision 2 (figure-polish pass)
-----------------------------------
* Open L-shaped axes: only the left and bottom spines are drawn (no top/right frame), neither is moved
  outward, so the x and y axes meet in one point at the lower-left corner.  ``offset_spines`` is kept as a
  no-op for backward compatibility; ``categorical`` keeps the bottom spine so the two axes still meet.
* Final-size typography for a 174-mm drawing that the manuscript inserts 146.5 mm wide (factor 0.842):
  ticks 7.8 pt, axis labels 8.4 pt, panel letters 9.6 pt bold.
* Grey-scale-safe encodings: every condition differs in colour *and* marker shape / fill / line style;
  markers are larger than before (scatter area 24 pt^2, line markers 4.4 pt).
* Panel letters are placed by each figure on an explicit millimetre grid (one x per column, one y per row).
Nothing here reads, filters or transforms data.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation, blended_transform_factory

MM = 1 / 25.4
FIG_WIDTH_MM = 174.0
PRINT_WIDTH_MM = 146.5

FONT_SANS = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
FS = {'letter': 9.6, 'label': 8.4, 'tick': 7.8, 'legend': 7.8, 'note': 7.8, 'cbar': 8.0}
LW = {'axis': .65, 'data': 1.2, 'thin': .6, 'raw': .55, 'ref': .6, 'bar': 1.6, 'min': .5}
MS = {'line': 4.4, 'scatter': 24., 'key': 4.6}
# Spectra are drawn over the dominant-frequency search range of Sec. S5 (0-50 Hz); change here only.
PSD_FMAX_HZ = 50.
# White outline that lifts a summary curve off grey single-trial curves (also in grey-scale print).
HALO = [patheffects.Stroke(linewidth=2.9, foreground='white'), patheffects.Normal()]

PANEL_LABEL_FORMAT = '({})'
PANEL_LABEL_WEIGHT = 'bold'


def panel_label(text):
    """Journal-style panel label; idempotent."""
    s = str(text).strip()
    if len(s) == 1 and s.isalpha():
        return PANEL_LABEL_FORMAT.format(s)
    return s


# Lower-case italic psi from DejaVu Serif (Arial's italic psi reads as a capital at 6-7 pt), routed through
# mathtext's \mathcal slot; no label uses \mathcal for anything else.
PSI_FONT = 'DejaVu Serif:italic'


def psi(sub=None):
    return r'$\mathcal{\psi}$' if sub is None else r'$\mathcal{\psi}_{%s}$' % sub


NEUTRAL = {'ink': '#1A1A1A', 'raw': '#4D4D4D', 'mid': '#8C8C8C', 'light': '#BDBDBD', 'band': '#9E9E9E'}

MODELS = {
    'full':         {'color': '#0072B2', 'marker': 'o', 'mfc': None,   'ls': '-',  'label': 'Full'},
    'output':       {'color': '#E69F00', 'marker': '^', 'mfc': None,   'ls': '--', 'label': 'Output-only'},
    'release':      {'color': '#009E73', 'marker': 's', 'mfc': None,   'ls': '-.', 'label': 'Release-only'},
    'static':       {'color': '#7A7A7A', 'marker': 'o', 'mfc': 'none', 'ls': ':',  'label': 'Static'},
    'mean_release': {'color': '#CC79A7', 'marker': 'D', 'mfc': None,
                     'ls': (0, (3, 1, 1, 1, 1, 1)), 'label': 'Mean-release'},
    'frozen':       {'color': '#3F3F3F', 'marker': 'D', 'mfc': 'none', 'ls': '--', 'label': 'Frozen profile'},
    'baseline':     {'color': '#A0A0A0', 'marker': 'X', 'mfc': None,   'ls': '-.', 'label': 'Baseline'},
}
VARIABLES = {'activity': '#0072B2', 'u': '#E69F00', 'resource': '#009E73', 'drive': '#D55E00'}
CMAPS = {'u': 'Oranges', 'depletion': 'Greens', 'frequency': 'cividis', 'difference': 'RdBu_r', 'rate': 'Greys'}


def rc():
    return {
        'font.family': 'sans-serif', 'font.sans-serif': FONT_SANS,
        'font.size': FS['label'], 'axes.labelsize': FS['label'], 'axes.titlesize': FS['label'],
        'xtick.labelsize': FS['tick'], 'ytick.labelsize': FS['tick'], 'legend.fontsize': FS['legend'],
        'legend.frameon': False, 'legend.handlelength': 1.9, 'legend.handletextpad': .5,
        'legend.borderaxespad': .35, 'legend.labelspacing': .3, 'legend.columnspacing': 1.1,
        'legend.borderpad': .2,
        'axes.linewidth': LW['axis'], 'axes.labelpad': 2.5, 'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': True, 'axes.spines.bottom': True,
        'lines.linewidth': LW['data'], 'lines.markersize': MS['line'], 'lines.markeredgewidth': .8,
        'xtick.major.size': 2.6, 'ytick.major.size': 2.6, 'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
        'xtick.major.width': LW['axis'], 'ytick.major.width': LW['axis'],
        'xtick.minor.width': .5, 'ytick.minor.width': .5,
        'xtick.major.pad': 2.0, 'ytick.major.pad': 2.0,
        'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.top': False, 'ytick.right': False,
        'mathtext.fontset': 'custom', 'mathtext.rm': 'sans', 'mathtext.it': 'sans:italic',
        'mathtext.bf': 'sans:bold', 'mathtext.sf': 'sans', 'mathtext.cal': PSI_FONT, 'mathtext.tt': 'monospace',
        'axes.unicode_minus': True,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.dpi': 600, 'figure.dpi': 120,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    }


def _clear_text_caches():
    """Drop matplotlib's cached text layouts so they are recomputed with the current x-height rule."""
    import matplotlib.mathtext as mathtext
    import matplotlib.text as mtext
    for obj in (getattr(mathtext.MathTextParser, '_parse_cached', None),
                getattr(mtext, '_get_text_metrics_with_cache_impl', None)):
        if hasattr(obj, 'cache_clear'):
            obj.cache_clear()


@contextmanager
def glyph_xheight():
    """CPL revision 4.  Matplotlib's mathtext reads the x-height from a font's PCLT table when the font has one
    (Windows Arial does; Liberation Sans and DejaVu do not) and scales it by (fontsize/12)*(dpi/100), which comes
    out about 1.9 times the real x-height of Arial.  Every super- and subscript is then pushed that much too far
    from the baseline (10^-3, nA^2, Hz^-1 float above the unit; F_rest, tau_r sink).  Within the context the
    x-height is measured from the 'x' glyph for every font -- matplotlib's own rule for fonts without a PCLT
    table -- so Arial and its metric clone Liberation Sans set scripts identically.  Fonts without a PCLT table
    are unaffected (identical output); only the vertical script offsets change, never text, data or layout
    boxes.  Restored on exit; if the private hook is missing, matplotlib is left untouched."""
    import matplotlib as mpl
    try:
        from matplotlib import _mathtext
        fonts = _mathtext.TruetypeFonts
        original = fonts.get_xheight
    except (ImportError, AttributeError):
        yield
        return

    def get_xheight(self, fontname, fontsize, dpi):
        font = self._get_font(fontname)
        font.set_size(fontsize, dpi)
        return self.get_metrics(fontname, mpl.rcParams['mathtext.default'], 'x', fontsize, dpi).iceberg

    fonts.get_xheight = get_xheight
    _clear_text_caches()
    try:
        yield
    finally:
        fonts.get_xheight = original
        _clear_text_caches()


@contextmanager
def style():
    """Apply the grammar while a figure is built *and* saved."""
    with plt.rc_context(rc()), glyph_xheight():
        yield


def marker_face(enc):
    return 'white' if enc.get('mfc') == 'none' else enc['color']


class Layout:
    """Axes, letters and keys placed in millimetres from the figure's bottom-left corner."""

    def __init__(self, height_mm, width_mm=FIG_WIDTH_MM, y_shift_mm=0.):
        self.w, self.h, self.dy = float(width_mm), float(height_mm), float(y_shift_mm)
        self.fig = plt.figure(figsize=(self.w * MM, self.h * MM))

    def axes(self, left, bottom, width, height, **kw):
        return self.fig.add_axes([left / self.w, (bottom + self.dy) / self.h, width / self.w, height / self.h], **kw)

    def letter(self, x, y, text, va='top'):
        """Panel label "(a)"; (x, y) = left edge and top (va='top') or vertical centre (va='center')."""
        return self.fig.text(x / self.w, (y + self.dy) / self.h, panel_label(text), ha='left', va=va,
                             fontsize=FS['letter'], fontweight=PANEL_LABEL_WEIGHT)

    def text(self, x, y, s, **kw):
        kw.setdefault('fontsize', FS['label'])
        return self.fig.text(x / self.w, (y + self.dy) / self.h, s, **kw)

    def letter_title(self, x, y, letter, title, gap_mm=1.3, **kw):
        """Panel label followed by a short title at a constant gap after the label's right edge."""
        t = self.letter(x, y, letter, va='center')
        try:
            x1 = t.get_window_extent(self.fig.canvas.get_renderer()).x1 / self.fig.dpi * 25.4
        except Exception:                                   # canvas without a renderer: estimate
            x1 = x + .62 * FS['letter'] * len(panel_label(letter)) * 25.4 / 72
        kw.setdefault('fontsize', FS['label'])
        return self.text(x1 + gap_mm, y, title, ha='left', va='center', **kw)

    def key(self, x, y, enc, text=None, size=None, gap_mm=1.8, fontsize=None, color=None):
        """Condition glyph (colour + marker + fill) centred at (x, y) mm, followed by a neutral label."""
        self.fig.add_artist(Line2D([x / self.w], [(y + self.dy) / self.h], transform=self.fig.transFigure,
                                   linestyle='none', marker=enc['marker'], markersize=size or MS['key'],
                                   markerfacecolor=marker_face(enc), markeredgecolor=enc['color'],
                                   markeredgewidth=.8))
        if text:
            return self.text(x + gap_mm, y, text, ha='left', va='center', fontsize=fontsize or FS['label'],
                             color=color or NEUTRAL['ink'])


def fig_legend(L, handles, x_mm, y_mm, loc='center left', ncol=None, **kw):
    """Figure-level key anchored at (x_mm, y_mm)."""
    kw.setdefault('columnspacing', 1.5)
    kw.setdefault('handlelength', 2.3)
    kw.setdefault('borderaxespad', 0.)
    kw.setdefault('fontsize', FS['legend'])
    return L.fig.legend(handles=handles, loc=loc, bbox_to_anchor=(x_mm / L.w, (y_mm + L.dy) / L.h),
                        ncol=ncol or len(handles), frameon=False, **kw)


def model_handle(enc, line=True, ms=None, lw=None, label=None):
    enc = MODELS[enc] if isinstance(enc, str) else enc
    return Line2D([], [], color=enc['color'], ls=enc['ls'] if line else 'none', lw=lw or LW['data'],
                  marker=enc['marker'], markersize=ms or MS['key'], markerfacecolor=marker_face(enc),
                  markeredgecolor=enc['color'], markeredgewidth=.8, label=label or enc.get('label'))


def axes_frame(ax):
    """Open L-shaped frame: left and bottom axes meet in the lower-left corner (no offset, no top/right)."""
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        sp = ax.spines[side]
        sp.set_visible(True)
        sp.set_linewidth(LW['axis'])
        sp.set_position(('outward', 0.))
    return ax


closed = axes_frame          # name used by revision-2 code


def offset_spines(ax, which=('left', 'bottom'), points=0.):
    """Backward-compatible no-op: spines stay on the plotting area so the axes meet in one corner."""
    for side in which:
        ax.spines[side].set_position(('outward', 0.))


def hide_x(ax, spine=True):
    """Upper panel of a time-aligned stack: keep ticks and frame, drop tick labels."""
    ax.tick_params(axis='x', labelbottom=False)
    ax.set_xlabel('')


def categorical(ax):
    """Category panels: L-shaped frame (axes meet in the corner), categories label the columns (no x ticks)."""
    axes_frame(ax)
    ax.tick_params(axis='x', length=0)


def zero_line(ax, y=0., vertical=False):
    kw = dict(color=NEUTRAL['mid'], ls=':', lw=LW['ref'], zorder=0)
    return ax.axvline(y, **kw) if vertical else ax.axhline(y, **kw)


def pad(lo, hi, frac=.05, lo_frac=None, hi_frac=None):
    span = float(hi) - float(lo)
    return (float(lo) - (frac if lo_frac is None else lo_frac) * span,
            float(hi) + (frac if hi_frac is None else hi_frac) * span)


def line_angle(ax, slope):
    """Screen angle (deg) of a data-space line of the given slope (for rotated guide labels)."""
    fig = ax.figure
    bb = ax.get_position()
    w, h = bb.width * fig.get_figwidth(), bb.height * fig.get_figheight()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return math.degrees(math.atan(slope * (h / (y1 - y0)) / (w / (x1 - x0))))


def interval_bar(ax, x0, x1, label=None, color=NEUTRAL['ink'], y=None, lw=LW['bar'], above=True, offset_mm=1.5):
    """Thin bar just above the plotting area marking an x interval (cue, zoom window, frequency band)."""
    fig = ax.figure
    base = ax.get_xaxis_transform()
    tr, yy = (base + ScaledTranslation(0, offset_mm * MM, fig.dpi_scale_trans), 1.) if y is None else (base, y)
    ax.plot([x0, x1], [yy, yy], transform=tr, color=color, lw=lw, solid_capstyle='butt', clip_on=False)
    if label:
        ax.text((x0 + x1) / 2, yy, label, transform=tr + ScaledTranslation(0, .6 * MM, fig.dpi_scale_trans),
                ha='center', va='bottom', fontsize=FS['note'], color=color)


def side_bar(ax, y0, y1, label=None, color=NEUTRAL['band'], lw=LW['bar'], offset_mm=1.6):
    """Thin bar just right of the plotting area marking a y interval (e.g. the 4-12 Hz theta band)."""
    fig = ax.figure
    base = blended_transform_factory(ax.transAxes, ax.transData)
    ax.plot([1, 1], [y0, y1], transform=base + ScaledTranslation(offset_mm * MM, 0, fig.dpi_scale_trans),
            color=color, lw=lw, solid_capstyle='butt', clip_on=False)
    if label:
        ax.text(1, (y0 + y1) / 2, label, rotation=90, ha='left', va='center', fontsize=FS['note'], color=color,
                transform=base + ScaledTranslation((offset_mm + .8) * MM, 0, fig.dpi_scale_trans), clip_on=False)


def right_tag(ax, y, text, color=NEUTRAL['mid'], offset_mm=.9):
    """Small label just outside the right spine at data height y (e.g. |Delta| next to a guide line)."""
    fig = ax.figure
    tr = blended_transform_factory(ax.transAxes, ax.transData) + ScaledTranslation(offset_mm * MM, 0, fig.dpi_scale_trans)
    return ax.text(1, y, text, transform=tr, ha='left', va='center', fontsize=FS['note'], color=color, clip_on=False)


# State-shift conditions reuse the model encodings (activity-only = Full; +u = orange/triangle;
# +R = green/square; all states = grey/open circle, a control).
SHIFTS = {
    'activity_shift':   dict(MODELS['full'], label='Activity only'),
    'activity_u_shift': dict(MODELS['output'], label='Activity + $u$'),
    'activity_r_shift': dict(MODELS['release'], label='Activity + $R$'),
    'co_shift':         dict(MODELS['static'], label='All states'),
}
TRACES = {
    'activity': {'color': VARIABLES['activity'], 'ls': '-',  'lw': 3.0, 'alpha': .42, 'marker': 'o', 'mfc': None},
    'u':        {'color': VARIABLES['u'],        'ls': '--', 'lw': 1.3, 'alpha': 1.0, 'marker': '^', 'mfc': None},
    'resource': {'color': VARIABLES['resource'], 'ls': ':',  'lw': 1.3, 'alpha': 1.0, 'marker': 's', 'mfc': None},
    'drive':    {'color': VARIABLES['drive'],    'ls': '-.', 'lw': 1.3, 'alpha': 1.0, 'marker': 'D', 'mfc': 'none'},
}


def marker_key(ax, x, y, enc, text=None, size=None, transform=None, gap=.03, **text_kw):
    tr = transform if transform is not None else ax.transData
    ax.plot([x], [y], linestyle='none', marker=enc['marker'], markersize=size or MS['key'],
            markerfacecolor=marker_face(enc), markeredgecolor=enc['color'], markeredgewidth=.8,
            clip_on=False, transform=tr)
    if text:
        text_kw.setdefault('fontsize', FS['label'])
        text_kw.setdefault('color', NEUTRAL['ink'])
        ax.text(x + gap, y, text, transform=tr, ha='left', va='center', **text_kw)


def trial_offsets(n, jitter=.08):
    """Horizontal display offsets of n matched trials: fixed by trial order, identical in every column, never
    dependent on the values (the plotted y coordinates are the data themselves)."""
    return np.linspace(-jitter, jitter, n)


def paired_dots(ax, values, encs, labels, connect=(), jitter=.08, size=None, mean_width=.34, mean_halo=False):
    """Matched observations (rows = trials) for categorical conditions (columns); thin grey lines join the
    same trial only for the requested pairs; a short bar marks each column mean."""
    from matplotlib.collections import LineCollection
    size = MS['scatter'] if size is None else size
    vals = np.asarray(values, dtype=float)
    x = np.arange(vals.shape[1], dtype=float)
    off = trial_offsets(len(vals), jitter)
    for i, j in connect:
        segs = [np.array([[x[i] + o, row[i]], [x[j] + o, row[j]]]) for o, row in zip(off, vals)
                if np.isfinite(row[i]) and np.isfinite(row[j])]
        ax.add_collection(LineCollection(segs, colors=NEUTRAL['light'], linewidths=.6, zorder=1))
    for k, enc in enumerate(encs):
        open_ = enc.get('mfc') == 'none'
        ax.scatter(x[k] + off, vals[:, k], s=size, marker=enc['marker'], facecolors=marker_face(enc),
                   edgecolors=enc['color'] if open_ else 'white', linewidths=.8 if open_ else .4,
                   alpha=.92, zorder=3)
        mean = ax.plot([x[k] - mean_width / 2, x[k] + mean_width / 2], [np.nanmean(vals[:, k])] * 2,
                       color=enc['color'], lw=2.0, solid_capstyle='butt', zorder=5 if mean_halo else 4)[0]
        if mean_halo:       # white outline: the mean stays visible where it crosses points of its own colour
            mean.set_path_effects([patheffects.Stroke(linewidth=3.6, foreground='white'), patheffects.Normal()])
    ax.set_xticks(x, labels)
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-.5, len(x) - .5)


# CPL revision 6: one specification for the two main-text trial-level F_rest(2 s) panels, Fig. 3(d) and Fig. 4(a).
# Same y range and ticks (data: Fig. 3d -0.23..0.67, Fig. 4a -0.38..0.59; nothing clipped), same width per group,
# jitter, marker, mean bar and pairing lines.  Default paired_dots() calls (Supplementary figures) are unchanged.
FREST_AXIS = dict(ylim=(-.43, .75), yticks=[-.4, -.2, 0, .2, .4, .6])
FREST_PANEL = dict(jitter=.14, mean_width=.56, mean_halo=True)
FREST_GROUP_MM = 11.0


def frest_panel(ax, values, encs, labels, connect):
    """Trial-level F_rest(2 s) panel: zero line, matched trials (rows) in groups (columns), group means, the
    prespecified paired contrast only, one axis format.  Returns the trial offsets (for marking a trial)."""
    zero_line(ax)
    paired_dots(ax, values, encs, labels, connect=connect, **FREST_PANEL)
    ax.set(ylabel=r'$F_\mathrm{rest}$(2 s)', **FREST_AXIS)
    categorical(ax)
    g_ticks(ax)
    return trial_offsets(len(values), FREST_PANEL['jitter'])


def g_ticks(ax, which='y'):
    """Tick labels without trailing zeros (0, 0.25, 0.5 rather than 0.00, 0.25, 0.50)."""
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: f'{v:g}'.replace('-', '\u2212'))
    for w in which:
        (ax.xaxis if w == 'x' else ax.yaxis).set_major_formatter(fmt)


def place_labels(L, panels, top_mm=4.5, snap_x_mm=12., snap_y_mm=3., x_mm=None, y_mm=None, min_x_mm=1.2):
    """Legacy automatic placement (kept for old scripts; revised figures use explicit letter grids)."""
    from matplotlib.axes import Axes
    fig = L.fig
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    k = fig.dpi / 25.4
    pos = {}
    for letter, arts in panels.items():
        arts = list(arts) if isinstance(arts, (list, tuple)) else [arts]
        x0, top = [], []
        for a in arts:
            if isinstance(a, Axes):
                bb = a.get_tightbbox(r)
                x0.append(bb.x0 / k)
                top.append(a.get_position().y1 * L.h if a.axison else bb.y1 / k)
            else:
                x0.append(a.get_window_extent(r).x0 / k)
        dt = top_mm.get(letter, 4.5) if isinstance(top_mm, dict) else top_mm
        pos[letter] = [max(min(x0), min_x_mm), max(top) + dt - L.dy]
    for letter, (x, y) in pos.items():
        L.letter((x_mm or {}).get(letter, x), (y_mm or {}).get(letter, y), letter)
    return pos
