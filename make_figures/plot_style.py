"""Frozen manuscript-wide plotting defaults for the four main figures.

The final figures use a 180-mm manuscript width, restrained final-size typography,
color-blind-safe mechanism semantics, and vector-safe font export.  Figure-specific
scripts may still override geometry when the scientific panel structure requires it.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import visual_grammar as _vg

# CPL asks for 600-dpi figures; the same value sets the resolution of rasterized layers in PDF/SVG.
EXPORT_DPI = 600

# Frozen semantic palette shared with the supplementary figure package.
COLORS = {
    'full': '#0072B2',
    'activity': '#0072B2',
    'output': '#E69F00',
    'u': '#E69F00',
    'release': '#009E73',
    'resource': '#009E73',
    'control': '#7A7A7A',
    'static': '#7A7A7A',
}
REFERENCE = '#7A7A7A'


def configure():
    """Apply the frozen final-size manuscript typography and export settings."""
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 6.8,
        'axes.titlesize': 7.2,
        'axes.labelsize': 6.8,
        'xtick.labelsize': 6.1,
        'ytick.labelsize': 6.1,
        'legend.fontsize': 6.0,
        'legend.frameon': False,
        'axes.linewidth': .55,
        'lines.linewidth': .95,
        'lines.markersize': 3.2,
        'xtick.major.size': 2.2,
        'ytick.major.size': 2.2,
        'xtick.major.width': .55,
        'ytick.major.width': .55,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'savefig.dpi': EXPORT_DPI,
        'figure.dpi': 120,
        'axes.titlelocation': 'left',
        'mathtext.fontset': 'dejavusans',
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
    })


def figure(title=None, height_mm=180, subtitle=None):
    """Create a 180-mm-wide figure using the frozen manuscript defaults.

    ``title`` and ``subtitle`` remain accepted for compatibility with older code,
    but presentation-style figure-level titles/footers are intentionally not drawn;
    full figure titles belong in the manuscript caption.
    """
    configure()
    fig = plt.figure(figsize=(180/25.4, height_mm/25.4), layout='constrained')
    fig.set_constrained_layout_pads(w_pad=.035, h_pad=.045, wspace=.055, hspace=.055)
    return fig



# Panel-letter case for the target journal: 'lower' (Nature style, matches the manuscript v3)
# or 'upper' (Cell Press style). Change here only; every figure uses panel_letter().
PANEL_LETTER_CASE = 'lower'


def panel_letter(letter):
    """Return the panel label required by the target journal, e.g. "(a)" for CPL."""
    letter = letter.lower() if PANEL_LETTER_CASE == 'lower' else letter.upper()
    return _vg.panel_label(letter)

def panel(ax, letter, title):
    ax.set_title(title, pad=4, loc='left', fontsize=7.2)
    ax.annotate(panel_letter(letter), xy=(0, 1), xycoords='axes fraction', xytext=(-4, 3), textcoords='offset points', annotation_clip=False, ha='right', va='bottom',
            fontsize=8.4, fontweight='bold', clip_on=False)


def minimal(ax):
    ax.spines[['top', 'right']].set_visible(False)


def paired_plot(ax, pivot, labels=None, colors=None, connect_pairs=None):
    """Matched observations with deterministic offsets.

    By default no categorical polyline is drawn.  Pass ``connect_pairs=[(i,j), ...]``
    only when a specific paired contrast is scientifically meaningful.
    """
    vals = pivot.to_numpy(dtype=float)
    x = np.arange(vals.shape[1], dtype=float)
    offsets = np.linspace(-.065, .065, len(vals))
    connect_pairs = [] if connect_pairs is None else list(connect_pairs)
    for i, j in connect_pairs:
        segs = [np.array([[x[i]+off, row[i]], [x[j]+off, row[j]]])
                for off, row in zip(offsets, vals)]
        ax.add_collection(LineCollection(segs, colors='#9A9A9A', alpha=.30,
                                         linewidths=.55, zorder=1))
    if colors is None:
        colors = [REFERENCE] * vals.shape[1]
    for i, color in enumerate(colors):
        ax.scatter(x[i]+offsets, vals[:, i], s=10, alpha=.75, zorder=3,
                   facecolor=color, edgecolor='white', linewidth=.22)
        ax.plot(x[i], np.nanmean(vals[:, i]), linestyle='none', marker='_',
                markersize=13, markeredgewidth=1.55, color=color, zorder=4)
    ax.set_xticks(x, labels or list(pivot.columns))
    ax.set_xlim(-.35, len(x)-.65)
    ax.axhline(0, linewidth=.55, linestyle=':', color=REFERENCE, alpha=.65)
    ax.autoscale_view(scalex=False)


def forest(ax, labels, means, lows, highs, colors=None, primary=None):
    y = np.arange(len(labels))[::-1]
    primary = [False] * len(labels) if primary is None else list(primary)
    colors = [COLORS['full'] if p else '#555555' for p in primary] if colors is None else colors
    for i, (yy, m, l, h) in enumerate(zip(y, means, lows, highs)):
        color = colors[i]
        ax.hlines(yy, l, h, lw=1.2 if primary[i] else .9, color=color)
        ax.plot([l, h], [yy, yy], ls='none', marker='|', ms=4, color=color)
        ax.scatter(m, yy, s=22 if primary[i] else 15,
                   marker='D' if primary[i] else 'o', color=color, zorder=3)
    ax.axvline(0, linestyle=':', linewidth=.6, color=REFERENCE)
    ax.set_yticks(y, labels)
    ax.set_ylim(-.5, len(labels)-.5)
    ax.spines[['left', 'top', 'right']].set_visible(False)
    ax.tick_params(axis='y', length=0)


class Export:
    def __init__(self, path: Path, tiff=False):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.source = self.path/'source_data'
        self.source.mkdir(exist_ok=True)
        self.tiff = tiff
        self.records = []

    def csv(self, name, table):
        path = self.source/(name+'.csv')
        table.to_csv(path, index=False, float_format='%.10g')
        return path

    def npz(self, name, **arrays):
        np.savez_compressed(self.source/(name+'.npz'), **arrays)

    def save(self, fig, name, caption):
        # Fixed physical dimensions; avoid bbox_inches='tight' changing page widths.
        # v9: report (never raise) text that enters another panel's plotting area.
        try:
            from layout_check import check_all
            layout_warnings = check_all(fig)
        except Exception as exc:  # the self-check must never block a build
            layout_warnings = [f'layout check skipped: {exc}']
        for w in layout_warnings:
            print(f'LAYOUT CHECK {name}: {w}', flush=True)
        for ext in ['pdf', 'svg', 'png'] + (['tiff'] if self.tiff else []):
            kwargs = {'dpi': EXPORT_DPI}
            if ext == 'tiff':
                kwargs['pil_kwargs'] = {'compression': 'tiff_lzw'}
            if ext == 'pdf':
                kwargs['metadata'] = {'CreationDate': None, 'ModDate': None,
                                      'Creator': 'GC-STP figures'}
            if ext == 'svg':
                kwargs['metadata'] = {'Date': None}
            fig.savefig(self.path/f'{name}.{ext}', **kwargs)
        self.records.append({'figure': name, 'caption': caption,
                             'width_mm': round(float(fig.get_size_inches()[0]*25.4), 1),
                             'height_mm': float(fig.get_size_inches()[1]*25.4),
                             'layout_warnings': layout_warnings})
        plt.close(fig)
        print(f'Saved {name}', flush=True)
