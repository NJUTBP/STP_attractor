"""Plotting-layer self-check (v9): report text that enters another panel's plotting area.

Reported: axis labels, tick labels, legends and in-panel text of one panel, and figure-level
text/legends (panel letters, shared keys), whose bounding box overlaps the plotting area of a
different panel.  Overlaps inside a panel's own area are not reported.  The check never raises;
``Export.save`` prints each finding as ``LAYOUT CHECK <stem>: ...`` and stores the list in the
figure record (``layout_warnings``).  Identical copies live in make_figures/ and make_suppfigures/.
"""
from __future__ import annotations

from matplotlib.transforms import Bbox


def _shrink(bb, px=1.0):
    if bb.width > 2 * px and bb.height > 2 * px:
        return Bbox.from_extents(bb.x0 + px, bb.y0 + px, bb.x1 - px, bb.y1 - px)
    return bb


def _where(fig, ax):
    x0, y0, _, _ = ax.get_position().bounds
    w, h = fig.get_size_inches() * 25.4
    return f'({x0 * w:.0f}, {y0 * h:.0f}) mm'


def cross_axes_collisions(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    axes = [ax for ax in fig.axes if ax.get_visible()]
    targets = [ax for ax in axes if ax.axison]
    found = []

    def check(kind, art, owner, where):
        if not art.get_visible():
            return
        if kind != 'legend' and not art.get_text().strip():
            return
        bb = _shrink(art.get_window_extent(r))
        if bb.width <= 0 or bb.height <= 0:
            return
        label = 'legend' if kind == 'legend' else repr(art.get_text())
        for other in targets:
            if other is not owner and bb.overlaps(_shrink(other.bbox)):
                found.append(f'{kind} {label} of {where} enters the panel at {_where(fig, other)}')

    for ax in axes:
        where = f'the panel at {_where(fig, ax)}'
        if ax.axison:
            for t in ax.get_xticklabels() + ax.get_yticklabels():
                check('tick label', t, ax, where)
            check('axis label', ax.xaxis.label, ax, where)
            check('axis label', ax.yaxis.label, ax, where)
        if ax.get_legend() is not None:
            check('legend', ax.get_legend(), ax, where)
        for t in ax.texts:
            check('text', t, ax, where)
    for t in fig.texts:
        check('figure text', t, None, 'the figure')
    for leg in fig.legends:
        check('legend', leg, None, 'the figure')
    return found


# ---------------------------------------------------------------------------------------------
# CPL revision: text-on-text collisions.  The bracketed labels "(a)" are wider than the former single
# letters, so a label can meet a tick label or an axis label without entering another panel.
def _drawn_texts(fig, renderer):
    """(text, bbox) of every visible text that is actually drawn (hidden tick labels excluded, clipped
    texts cut to their clip box)."""
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox
    tick_texts, drawn = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            ticks = axis.get_major_ticks() + axis.get_minor_ticks()
            tick_texts.update(id(t) for tk in ticks for t in (tk.label1, tk.label2))
            if ax.get_visible() and ax.axison and axis.get_visible():
                drawn.update(id(t) for tk in axis._update_ticks() for t in (tk.label1, tk.label2))
    out = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if id(t) in tick_texts and id(t) not in drawn:
            continue
        if t.axes is not None and not t.axes.get_visible():
            continue
        bb = t.get_window_extent(renderer)
        clip = t.get_clip_box() if t.get_clip_on() else None
        if clip is not None:
            bb = Bbox.intersection(bb, clip)
            if bb is None:
                continue
        if bb.width > 0 and bb.height > 0:
            out.append((t, bb))
    return out


def text_collisions(fig, overlap_pt=0.6, label_gap_mm=0.5):
    """Report pairs of drawn texts that overlap by more than ``overlap_pt`` in both directions, and panel
    labels such as "(a)" that come closer than ``label_gap_mm`` to any other text."""
    import re
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    k = fig.dpi / 72.
    w_mm, h_mm = fig.get_size_inches() * 25.4
    items = _drawn_texts(fig, r)
    found = []

    def at(bb):
        return f'({bb.x0 / fig.bbox.width * w_mm:.0f}, {bb.y0 / fig.bbox.height * h_mm:.0f}) mm'

    for i, (t1, b1) in enumerate(items):
        for t2, b2 in items[i + 1:]:
            ox = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            oy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if ox > overlap_pt * k and oy > overlap_pt * k:
                found.append(f'text {t1.get_text()!r} overlaps text {t2.get_text()!r} at {at(b1)}')
    labels = [(t, b) for t, b in items if t in fig.texts and re.fullmatch(r'\(?[a-z]\)?', t.get_text().strip())]
    for t1, b1 in labels:
        for t2, b2 in items:
            if t2 is t1:
                continue
            gap = max(b2.x0 - b1.x1, b1.x0 - b2.x1, b2.y0 - b1.y1, b1.y0 - b2.y1) / k * 25.4 / 72.
            if gap < label_gap_mm:
                found.append(f'panel label {t1.get_text()!r} is {max(gap, 0.):.1f} mm from {t2.get_text()!r} at {at(b1)}')
    return found


def check_all(fig):
    """Cross-panel and text-on-text checks; each part reports its own failure instead of raising."""
    out = []
    for name, fn in (('cross-panel', cross_axes_collisions), ('text-collision', text_collisions)):
        try:
            out += fn(fig)
        except Exception as exc:  # the self-check must never block a build
            out.append(f'{name} check skipped: {exc}')
    return out
