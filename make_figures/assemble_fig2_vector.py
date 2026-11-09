"""Assemble vector CPL Fig. 2 from the author's native Fig2a.pdf and the unchanged data panels.

CPL revision 2: the wide author schematic (1975 x 905 pt) is placed in FIG2_LAYOUTS['short']['a_box'] of
fig2_theta (aspect ratio kept, centred, vector), PDF composition goes through pdf_compose (PyMuPDF when
installed, otherwise pikepdf).  The author's PDF is never edited.  Legacy tall artwork still triggers the old
typography repair (fix_math_type, PyMuPDF only), but its tall layouts are no longer drawn.

    python assemble_fig2_vector.py --schematic assets/Fig2a.pdf --source source_data --output OUT

CPL revision 3: the wide schematic now resolves to the 'mechanism' layout (145 mm), whose bottom row (d)-(f) is the
former Supplementary Fig. S11, read from the frozen RevS11* tables (--mechanism-source, default
../make_suppfigures/source_data; SHA256 and recomputation checks in fig2_mechanism.py).  '--layout legacy-short'
redraws the 85-mm revision-2 figure.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import fig2_theta as mod
import pdf_compose as pc
import visual_grammar as vg

FIG_W_MM = vg.FIG_WIDTH_MM
FIG_HEIGHTS_MM = {name: cfg['height_mm'] for name, cfg in mod.FIG2_LAYOUTS.items()}
DEFAULT_SCHEMATIC_SCALE = 0.9
DEFAULT_SCHEMATIC_SHIFT_X_MM = 0.0
DEFAULT_SCHEMATIC_SHIFT_Y_MM = 0.0
LEGACY_GEOMETRY_PT = (1939.0, 2026.0)
AUTO_SHORT_ASPECT_RATIO = 1.35
DPI = 600


def fix_math_type(source_pdf:Path, fixed_pdf:Path):
    """Correct three typesetting slips in the supplied artwork without touching its source.

    Illustrator put the '+' below 'u' in both u^+ rows and omitted the bar on
    fixed static efficacy p. Reposition original vector '+' glyphs (no font
    substitution) and add a 6-point overline. The schematic topology, rows,
    values, conditions, vectors and colors are otherwise exact PDF contents.
    """
    import pymupdf as fitz  # legacy artwork only
    with fitz.open(source_pdf) as changed, fitz.open(source_pdf) as original:
        pg=changed[0]
        # Masks hit only the '+' glyph bounding boxes, never the adjacent u.
        for dy in (0., 412.):
            pg.add_redact_annot(fitz.Rect(1652., 1318.+dy, 1687., 1360.+dy),
                                fill=(1.,1.,1.))
        pg.apply_redactions(images=0,graphics=0,text=0)
        for dy in (0., 412.):
            crop=fitz.Rect(1650.,1307.+dy,1696.,1375.+dy)
            target=fitz.Rect(1650.,1237.+dy,1696.,1305.+dy)
            pg.show_pdf_page(target,original,0,clip=crop,keep_proportion=False,overlay=True)
        # Italic p already exists; the overbar is the only addition.
        pg.draw_line(fitz.Point(989.,1899.),fitz.Point(1043.,1899.),
                     color=(0.,0.,0.),width=6.0,overlay=True)
        # Harmonize ONLY model-identity markers with the manuscript's fixed
        # condition palette. The author's Illustrator PDF uses a different
        # shade for Full and Output-only; release green already matches.
        from matplotlib.colors import to_rgb
        pg.draw_circle(fitz.Point(161.,1286.6),34.8,
                       fill=to_rgb(vg.MODELS['full']['color']),color=None,overlay=True)
        pg.draw_polyline([fitz.Point(161.,1455.2),fitz.Point(195.,1510.2),
                          fitz.Point(127.,1510.2)],closePath=True,
                         fill=to_rgb(vg.MODELS['output']['color']),color=None,overlay=True)
        pg.draw_circle(fitz.Point(160.5,1905.7),27.6,color=to_rgb(vg.MODELS['static']['color']),
                       fill=None,width=10.5,overlay=True)
        changed.save(fixed_pdf,garbage=4,deflate=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def resolve_layout(requested, ratio):
    if requested == 'legacy-short':
        return 'short'
    if requested in ('auto', 'balanced', 'short', 'mechanism'):
        return 'mechanism' if ratio >= AUTO_SHORT_ASPECT_RATIO else 'balanced'
    raise ValueError('Unknown Fig2 layout: ' + str(requested))


def insert_rect_mm(layout, scale, shift_x, shift_y):
    if not (0.0 < scale <= 1.0):
        raise ValueError('Fig2a --schematic-scale must be > 0 and <= 1.0.')
    x, y, w, h = mod.FIG2_LAYOUTS[layout]['a_box']
    cx, cy = x + w / 2 + shift_x, y + h / 2 + shift_y
    return (cx - w * scale / 2, cy - h * scale / 2, w * scale, h * scale)


def run(schematic, source, output, layout='balanced', dpi=DPI, schematic_scale=DEFAULT_SCHEMATIC_SCALE,
        schematic_shift_x_mm=DEFAULT_SCHEMATIC_SHIFT_X_MM, schematic_shift_y_mm=DEFAULT_SCHEMATIC_SHIFT_Y_MM,
        mechanism_source=None):
    schematic, source, output = Path(schematic).resolve(), Path(source).resolve(), Path(output).resolve()
    if not schematic.is_file():
        raise FileNotFoundError(schematic)
    if schematic.suffix.lower() != '.pdf':
        raise ValueError('Fig2a must be a vector PDF')
    engine = pc.require_backend()
    w_pt, h_pt, n = pc.page_info(schematic)
    if n != 1:
        raise ValueError('Fig2a PDF must contain exactly one page.')
    requested, layout = layout, resolve_layout(layout, w_pt / h_pt)
    if layout not in ('mechanism', 'short'):
        raise RuntimeError('Fig. 2 is drawn only with the wide author Fig2a.pdf (mechanism layout; legacy-short on request).')
    output.mkdir(parents=True, exist_ok=True)
    d = mod.load_source_data(source, mech_source=(mechanism_source or 'auto') if layout == 'mechanism' else None)
    hashes = {n_: sha(source / n_) for n_ in mod.SOURCE_FILES}
    rect = insert_rect_mm(layout, schematic_scale, schematic_shift_x_mm, schematic_shift_y_mm)
    H = FIG_HEIGHTS_MM[layout]
    if rect[0] < 0 or rect[1] < 0 or rect[0] + rect[2] > FIG_W_MM or rect[1] + rect[3] > H:
        raise RuntimeError('Fig2a schematic box extends outside the Figure2 canvas')
    out, png = output / 'Figure2_final.pdf', output / 'Figure2_final.png'
    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp) / 'fig2_without_a.pdf'
        with vg.style():
            fig = mod.draw_figure(d, layout_variant=layout)
            fig.savefig(base, dpi=dpi, metadata={'CreationDate': None, 'ModDate': None})
            plt.close(fig)
        pc.overlay(base, schematic, rect, out)
    pc.to_png(out, png, dpi)
    W, Hp, _ = pc.page_info(out)
    if abs(W - FIG_W_MM * pc.MM_PT) > .06 or abs(Hp - H * pc.MM_PT) > .06:
        raise RuntimeError('Figure2 PDF size mismatch')
    outside = pc.text_outside(out)
    print('Rendered', out, '| backend:', engine)
    if outside:
        raise RuntimeError('Figure2 PDF text outside canvas ' + str(outside))
    return out, png


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--schematic', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--layout', choices=['auto', 'balanced', 'short', 'mechanism', 'legacy-short'], default='balanced')
    ap.add_argument('--mechanism-source', type=Path, default=None,
                    help='folder with the frozen RevS11* tables of panels (d)-(f); default ../make_suppfigures/source_data')
    ap.add_argument('--schematic-scale', type=float, default=DEFAULT_SCHEMATIC_SCALE)
    ap.add_argument('--schematic-shift-x-mm', type=float, default=DEFAULT_SCHEMATIC_SHIFT_X_MM)
    ap.add_argument('--schematic-shift-y-mm', type=float, default=DEFAULT_SCHEMATIC_SHIFT_Y_MM)
    a = ap.parse_args()
    run(a.schematic, a.source, a.output, a.layout, schematic_scale=a.schematic_scale,
        schematic_shift_x_mm=a.schematic_shift_x_mm, schematic_shift_y_mm=a.schematic_shift_y_mm,
        mechanism_source=a.mechanism_source)


if __name__ == '__main__':
    main()
