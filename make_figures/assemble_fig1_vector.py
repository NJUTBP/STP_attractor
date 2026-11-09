"""Assemble CPL Fig. 1 with the author's standalone ring schematic (assets/Fig1a.pdf) as native vector content.

CPL revision 2: panel (a) is placed in fig1_model.PANEL_A_RECT_MM (same printed size as before, moved left and
up), and PDF composition goes through pdf_compose (PyMuPDF when installed, otherwise pikepdf).  The data layer
of fig1_model.py is not modified; the schematic PDF itself is never edited.

    python assemble_fig1_vector.py --schematic assets/Fig1a.pdf --source source_data --output OUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import fig1_model as fig1
import pdf_compose as pc
import visual_grammar as vg
from layout_check import check_all

FIG_W_MM, FIG_H_MM = vg.FIG_WIDTH_MM, fig1.FIG_H_MM
DPI = 600
DEFAULT_SCHEMATIC_SCALE = 1.0      # relative to PANEL_A_RECT_MM
DEFAULT_SCHEMATIC_SHIFT_X_MM = 0.0
DEFAULT_SCHEMATIC_SHIFT_Y_MM = 0.0
# Point diameter relative to the original; adjust each raster independently.
DEFAULT_RASTER_MARKER_SCALE_B = 0.4         # panel (b)
DEFAULT_RASTER_MARKER_SCALE_C_TOP = 0.3     # panel (c), upper E raster
DEFAULT_RASTER_MARKER_SCALE_C_BOTTOM = 1.0  # panel (c), lower I raster


def insert_rect_mm(scale=DEFAULT_SCHEMATIC_SCALE, shift_x=DEFAULT_SCHEMATIC_SHIFT_X_MM,
                   shift_y=DEFAULT_SCHEMATIC_SHIFT_Y_MM):
    if not (0.0 < scale <= 1.0):
        raise ValueError('Schematic scale must be greater than 0 and at most 1.')
    x, y, w, h = fig1.PANEL_A_RECT_MM
    cx, cy = x + w / 2 + shift_x, y + h / 2 + shift_y
    return (cx - w * scale / 2, cy - h * scale / 2, w * scale, h * scale)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scale_raster_markers(figure, scale_b, scale_c_top, scale_c_bottom):
    scales = (scale_b, scale_c_top, scale_c_bottom)
    if any(not (0.0 < scale <= 1.0) for scale in scales):
        raise ValueError('Each raster marker scale must be greater than 0 and at most 1.')
    # fig1_model.draw_figure creates its three scatter axes in order: b, c E, c I.
    rasters = [artist for ax in figure.axes for artist in ax.collections
               if isinstance(artist, PathCollection)]
    if len(rasters) != 3:
        raise RuntimeError(f'Expected three raster scatter plots (b, c E, c I), found {len(rasters)}.')
    for artist, scale in zip(rasters, scales):
        # scatter sizes are areas in pt^2; scale outlines as well.
        artist.set_sizes(artist.get_sizes() * scale ** 2)
        artist.set_linewidths(artist.get_linewidths() * scale)


def run(panel_file: Path, source: Path, output: Path, layout: str = 'A', schematic_scale=DEFAULT_SCHEMATIC_SCALE,
        shift_x=DEFAULT_SCHEMATIC_SHIFT_X_MM, shift_y=DEFAULT_SCHEMATIC_SHIFT_Y_MM,
        raster_marker_scale_b=DEFAULT_RASTER_MARKER_SCALE_B,
        raster_marker_scale_c_top=DEFAULT_RASTER_MARKER_SCALE_C_TOP,
        raster_marker_scale_c_bottom=DEFAULT_RASTER_MARKER_SCALE_C_BOTTOM):
    if not panel_file.is_file():
        raise FileNotFoundError(f'Fig. 1(a) vector PDF not supplied: {panel_file}')
    if panel_file.suffix.lower() != '.pdf':
        raise ValueError('For native vector insertion, supply Fig. 1(a) as a PDF.')
    engine = pc.require_backend()
    output.mkdir(parents=True, exist_ok=True)
    d = fig1.load_source_data(source)
    source_hashes = {name: sha256(source / name) for name in fig1.SOURCE_FILES}
    metadata_path = source/'Fig1_representative_trial.json'
    if metadata_path.is_file():
        source_hashes[metadata_path.name] = sha256(metadata_path)
    rect = insert_rect_mm(schematic_scale, shift_x, shift_y)
    x, y, w, h = rect
    if x < 0 or y < 0 or x + w > FIG_W_MM or y + h > FIG_H_MM:
        raise RuntimeError(f'Fig. 1(a) rectangle {rect} lies outside the canvas.')
    if pc.page_info(panel_file)[2] != 1:
        raise ValueError('The schematic PDF must have exactly one page.')
    pdf_final, png_final = output / 'Figure1_final.pdf', output / 'Figure1_final.png'
    with tempfile.TemporaryDirectory() as tmp:
        raw_fig = Path(tmp) / 'Fig1_awaiting_a.pdf'
        with vg.style():
            figure = fig1.draw_figure(d, layout_variant=layout, schematic='external')
            scale_raster_markers(figure, raster_marker_scale_b, raster_marker_scale_c_top,
                                 raster_marker_scale_c_bottom)
            warnings = check_all(figure)
            if warnings:
                raise RuntimeError('Layout collision detected: ' + '; '.join(warnings))
            figure.savefig(raw_fig, dpi=DPI, metadata={'CreationDate': None, 'ModDate': None,
                                                       'Creator': 'CPL Fig1 vector compositor'})
            plt.close(figure)
        pc.overlay(raw_fig, panel_file, rect, pdf_final)
    pc.to_png(pdf_final, png_final, DPI)
    W, H, n = pc.page_info(pdf_final)
    if n != 1 or abs(W - FIG_W_MM * pc.MM_PT) > .5 or abs(H - FIG_H_MM * pc.MM_PT) > .5:
        raise RuntimeError('Unexpected final PDF page size or page count')
    print('Assembled vector PDF:', pdf_final, '| backend:', engine)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--schematic', type=Path, required=True)
    p.add_argument('--source', type=Path, default=Path(__file__).resolve().parent / 'source_data')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layout', choices=('A',), default='A')
    p.add_argument('--schematic-scale', type=float, default=DEFAULT_SCHEMATIC_SCALE)
    p.add_argument('--schematic-shift-x-mm', type=float, default=DEFAULT_SCHEMATIC_SHIFT_X_MM,
                   help='extra horizontal shift of Fig. 1(a) in mm (negative = left)')
    p.add_argument('--schematic-shift-y-mm', type=float, default=DEFAULT_SCHEMATIC_SHIFT_Y_MM,
                   help='extra vertical shift of Fig. 1(a) in mm (positive = up)')
    p.add_argument('--raster-marker-scale-b', type=float, default=DEFAULT_RASTER_MARKER_SCALE_B,
                   help='point diameter multiplier for Fig. 1(b), from 0 (exclusive) to 1')
    p.add_argument('--raster-marker-scale-c-top', type=float, default=DEFAULT_RASTER_MARKER_SCALE_C_TOP,
                   help='point diameter multiplier for Fig. 1(c) upper E raster, from 0 (exclusive) to 1')
    p.add_argument('--raster-marker-scale-c-bottom', type=float, default=DEFAULT_RASTER_MARKER_SCALE_C_BOTTOM,
                   help='point diameter multiplier for Fig. 1(c) lower I raster, from 0 (exclusive) to 1')
    o = p.parse_args()
    run(o.schematic.resolve(), o.source.resolve(), o.output.resolve(), o.layout, o.schematic_scale,
        o.schematic_shift_x_mm, o.schematic_shift_y_mm, o.raster_marker_scale_b,
        o.raster_marker_scale_c_top, o.raster_marker_scale_c_bottom)


if __name__ == '__main__':
    main()
