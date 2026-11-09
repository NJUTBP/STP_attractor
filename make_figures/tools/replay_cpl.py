"""Re-render main Figs. 1-4 in Chinese Physics Letters style from their exported source data.

Usage (from make_figures/ or repository root):
    python make_figures/tools/replay_cpl.py --source make_figures/source_data --output output_main --figures 3 4
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

MODULES = {'1': 'fig1_model', '2': 'fig2_theta', '3': 'fig3_attractor', '4': 'fig4_synthesis'}


def replay(number, source, output, window, tiff):
    from plot_style import Export
    import visual_grammar as vg
    from captions_cpl import CAPTIONS
    mod = importlib.import_module(MODULES[number])
    d = mod.load_source_data(source, window)
    output.mkdir(parents=True, exist_ok=True)
    ex = Export(output, tiff)
    with vg.style():
        fig = mod.draw_figure(d)
        ex.save(fig, mod.STEM, CAPTIONS[mod.CAPTION_KEY])
    return ex.records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', type=Path, default=BASE / 'source_data',
                    help='Folder with exported source data (default: make_figures/source_data).')
    ap.add_argument('--output', type=Path, default=BASE / 'output_cpl', help='Output figure folder.')
    ap.add_argument('--figures', nargs='+', choices=sorted(MODULES), default=['3', '4'])
    ap.add_argument('--window', nargs=2, type=float, default=(5., 6.2), metavar=('START_S', 'END_S'))
    ap.add_argument('--tiff', action='store_true', help='Also write 600-dpi LZW TIFFs.')
    args = ap.parse_args(argv)
    source, output = args.source.resolve(), args.output.resolve()
    if output == source or source in output.parents:
        raise SystemExit('Write the figures to a new folder, not into the source-data folder.')
    for n in args.figures:
        replay(n, source, output, tuple(args.window), args.tiff)
        print(f'Fig. {n}: rendered.', flush=True)
    print(f'Done: {output}')


if __name__ == '__main__':
    main()
