"""Build Supplementary Figs. S4 and S10 from the local results tree and export their source tables.

Usage (from make_suppfigures/):
    python tools/freeze_revision_sources.py --results-root ../results
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

REVISION_MODULES = {
    4: 'supp_rev_s9_parameter_sensitivity',
    10: 'supp_rev_s10_efficacy_statistics',
}


def main(argv=None) -> int:
    from settings import RESULTS_ROOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-root', type=Path, default=RESULTS_ROOT,
                    help='Folder that contains revision_r1/ (default: settings.RESULTS_ROOT)')
    ap.add_argument('--figures', nargs='+', type=int, choices=sorted(REVISION_MODULES), default=sorted(REVISION_MODULES))
    ap.add_argument('--output', type=Path, default=None, help='Optional folder to also render the figures into')
    a = ap.parse_args(argv)

    from data_io import DataStore
    source = BASE / 'source_data'
    root = a.results_root.expanduser().resolve()
    if not (root / 'revision_r1').is_dir():
        raise SystemExit(f'No revision_r1/ under {root}. Point --results-root to the results folder.')
    for n in a.figures:
        mod = importlib.import_module(REVISION_MODULES[n])
        ds = DataStore(root=root)
        d = mod.compute_panels(ds)
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            class SimpleExport:
                def __init__(self, path):
                    self.path = Path(path)
                    self.files = []
                def csv(self, name, df):
                    p = self.path / f'{name}.csv'
                    df.to_csv(p, index=False, float_format='%.12g')
                    self.files.append(p)
                    return p
            cap = SimpleExport(tmp)
            mod.export_source_data(cap, d)
            for p in cap.files:
                shutil.copy2(p, source / p.name)
            files = sorted(p.name for p in cap.files)
        print(f'Fig. S{n} ({mod.STEM}): exported {", ".join(files)}', flush=True)
        if a.output is not None:
            import visual_grammar as vg
            from plot_style import Export
            ex = Export(a.output.expanduser().resolve(), ds, ('pdf', 'png'))
            mod.export_source_data(ex, d)
            with vg.style():
                fig = mod.draw_figure(d)
                ex.save(fig, mod.STEM, mod.caption_for(d) if hasattr(mod, 'caption_for') else mod.CAPTION)
    return 0


if __name__ == '__main__':
    sys.exit(main())
