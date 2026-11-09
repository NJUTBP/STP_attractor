"""Re-render Supplementary Figs. S1-S10 in Chinese Physics Letters style from archived source tables.

Usage (from make_suppfigures/ or repository root):
    python make_suppfigures/tools/replay_supp_cpl.py --source make_suppfigures/source_data --output output_supp
    python make_suppfigures/tools/replay_supp_cpl.py --figures 1 2 3 4 --formats pdf png
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from supp_captions_cpl import SUPP_CAPTIONS

# Latest manuscript supplementary figure numbering (Figs. S1-S10).
# Historical source-data filenames (FigS2*, FigS3*, FigS6*, RevS9*, FigS12*, FigS14*, FigS15*, FigS16*, FigS9*, RevS10*)
# are preserved so underlying data tables remain untouched.
MODULES = {
    1: 'supp_s2_phase6',
    2: 'supp_s3_phase7',
    3: 'supp_s6_phase10',
    4: 'supp_rev_s9_parameter_sensitivity',
    5: 'supp_s12_phase15',
    6: 'supp_s14_phase16A',
    7: 'supp_s15_phase16B',
    8: 'supp_s16_phase16C',
    9: 'supp_s9_phase13',
    10: 'supp_rev_s10_efficacy_statistics',
}
PHASE16 = {6, 7, 8}
REVISION = {4, 10}

# Archived files read by each replayed figure: {figure_number: {file: key in the drawing dictionary}}.
READ = {
    1: {'FigS2_theta.csv': 'theta', 'FigS2_oscillation_diagnostics.csv': 'diag',
        'FigS2C_current.csv': 'current', 'FigS2D_PSD.csv': 'psd'},
    2: {'FigS3_stationarity.csv': 'stat', 'FigS3_theta.csv': 'theta', 'FigS3_profile.csv': 'profile',
        'FigS3D_PSD_by_seed.csv': 'spectra'},
    3: {'FigS6_phase10_combo_summary.csv': 'summary', 'FigS6_gain_sensitivity_descriptive.csv': 'gain_sensitivity',
        'FigS6K_error_ecdf.csv': 'ecdf', 'FigS4_phase8_full_theta_fraction.csv': 'theta8'},
    5: {'FigS12_seed_metrics.csv': 'seed', 'FigS12_contrasts.csv': 'contrasts'},
    9: {'FigS9_seed_noise_response.csv': 'seed', 'FigS9_noise_level_summary.csv': 'noise',
        'FigS9_anchor_integrated_seed.csv': 'burdens', 'FigS9C_anchor_forest.csv': 'forest'},
}
CARRIED = {
    1: ['FigS2A_spikes.csv', 'FigS2B_spikes.csv'],
    5: ['FigS12B_paired_routes.csv', 'FigS12C_paired_routes.csv',
        'FigS12D_paired_surgery.csv', 'FigS12E_paired_surgery.csv'],
}


class SourceStore:
    """Lightweight provenance placeholder for Export compatibility."""

    def __init__(self):
        self.records = {}

    def record(self, path, identity=''):
        path = Path(path)
        self.records[path.name] = {'path': f'source_data/{path.name}', 'identity': identity}


def replay_from_source(n, mod, source, store):
    d = {}
    for name, key in READ[n].items():
        path = source / name
        if not path.is_file():
            raise SystemExit(f'S{n}: missing {path}.')
        d[key] = pd.read_csv(path)
        store.record(path, 'archived source data')
    if n == 9:
        seed = d['seed']
        d['anchor'] = seed[(seed.combo_id == 'tf_1000_tr_20') & (seed['mode'] == 'dynamic')]
    files = list(READ[n]) + CARRIED.get(n, [])
    return d, files, 'drawn from archived source data'


def replay_revision(n, mod, source, store):
    d, files = {}, []
    for name, key in mod.REPLAY_READ.items():
        path = source / name
        if not path.is_file():
            raise SystemExit(f'S{n}: missing {path}.')
        d[key] = pd.read_csv(path)
        store.record(path, 'revision source data')
        files.append(name)
    optional = getattr(mod, 'REPLAY_OPTIONAL', {})
    present = [name for name in optional if (source / name).is_file()]
    for name, key in optional.items():
        if name in present:
            d[key] = pd.read_csv(source / name)
            store.record(source / name, 'revision source data')
            files.append(name)
        else:
            d[key] = None
    return d, files, 'drawn from revision source data'


def rebuild_phase16(n, mod, source, tmp, store):
    import phase16_data
    from data_io import DataStore
    ds = DataStore(BASE / 'bundled_data')
    phase16_data.attach(ds, root=None, verify=False)
    d = mod.compute_panels(ds)
    files = []
    for p in sorted(source.glob(f'{mod.LEGACY_PREFIX}*')):
        if p.is_file():
            files.append(p.name)
    store.records = dict(ds.records)
    return d, files, 'rebuilt from bundled_data/phase16'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', type=Path, default=BASE / 'source_data')
    ap.add_argument('--output', type=Path, default=BASE / 'output_cpl')
    ap.add_argument('--figures', nargs='+', type=int, choices=sorted(MODULES), default=sorted(MODULES))
    ap.add_argument('--formats', nargs='+', choices=['pdf', 'svg', 'png', 'tiff'], default=['pdf', 'png'])
    a = ap.parse_args(argv)
    source, output = a.source.resolve(), a.output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise SystemExit('Write the figures to a new folder, not into (or above) the source-data folder.')
    import visual_grammar as vg
    from plot_style import Export
    for n in a.figures:
        mod = importlib.import_module(MODULES[n])
        store = SourceStore()
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            if n in PHASE16:
                d, files, method = rebuild_phase16(n, mod, source, tmp, store)
            elif n in REVISION:
                d, files, method = replay_revision(n, mod, source, store)
            else:
                d, files, method = replay_from_source(n, mod, source, store)
            output.mkdir(parents=True, exist_ok=True)
            ex = Export(output, store, a.formats)
            ex.assets = list(files)
            with vg.style():
                out = mod.draw_figure(d)
                fig = out[0] if isinstance(out, tuple) else out
                ex.save(fig, mod.STEM, mod.caption_for(d) if hasattr(mod, 'caption_for') else mod.CAPTION)
        print(f'Fig. S{n} ({mod.STEM}): {method}.', flush=True)
    print(f'Done: {output}')


if __name__ == '__main__':
    main()
