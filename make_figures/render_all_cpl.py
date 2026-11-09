#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Official CPL graphics renderer: four Main figures (Figs. 1-4) and ten Supplementary figures (Figs. S1-S10).

All inputs are pre-computed source tables stored inside make_figures/source_data,
make_figures/source_data_fig1_noise90, make_suppfigures/source_data, and
make_suppfigures/bundled_data/phase16. No simulations are launched by this script.

Run from the repository root:
    python make_figures/render_all_cpl.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

N_SUPP = 10
FIGURE_NAMES = [f'Figure{i}_final' for i in range(1, 5)] + [f'FigS{i}_final' for i in range(1, N_SUPP + 1)]
ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / 'make_figures'
SI = ROOT / 'make_suppfigures'


def check_libraries():
    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import scipy  # noqa: F401
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f'Missing Python dependency: {exc}.\n'
            f'Run: "{sys.executable}" -m pip install -r "{ROOT / "requirements.txt"}"'
        ) from exc
    sys.path.insert(0, str(MAIN))
    import pdf_compose
    pdf_compose.require_backend()


def run(label: str, cmd: list[str], cwd: Path):
    print(f'\n========== {label} ==========', flush=True)
    print('Command: ' + ' '.join(f'"{x}"' if ' ' in x else x for x in cmd), flush=True)
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONIOENCODING='utf-8')
    process = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, encoding='utf-8', errors='replace', bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end='', flush=True)
    status = process.wait()
    if status != 0:
        raise RuntimeError(f'{label} FAILED with exit code {status}.')


def main(argv=None):
    default_fig1_src = MAIN / 'source_data_fig1_noise90'
    if not default_fig1_src.is_dir():
        default_fig1_src = MAIN / 'source_data'

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'CPL_Final_Figures',
                    help='Output folder (default: CPL_Final_Figures)')
    ap.add_argument('--fig1-source', type=Path, default=default_fig1_src,
                    help='Fig. 1 source_data folder (default: make_figures/source_data_fig1_noise90)')
    ap.add_argument('--fig3-source', type=Path, default=None,
                    help='Optional alternative Fig. 3 source_data folder')
    opts = ap.parse_args(argv)

    output = opts.output.expanduser().resolve()
    if output in (MAIN, SI, MAIN / 'source_data', SI / 'source_data') or MAIN in output.parents or SI in output.parents:
        raise RuntimeError('Output cannot be inside make_figures or make_suppfigures.')
    if output == ROOT:
        raise ValueError('Output must be a subfolder, not the repository root.')

    for path in [MAIN / 'assemble_fig1_vector.py', MAIN / 'assemble_fig2_vector.py',
                 MAIN / 'assets' / 'Fig1a.pdf', MAIN / 'assets' / 'Fig2a.pdf',
                 MAIN / 'tools' / 'replay_cpl.py', SI / 'tools' / 'replay_supp_cpl.py']:
        if not path.is_file():
            raise FileNotFoundError(f'Missing required file: {path}')

    check_libraries()
    src_m = MAIN / 'source_data'
    src_s = SI / 'source_data'
    fig1_source = opts.fig1_source.expanduser().resolve() if opts.fig1_source else src_m
    fig3_source = opts.fig3_source.expanduser().resolve() if opts.fig3_source else None

    with tempfile.TemporaryDirectory(prefix='cpl_plot_') as work:
        temp = Path(work)
        main_out = temp / 'main'
        si_out = temp / 'si'

        run('Main Fig. 1 (vector Fig1a)',
            [sys.executable, str(MAIN / 'assemble_fig1_vector.py'),
             '--schematic', str(MAIN / 'assets' / 'Fig1a.pdf'), '--source', str(fig1_source),
             '--output', str(main_out), '--layout', 'A'], MAIN)
        run('Main Fig. 2 (vector Fig2a; panels d-f = release-fraction mechanism)',
            [sys.executable, str(MAIN / 'assemble_fig2_vector.py'),
             '--schematic', str(MAIN / 'assets' / 'Fig2a.pdf'), '--source', str(src_m),
             '--mechanism-source', str(src_s),
             '--output', str(main_out), '--layout', 'balanced'], MAIN)
        if fig3_source is None:
            run('Main Fig. 3 and Fig. 4',
                [sys.executable, str(MAIN / 'tools' / 'replay_cpl.py'),
                 '--source', str(src_m), '--output', str(main_out), '--figures', '3', '4'], MAIN)
        else:
            run('Main Fig. 3',
                [sys.executable, str(MAIN / 'tools' / 'replay_cpl.py'),
                 '--source', str(fig3_source), '--output', str(main_out), '--figures', '3'], MAIN)
            run('Main Fig. 4',
                [sys.executable, str(MAIN / 'tools' / 'replay_cpl.py'),
                 '--source', str(src_m), '--output', str(main_out), '--figures', '4'], MAIN)
        run(f'Supplementary Fig. S1–S{N_SUPP}',
            [sys.executable, str(SI / 'tools' / 'replay_supp_cpl.py'),
             '--source', str(src_s), '--output', str(si_out), '--figures',
             *[str(i) for i in range(1, N_SUPP + 1)], '--formats', 'pdf', 'png'], SI)

        stem_lookup = {
            'Figure1_final': (main_out, 'Figure1_final'),
            'Figure2_final': (main_out, 'Figure2_final'),
            'Figure3_final': (main_out, 'Fig3_Attractor_state_surgery'),
            'Figure4_final': (main_out, 'Fig4_Output_route_on_rewritten_reference'),
            'FigS1_final': (si_out, 'FigS1_oscillation_diagnostics'),
            'FigS2_final': (si_out, 'FigS2_frozen_profile_control'),
            'FigS3_final': (si_out, 'FigS3_route_decomposition_frequency'),
            'FigS4_final': (si_out, 'FigS4_parameter_sensitivity'),
            'FigS5_final': (si_out, 'FigS5_confirmation_set'),
            'FigS6_final': (si_out, 'FigS6_clamp_field'),
            'FigS7_final': (si_out, 'FigS7_profile_snapshot'),
            'FigS8_final': (si_out, 'FigS8_relative_coordinate_noise'),
            'FigS9_final': (si_out, 'FigS9_noise_robustness'),
            'FigS10_final': (si_out, 'FigS10_efficacy_statistics'),
        }

        output.mkdir(parents=True, exist_ok=True)
        for output_stem in FIGURE_NAMES:
            folder, source_stem = stem_lookup[output_stem]
            for ext in ('.pdf', '.png'):
                src = folder / f'{source_stem}{ext}'
                if not src.is_file() or src.stat().st_size < 1000:
                    raise RuntimeError(f'Missing or empty rendered figure: {src}')
                dst = output / f'{output_stem}{ext}'
                shutil.copy2(src, dst)

        print('\n=================================================', flush=True)
        print(f'SUCCESS: {len(FIGURE_NAMES)} CPL figures ({len(FIGURE_NAMES)} PDF + {len(FIGURE_NAMES)} PNG) saved in {output}', flush=True)
        print('=================================================', flush=True)


if __name__ == '__main__':
    main()
