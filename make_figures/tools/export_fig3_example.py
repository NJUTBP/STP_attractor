#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export the source data of Fig. 3(a)-(c) (CPL revision 5) for one confirmation trial.  Nothing is simulated.

Inputs
  * archived tables (always): Fig3D_confirmation_paired.csv (direction-averaged F_rest(2 s) of the confirmation set)
    and the bundled profile-snapshot table B_kinematics.csv (or --phase16-root);
  * the module-B profile-snapshot branches of the chosen trial from the owner's local results tree (--raw-root):
        tf_1000_tr_20__dynamic__seedNNNN__no_shift.npz
        tf_1000_tr_20__dynamic__seedNNNN__{activity_shift,activity_u_shift,activity_r_shift}__{p,m}11p25.npz
    in <raw-root>/phase16/module_B/branches, <raw-root>/module_B/branches, <raw-root>/branches or <raw-root>.

Trial choice (unless --seed): the archived rule of revision 4 (activity-only F_rest(2 s) closest to the
confirmation-set mean) is printed, but the trial is chosen jointly over the three conditions drawn in (a)-(c):
minimum J = sum_k z_k^2, z_k = standardized F_rest(2 s) of condition k (activity only, + u, + R).  The chosen trial
must also pass, without tuning: |z_k| <= 1 for every k; activity-only restoration time t90 (90% of the change of phi_s
reached by 2 s) inside the interquartile range of the 16 trials; RMS distance of its activity-only phi_s trajectory to
the 16-trial mean no larger than the median.  The full table is written to Fig3_example_selection.csv.

Checks before anything is written: the direction-averaged trajectories recomputed from the raw snapshots must equal the
archived B_kinematics rows of that trial (max |diff| < 1e-6 deg); Fig. 3(d) tables are copied byte for byte.

Usage (from make_figures/):
    python tools/export_fig3_example.py --raw-root "../results" --output source_data_fig3_r5
    python tools/export_fig3_example.py --raw-root "../results" --output source_data_fig3_r5 --install
``--install`` then adds the new Fig. 3 files to source_data/ (existing files are never overwritten).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import fig3_attractor as f3  # noqa: E402
import phase16_data as P16  # noqa: E402
from data_io import DataStore  # noqa: E402
from dynamics import axial_delta, pav  # noqa: E402

DELTA = f3.DELTA
PANEL_D_FILES = ['Fig3D_confirmation_paired.csv', 'Fig3D_archived_contrasts.csv']
NEW_FILES = ['Fig3ABC_example_profiles.csv', 'Fig3ABC_example_state_extents.csv', 'Fig3ABC_example_kinematics.csv', 'Fig3_example_trial.json',
             'Fig3_example_selection.csv']


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archived_kinematics(phase16_root=None) -> pd.DataFrame:
    store = P16.Phase16Store(phase16_root)
    k = store.table('B', 'kinematics')
    return k[(k.condition == 'dynamic') & np.isclose(k.abs_delta, DELTA)].copy(), store.path('B', 'kinematics')


def selection_table(paired: pd.DataFrame, kin: pd.DataFrame) -> pd.DataFrame:
    p = paired.set_index('seed')[f3.KINDS].astype(float)
    z = (p - p.mean()) / p.std(ddof=1)
    out = pd.DataFrame({'seed': p.index})
    out = out.set_index('seed')
    for kind, short in zip(f3.KINDS, ('A', 'U', 'R')):
        out[f'F_rest_2s_{short}'] = p[kind]
        out[f'z_{short}'] = z[kind]
    out['J'] = (z ** 2).sum(axis=1)
    out['revision4_rule_abs_dev_A'] = (p['activity_shift'] - p['activity_shift'].mean()).abs()
    for kind, short in zip(f3.KINDS, ('A', 'U', 'R')):
        g = kin[(kin.kind == kind) & (kin.t_ms <= 2000)].pivot(index='seed', columns='t_ms', values='x')
        out[f'traj_rms_{short}_deg'] = np.sqrt(((g - g.mean(axis=0)) ** 2).mean(axis=1))
    g = kin[(kin.kind == 'activity_shift') & (kin.t_ms <= 2000)].pivot(index='seed', columns='t_ms', values='x')
    t = g.columns.to_numpy(float)

    def t90(x):
        change = DELTA - x[-1]
        return float(t[np.argmax(DELTA - x >= .9 * change)]) if change > 0 else np.nan
    out['t90_A_ms'] = [t90(g.loc[s].to_numpy()) for s in out.index]
    q25, q75 = np.nanpercentile(out.t90_A_ms, [25, 75])
    out['passes_checks'] = ((out[['z_A', 'z_U', 'z_R']].abs() <= 1).all(axis=1)
                            & out.t90_A_ms.between(q25, q75)
                            & (out.traj_rms_A_deg <= out.traj_rms_A_deg.median()))
    return out.sort_values('J')


def branch_f_rest(ex: DataStore, kind: str, t_after_ms: float = 2000.) -> dict:
    """Single-branch F_rest(2 s) (Eq. S10; trailing 400-ms population vector) from the snapshot-set spike trains."""
    def spikes(raw):
        return {'spike_times_e_ms': raw['spike_t'].astype(float), 'spike_angles_e_deg': raw['spike_a'].astype(float)}
    t = np.array([t_after_ms])
    ref = pav(spikes(ex.snapshot_branch('no_shift')), t)[0]
    out = {}
    for sign, tag in ((1., 'p'), (-1., 'm')):
        phi = pav(spikes(ex.snapshot_branch(kind, tag, '11p25')), t)[0]
        out[tag] = float(1. - sign * axial_delta(phi, ref) / DELTA)
    out['directions_averaged'] = .5 * (out['p'] + out['m'])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw-root', type=Path, required=True, help='Folder with the module-B snapshot branches.')
    ap.add_argument('--seed', type=int, default=None, help='Trial to show; default: rule described above.')
    ap.add_argument('--panel-d-source', type=Path, default=BASE / 'source_data')
    ap.add_argument('--phase16-root', type=Path, default=None, help='Default: bundled_data/phase16.')
    ap.add_argument('--output', type=Path, required=True, help='New source-data folder for Fig. 3.')
    ap.add_argument('--install', action='store_true', help='Also add the new files to source_data/ (no overwrite).')
    args = ap.parse_args(argv)

    out = (args.output if args.output.is_absolute() else BASE / args.output).resolve()
    if out in (BASE / 'source_data', args.panel_d_source.resolve()):
        raise SystemExit('Write to a new folder; use --install to add files to source_data/.')
    paired = pd.read_csv(args.panel_d_source / 'Fig3D_confirmation_paired.csv')
    kin, kin_path = archived_kinematics(args.phase16_root)
    sel = selection_table(paired, kin)
    rule4 = int(sel.revision4_rule_abs_dev_A.idxmin())
    passing = sel[sel.passes_checks]
    if args.seed is not None:
        seed = int(args.seed)
    elif passing.empty:
        raise SystemExit('No trial passes the representativeness checks; choose --seed explicitly and report why.')
    else:
        seed = int(passing.index[0])
    print(f'revision-4 rule (activity-only closest to mean): seed {rule4}; '
          f'joint rule: seed {int(sel.index[0])}; first trial passing all checks: '
          f'{int(passing.index[0]) if len(passing) else None}; shown: {seed}')

    ex = DataStore(args.raw_root, BASE / 'examples', raw_root=args.raw_root, seed=seed)
    profiles, kinematics = f3.example_data(ex)
    # identity with the archived profile-snapshot table (direction-averaged, every second 10-ms snapshot)
    worst = {}
    for kind in f3.KINDS:
        a = kin[(kin.seed == seed) & (kin.kind == kind)].sort_values('t_ms')
        b = kinematics[kinematics.kind == kind].set_index('t_ms').loc[a.t_ms.to_numpy()]
        for col_a, col_b in (('x', 'phi_s_disp_deg'), ('psi_u', 'psi_u_disp_deg'), ('psi_D', 'psi_D_disp_deg')):
            worst[f'{kind}:{col_b}'] = float(np.nanmax(np.abs(a[col_a].to_numpy() - b[col_b].to_numpy())))
    if max(worst.values()) > 1e-6:
        raise SystemExit(f'Raw snapshots do not reproduce the archived B_kinematics rows: {worst}. Nothing written.')
    f_single = {kind: branch_f_rest(ex, kind) for kind in f3.KINDS}

    out.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(out / 'Fig3ABC_example_profiles.csv', index=False, float_format='%.10g')
    f3.state_extents(profiles).to_csv(out / 'Fig3ABC_example_state_extents.csv', index=False, float_format='%.10g')
    kinematics.to_csv(out / 'Fig3ABC_example_kinematics.csv', index=False, float_format='%.10g')
    sel.reset_index().to_csv(out / 'Fig3_example_selection.csv', index=False, float_format='%.6g')
    for name in PANEL_D_FILES:
        shutil.copy2(args.panel_d_source / name, out / name)
    row = sel.loc[seed]
    k2 = kinematics[np.isclose(kinematics.t_ms, 2000.)].set_index('kind')
    meta = {
        'schema': 'Fig3_example_trial_v2 (CPL revision 5)',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'data_set': 'profile-snapshot set (Phase-16 module B): continuations of the noise-free confirmation-set '
                    '(Phase-15) pre-perturbation states; 10-ms snapshots of s_NMDA, u, R',
        'seed': seed, 'shift_deg': DELTA, 'shift_time_ms': float(ex.snapshot_branch('no_shift').metadata['t0_ms']),
        'noise_pa': 0.0,
        'selection': ('manual --seed' if args.seed is not None else
                      'minimum J = sum of squared z of F_rest(2 s) over activity only, + u, + R, among trials passing '
                      '|z| <= 1, t90 within IQR, trajectory RMS <= median'),
        'revision4_rule_seed': rule4,
        'this_trial': {k: (None if pd.isna(v) else (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)))
                       for k, v in row.items()},
        'confirmation_means_F_rest_2s': {k: float(paired[k].mean()) for k in f3.GROUPS},
        'phi_s_restoration_2s_directions_averaged': {k: float(1 - k2.loc[k, 'phi_s_disp_deg'] / DELTA) for k in f3.KINDS},
        'phi_s_restoration_2s_single_branches': {k: {'p': float(1 - k2.loc[k, 'phi_s_disp_plus_deg'] / DELTA),
                                                     'm': float(1 - k2.loc[k, 'phi_s_disp_minus_deg'] / DELTA)}
                                                 for k in f3.KINDS},
        'F_rest_2s_400ms_pav_single_branches_snapshot_set': f_single,
        'max_abs_diff_vs_archived_B_kinematics_deg': worst,
        'raw_inputs_sha256': {str(p): v['sha256'] for p, v in ex.used.items()},
        'archived_B_kinematics_sha256': sha256(kin_path),
        'panel_d_tables_sha256': {name: sha256(out / name) for name in PANEL_D_FILES},
    }
    (out / 'Fig3_example_trial.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    print(f'Fig. 3 example: seed {seed} (shift at {meta["shift_time_ms"]:.0f} ms); written: {out}')
    print('  F_rest(2 s), directions averaged (Fig. 3d):',
          {k: round(float(row[f'F_rest_2s_{s}']), 3) for k, s in zip(f3.KINDS, 'AUR')})
    print('  single branches (+/-):', {k: (round(v['p'], 3), round(v['m'], 3)) for k, v in f_single.items()})
    if args.install:
        dst = BASE / 'source_data'
        for name in PANEL_D_FILES:
            if sha256(dst / name) != sha256(out / name):
                raise SystemExit(f'{name} differs from source_data; refusing to install.')
        for name in NEW_FILES:
            if (dst / name).exists():
                if sha256(dst / name) != sha256(out / name):
                    raise SystemExit(f'source_data/{name} exists and differs; not overwritten.')
                continue
            shutil.copy2(out / name, dst / name)
        print(f'Installed into {dst} (no existing file overwritten).')


if __name__ == '__main__':
    main()
