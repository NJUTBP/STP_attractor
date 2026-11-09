"""Phase-16 archival tables for Fig. 5 and Extended Data Figs. 11–14 (manuscript v6).

The four second-stage mechanism experiments are, in manuscript terms,
    module A = clamp set, module B = profile-snapshot set,
    module C = relative-coordinate noise set, module D = timescale-scaling set.

This module never runs a simulation and never re-estimates an archived inferential
statistic. It (1) resolves the Phase-16 tables, (2) records SHA-256 provenance through the
caller's DataStore, and (3) checks that the tables are the ones against which the v6
manuscript numbers were verified (run identity + key statistics). A mismatch raises an
error instead of silently drawing a different run.

Default source: ``bundled_data/phase16`` inside this package. These are byte copies of the
handoff outputs (Phase16_A_out, phase16_B_out, phase16_C_out and the final 8-seed module-D
run PHASE_d_out_new). The module-A ``analysis/recovery`` tables were not in the handoff zip;
they were regenerated with the project's own ``phase16/recovery_analysis.py --module A``
(seed 0, 10,000 resamples). Re-running that script for modules B and D reproduces the
bundled B/D recovery tables to machine precision.

To read a local results tree instead, set ``PHASE16_ROOT`` in settings.py (or pass
``--phase16-root``). Expected layout, as written by run/analyze/recovery_analysis.py:
    <root>/module_A/run_manifest.json
    <root>/module_A/analysis/A_*.csv
    <root>/module_A/analysis/recovery/A_P*.csv      (likewise module_B, module_C, module_D)
``<root>`` may be ``results``, ``results/phase16`` or a folder holding module_A..module_D.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BUNDLED_ROOT = HERE / 'bundled_data' / 'phase16'

MANUSCRIPT_NAMES = {'A': 'clamp set', 'B': 'profile-snapshot set',
                    'C': 'relative-coordinate noise set', 'D': 'timescale-scaling set'}

FILES = {
    'A': {
        'restoration_by_trial': 'analysis/A_restoration_by_trial.csv',
        'restoration_mean_ci': 'analysis/A_restoration_mean_ci.csv',
        'force_curve_mean_ci': 'analysis/A_force_curve_mean_ci.csv',
        'velocity_by_trial': 'analysis/A_velocity_by_trial.csv',
        'rate_fits_by_trial': 'analysis/A_rate_fits_by_trial.csv',
        'P1_paired_contrasts': 'analysis/recovery/A_P1_paired_contrasts.csv',
        'P2_small_delta_slopes_by_trial': 'analysis/recovery/A_P2_small_delta_slopes_by_trial.csv',
        'P2_small_delta_slopes_summary': 'analysis/recovery/A_P2_small_delta_slopes_summary.csv',
        'P2_route_contrasts': 'analysis/recovery/A_P2_route_contrasts.csv',
        'P2_peak_force': 'analysis/recovery/A_P2_peak_force.csv',
    },
    'B': {
        'kinematics': 'analysis/B_kinematics.csv',
        'two_variable_fits': 'analysis/B_two_variable_fits.csv',
        'P3_gain_cluster_bootstrap': 'analysis/recovery/B_P3_gain_cluster_bootstrap.csv',
        'P3_gain_point_estimates': 'analysis/recovery/B_P3_gain_point_estimates.csv',
        'P4_a_comparison_by_trial': 'analysis/recovery/B_P4_a_comparison_by_trial.csv',
        'P4_a_comparison_summary': 'analysis/recovery/B_P4_a_comparison_summary.csv',
    },
    'C': {
        'curves': 'analysis/C_msd_and_relative_variance.csv',
        'P5_cluster_bootstrap': 'analysis/recovery/C_P5_cluster_bootstrap.csv',
        'P5_direct_B_prediction_test': 'analysis/recovery/C_P5_direct_B_prediction_test.csv',
        'P5_point_estimates': 'analysis/recovery/C_P5_point_estimates.csv',
    },
    'D': {
        'restoration_2s_by_trial': 'analysis/D_restoration_2s_by_trial.csv',
        'P6_grouped': 'analysis/recovery/D_P6_grouped_restoration_seed_bootstrap.csv',
        'P6_key_contrasts': 'analysis/recovery/D_P6_key_paired_contrasts.csv',
        'P6_trend_summary': 'analysis/recovery/D_P6_trend_slopes_summary.csv',
        'P6_compensation': 'analysis/recovery/D_P6_compensation_decomposition.csv',
    },
}

# Run identity of the tables used for manuscript v6 (from each module's run_manifest.json).
EXPECTED_RUNS = {
    'A': (list(range(200, 216)), 1440),
    'B': (list(range(200, 216)), 368),
    'C': (list(range(200, 216)), 128),
    'D': (list(range(200, 208)), 720),
}


def _sel(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        col = df[k]
        if isinstance(v, float):
            m &= np.isclose(col.to_numpy(float), v)
        else:
            m &= (col.astype(str).to_numpy() == str(v))
    out = df[m]
    if len(out) != 1:
        raise ValueError(f'Expected exactly one row for {kw}; found {len(out)}')
    return out.iloc[0]


# Key statistics quoted in manuscript v6. Tolerances are far below the reported precision;
# they only guard against drawing a different run (e.g. an older module-D pilot).
IDENTITY_CHECKS = [
    ('A', 'P1 full clamp-minus-normal F(600 ms)',
     lambda s: _sel(s.table('A', 'P1_paired_contrasts'), condition='dynamic')['mean'], 0.3494868808, 1e-6),
    ('A', 'P1 release-only pairs (seed 212 has no spikes at 600 ms)',
     lambda s: _sel(s.table('A', 'P1_paired_contrasts'), condition='u_release_only')['n_pairs'], 15, 0),
    ('A', 'P2 full small-offset gain',
     lambda s: _sel(s.table('A', 'P2_small_delta_slopes_summary'), condition='dynamic')['mean_slope_per_s'], 1.2605382551, 1e-6),
    ('B', 'P3 pooled gain',
     lambda s: _sel(s.table('B', 'P3_gain_point_estimates'), scope='pooled')['gain_per_s'], 4.0559815309, 1e-6),
    ('B', 'P3 pooled R^2',
     lambda s: _sel(s.table('B', 'P3_gain_point_estimates'), scope='pooled')['r2'], 0.0184459271, 1e-6),
    ('B', 'full 11.25-deg F_inf (mean of trials)',
     lambda s: s.table('B', 'two_variable_fits').query("condition == 'dynamic' and abs_delta == 11.25")['F_inf'].mean(), 0.2949187452, 1e-6),
    ('C', 'full 60-pA late (4-8 s) MSD slope',
     lambda s: _sel(s.table('C', 'P5_cluster_bootstrap'), condition='dynamic', noise=60.0,
                    parameter='late_msd_slope')['point_estimate'], 33.5459055008, 1e-6),
    ('C', 'predicted late/early ratio (1-F_inf)^2',
     lambda s: _sel(s.table('C', 'P5_direct_B_prediction_test'), noise=60.0)['predicted_ratio_from_B'], 0.4971395759, 1e-6),
    ('D', 'full 11.25-deg tau_f 250 minus 1000 ms',
     lambda s: _sel(s.table('D', 'P6_key_contrasts'), contrast='F_tf250_minus_tf1000', condition='dynamic',
                    abs_delta=11.25)['mean'], -0.2120695750, 1e-6),
    ('D', 'output-only 22.5 deg, tau_r 5 ms, seed 202',
     lambda s: _sel(s.table('D', 'restoration_2s_by_trial'), tf=1000.0, tr=5.0, condition='u_output_only',
                    kind='activity_shift', seed=202, abs_delta=22.5)['F_2s'], -1.4820258670, 1e-6),
]


def resolve_root(root=None):
    """Return the folder that contains module_A..module_D."""
    if root is None:
        return BUNDLED_ROOT
    p = Path(root).expanduser().resolve()
    for q in [p, p / 'phase16', p / 'results' / 'phase16']:
        if (q / 'module_A').is_dir() or (q / 'A').is_dir():
            return q
    raise FileNotFoundError(f'No Phase-16 module folders (module_A..module_D) under {p}')


class Phase16Store:
    """Read-only access to the Phase-16 tables with provenance and identity checks."""

    def __init__(self, root=None, record=None, verify=False):
        self.root = resolve_root(root)
        self.bundled = self.root == BUNDLED_ROOT.resolve() or self.root == BUNDLED_ROOT
        self.identity = ('bundled Phase-16 archival table (handoff copy)' if self.bundled
                         else 'local Phase-16 archival table')
        self._record = record
        self.verify = bool(verify)
        self._cache = {}
        self._checked = set()
        self._checking = set()

    # -- paths ---------------------------------------------------------------------------
    def module_dir(self, module):
        for q in [self.root / f'module_{module}', self.root / module]:
            if q.is_dir():
                return q
        raise FileNotFoundError(f'Missing Phase-16 module {module} under {self.root}. No data will be fabricated.')

    def path(self, module, key):
        p = self.module_dir(module) / FILES[module][key]
        if not p.is_file():
            raise FileNotFoundError(f'Missing required Phase-16 table {p}. No data will be fabricated.')
        return p

    def _note(self, p):
        if self._record is not None:
            self._record(p, self.identity)

    # -- data ----------------------------------------------------------------------------
    def run_manifest(self, module):
        p = self.module_dir(module) / 'run_manifest.json'
        if not p.is_file():
            raise FileNotFoundError(f'Missing {p}; cannot confirm which Phase-16 run this is.')
        self._note(p)
        return json.loads(p.read_text(encoding='utf-8'))

    def table(self, module, key):
        k = (module, key)
        p = self.path(module, key)
        if k not in self._cache:
            self._cache[k] = pd.read_csv(p)
        # Provenance on every access, not only the first (cached) read: make_suppfigures resets
        # its record after each figure, so each per-figure manifest must see its own inputs,
        # together with the run manifest that identifies the Phase-16 run.
        self._note(p)
        rm = self.module_dir(module) / 'run_manifest.json'
        if rm.is_file():
            self._note(rm)
        if self.verify and module not in self._checked and module not in self._checking:
            self.check_module(module)
        return self._cache[k].copy()

    # -- identity ------------------------------------------------------------------------
    def check_module(self, module):
        self._checking.add(module)
        try:
            problems = []
            man = self.run_manifest(module)
            seeds, n_branches = EXPECTED_RUNS[module]
            if [int(s) for s in man.get('seeds', [])] != seeds:
                problems.append(f"seeds {man.get('seeds')} != expected {seeds[0]}-{seeds[-1]}")
            if int(man.get('n_branches', -1)) != n_branches:
                problems.append(f"n_branches {man.get('n_branches')} != expected {n_branches}")
            for mod, label, fn, expected, tol in IDENTITY_CHECKS:
                if mod != module:
                    continue
                value = float(fn(self))
                if not np.isfinite(value) or abs(value - float(expected)) > tol:
                    problems.append(f'{label}: {value:.10g} (manuscript v6 table: {expected})')
            if problems:
                raise ValueError(
                    f'Phase-16 module {module} ({MANUSCRIPT_NAMES[module]}) at {self.module_dir(module)} is not the run '
                    'used for manuscript v6:\n  - ' + '\n  - '.join(problems) +
                    '\nPoint PHASE16_ROOT to the correct run, or set PHASE16_VERIFY=False only if you intend '
                    'to redraw from different results (then the manuscript numbers must be updated too).')
            self._checked.add(module)
        finally:
            self._checking.discard(module)

    def check_all(self):
        for m in 'ABCD':
            self.check_module(m)
        return True


def attach(ds, root=None, verify=None):
    """Attach a Phase16Store to a DataStore (``ds.phase16``), reusing its provenance recorder."""
    if getattr(ds, 'phase16', None) is not None:
        return ds.phase16
    try:
        import settings
        if root is None:
            root = getattr(settings, 'PHASE16_ROOT', None)
        if verify is None:
            verify = getattr(settings, 'PHASE16_VERIFY', True)
    except ImportError:
        verify = True if verify is None else verify
    record = getattr(ds, 'record', None)
    ds.phase16 = Phase16Store(root, record=record, verify=verify)
    return ds.phase16


# -- small shared transforms (no inference) ---------------------------------------------
KIN_COLUMNS = ('x', 'psi_eff', 'psi_u', 'psi_D')


def kinematics_long(kin, condition, kind, delta):
    """Per-trial displacement traces (already direction-averaged and reference-subtracted
    by analyze_phase16.py) with the effective drive centre psi_eff = x + drive_offset."""
    g = kin[(kin.condition == condition) & (kin.kind == kind) & np.isclose(kin.abs_delta, delta)].copy()
    if g.empty:
        raise ValueError(f'No kinematics for {condition}/{kind}/{delta}')
    if g.duplicated(['seed', 't_ms']).any():
        raise ValueError('Duplicate seed/time rows in B_kinematics; refusing silent aggregation')
    g['psi_eff'] = g['x'] + g['drive_offset']
    return g[['seed', 't_ms', *KIN_COLUMNS, 'drive_offset', 'v_toward_old']].sort_values(['seed', 't_ms'])


def running_mean(y, window=25):
    """Centred running mean (window in samples) that shrinks at the edges; display only."""
    return pd.Series(np.asarray(y, float)).rolling(window, center=True, min_periods=max(1, window // 5)).mean().to_numpy()


def fmt_ci(mean, lo, hi, digits=3, signed=False):
    f = f'{{:{"+" if signed else ""}.{digits}f}}'
    s = f'{f.format(mean)} [{lo:.{digits}f}, {hi:.{digits}f}]'
    return s.replace('-', '−')
