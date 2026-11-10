"""Figure 4 (manuscript v10, GC_STP-text-14): stored facilitation acts through the output route on a
rapidly rewritten synaptic reference.

  a  former Fig. 4b: F_rest(2 s) of the full, output-only and release-only models after an 11.25 deg
     activity-only shift in the confirmation set (prespecified output-only vs release-only contrast);
  b  former Fig. 5b: clamp set, restoration with the facilitation profile clamped vs free;
  c  former Fig. 5d: profile-snapshot set, phi_s, psi_eff, psi_u and psi_D after an 11.25 deg shift,
     with lower strip showing delta_eff = d_ax(phi_s, psi_eff).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from dynamics import paired_matrix, seed_mean_band
from plot_style import panel_letter
from captions_cpl import CAPTIONS
import phase16_data as P16
import visual_grammar as vg
import fig5_mechanism as F5

STEM = 'Fig4_Output_route_on_rewritten_reference'
CAPTION_KEY = 'Fig4'
SOURCE_FILES = [
    'Fig4A_recovery_paired.csv',
    'Fig4A_archived_contrasts.csv',
    'Fig4B_clamp_time_courses.csv',
    'Fig4B_clamp_contrasts.csv',
    'Fig4C_full_11p25_kinematics_band.csv',
    'Fig4C_full_11p25_kinematics_by_trial.csv',
]
RECOVERY_MODES = ['dynamic', 'u_output_only', 'u_release_only']
DELTA = 11.25


# ================================================================== data layer
def confirmation_data(ds):
    """a: Phase-15 confirmation set (same selection and pairing as former Fig. 4b)."""
    seed = ds.table('seed15')
    s = seed[(seed.shift_kind == 'activity_shift') & np.isclose(seed.abs_delta_deg, DELTA)]
    return {'a': paired_matrix(s, RECOVERY_MODES), 'contrasts': ds.table('contrasts15')}


def phase16_data(ds):
    """b, c: Phase-16 clamp and profile-snapshot sets (same rows and bands as former Fig. 5b,d)."""
    p16 = P16.attach(ds)
    rest_ci = p16.table('A', 'restoration_mean_ci')
    rows = []
    for cond in F5.COND_ORDER:
        for mode in ('normal', 'clamp'):
            g = rest_ci[(rest_ci.condition == cond) & (rest_ci['mode'] == mode)
                        & np.isclose(rest_ci.abs_delta, DELTA)].sort_values('t_ms')
            if g.empty:
                raise ValueError(f'Missing clamp-set time course {cond}/{mode}')
            rows.append(g.assign(model=F5.COND_ENC[cond]['label']))
    kl = P16.kinematics_long(p16.table('B', 'kinematics'), 'dynamic', 'activity_shift', DELTA)
    kl_band = kl.copy()
    kl_band['delta_eff'] = -kl_band['drive_offset']
    band_rows = []
    for col in ('x', 'psi_eff', 'psi_u', 'psi_D', 'delta_eff'):
        t, m, lo, hi, n = seed_mean_band(kl_band, col, group='t_ms')
        band_rows.append(pd.DataFrame({'trace': col, 't_ms': t, 'mean_deg': m,
                                       'ci_low': lo, 'ci_high': hi, 'n_trials': n}))
    return {'b_time': pd.concat(rows, ignore_index=True),
            'b_contrasts': p16.table('A', 'P1_paired_contrasts'),
            'c_band': pd.concat(band_rows, ignore_index=True), 'c_trials': kl}


def compute_panels(ds, window=None):
    return {**confirmation_data(ds), **phase16_data(ds)}


def export_source_data(ex, d):
    ex.csv('Fig4A_recovery_paired', d['a'].reset_index())
    ex.csv('Fig4A_archived_contrasts', d['contrasts'])
    ex.csv('Fig4B_clamp_time_courses', d['b_time'])
    ex.csv('Fig4B_clamp_contrasts', d['b_contrasts'])
    ex.csv('Fig4C_full_11p25_kinematics_band', d['c_band'])
    ex.csv('Fig4C_full_11p25_kinematics_by_trial', d['c_trials'])


def load_source_data(source, window=None):
    source = Path(source)
    missing = [n for n in SOURCE_FILES if not (source/n).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing Fig. 4 source data in {source}: {missing}')
    c_band = pd.read_csv(source/'Fig4C_full_11p25_kinematics_band.csv')
    c_trials = pd.read_csv(source/'Fig4C_full_11p25_kinematics_by_trial.csv')
    if 'delta_eff' not in set(c_band['trace']):
        kl_band = c_trials.copy()
        kl_band['delta_eff'] = -kl_band['drive_offset']
        t, m, lo, hi, n = seed_mean_band(kl_band, 'delta_eff', group='t_ms')
        c_band = pd.concat([c_band, pd.DataFrame({'trace': 'delta_eff', 't_ms': t, 'mean_deg': m,
                                                  'ci_low': lo, 'ci_high': hi, 'n_trials': n})],
                           ignore_index=True)
    return {
        'a': pd.read_csv(source/'Fig4A_recovery_paired.csv').set_index('seed'),
        'contrasts': pd.read_csv(source/'Fig4A_archived_contrasts.csv'),
        'b_time': pd.read_csv(source/'Fig4B_clamp_time_courses.csv'),
        'b_contrasts': pd.read_csv(source/'Fig4B_clamp_contrasts.csv'),
        'c_band': c_band,
        'c_trials': c_trials,
    }


# ================================================================== plot layer
INK, MID = vg.NEUTRAL['ink'], vg.NEUTRAL['mid']


def _key(fig, L, x_mm, y_mm):
    """Shared key: model = colour / marker shape / line style; clamp state = marker fill and line weight."""
    handles = [vg.model_handle(k) for k in ('full', 'output', 'release')]
    handles += [Line2D([], [], color=INK, lw=1.35, marker='o', markersize=vg.MS['key'], markerfacecolor=INK,
                       markeredgecolor=INK, label=r'$u$ clamped'),
                Line2D([], [], color=MID, lw=.8, marker='o', markersize=vg.MS['key'], markerfacecolor='white',
                       markeredgecolor=MID, label='Free')]
    vg.fig_legend(L, handles, x_mm, y_mm, loc='center left', ncol=5, columnspacing=1.6, handlelength=2.6)


def draw_figure(d):
    """Fig. 4, CPL revision 2: one row, closed axes, letters on one row, one shared key."""
    L = vg.Layout(height_mm=75.5)
    fig = L.fig
    y, h = 11.0, 46.0
    xa, wa = 16.0, 31.0
    xb, wb = 67.0, 45.0
    xc, wc = 130.0, 37.0

    ax_a = L.axes(xa, y, len(RECOVERY_MODES) * vg.FREST_GROUP_MM, h)   # CPL revision 6: same spec as Fig. 3(d)
    vg.frest_panel(ax_a, d['a'][RECOVERY_MODES].to_numpy(float),
                   [vg.MODELS['full'], vg.MODELS['output'], vg.MODELS['release']],
                   ['Full', 'Output-\nonly', 'Release-\nonly'], connect=[(1, 2)])

    ax_b = L.axes(xb, y, wb, h)
    F5._panel_b(ax_b, d)
    ax_c_sub = L.axes(xc, y, wc, 11.0)
    ax_c = L.axes(xc, y + 13.5, wc, h - 13.5, sharex=ax_c_sub)
    F5._panel_d(ax_c, {'d_band': d['c_band'], 'd_trials': d.get('c_trials')}, markers=False, ax_sub=ax_c_sub)

    yl = y + h + 4.8
    for letter, x in (('a', 1.5), ('b', 53.0), ('c', 116.0)):
        L.letter(x, yl, letter, va='center')
    _key(fig, L, 1.5, 70.0)          # CPL revision 4 (C): aligned with letter (a); the key now ends above (b)
    return fig


def make_figure(ds, ex, window=None):
    d = compute_panels(ds, window)
    export_source_data(ex, d)
    with vg.style():
        fig = draw_figure(d)
        ex.save(fig, STEM, CAPTIONS[CAPTION_KEY])
