"""Figure 3: selective state shifts -- one representative confirmation trial (a)-(c) and the confirmation set (d).

CPL revision 5 (supersedes the revision-4 panels a-c; panel d data, statistics and layout grammar unchanged)
---------------------------------------------------------------------------------------------------------------
(a)-(c) previously showed the +11.25 deg continuation of seed 0 of the *initial* set (a different data set from d),
with the bump position phi from a trailing 400-ms spike population vector drawn only from 400 ms after the shift,
i.e. after most of the restoring movement (Fig. 4c) had already happened.  They now show one *confirmation* trial
(the trial ringed in d, chosen by tools/export_fig3_example.py; criteria in Fig3_example_trial.json) from the
profile-snapshot set, which continues the same noise-free pre-perturbation states as the confirmation set:
  top    -- folded profiles of s_NMDA (activity), u - U (facilitation) and D = 1 - R (depletion), each normalized to
            its peak, at the first stored snapshot (10 ms) after the shift;
  bottom -- displacement from the old position of the activity center phi_s (center of s_NMDA) and of the
            facilitation-trace position psi_u, from 10-ms snapshots.
Angles and displacements are measured from the unshifted continuation, sign-aligned and averaged over the +/-11.25 deg
shifts, exactly as for d and Fig. 4(c) (SI Secs. S5-S6).  phi_s is a different readout from the phi of F_rest in d;
F_rest is neither redefined nor recomputed here.

Data layer -- ``compute_panels`` (raw module-B snapshots through ``DataStore.snapshot_branch``) / ``export_source_data``.
Plot layer -- ``draw_figure`` draws only exported quantities (``tools/replay_cpl.py`` re-renders from source data).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from data_io import DataStore
from dynamics import axial_delta, state_axis, paired_matrix
from captions_cpl import CAPTIONS
import visual_grammar as vg

STEM = 'Fig3_Attractor_state_surgery'
CAPTION_KEY = 'Fig3'
KINDS = ['activity_shift', 'activity_u_shift', 'activity_r_shift']
GROUPS = ['activity_shift', 'activity_u_shift', 'activity_r_shift', 'co_shift']
DELTA = 11.25
EXAMPLE_SEED = 213            # see source_data/Fig3_example_trial.json (selection rule and checks)
U_BASELINE = .4
PROFILE_GRID = np.round(np.arange(-89.75, 90.0 + 1e-9, .25), 2)   # degrees from the old position (one axial period)
SOURCE_FILES = ['Fig3ABC_example_profiles.csv', 'Fig3ABC_example_state_extents.csv', 'Fig3ABC_example_kinematics.csv',
                'Fig3D_confirmation_paired.csv', 'Fig3D_archived_contrasts.csv']
KIN_COLUMNS = ['seed', 'kind', 't_ms', 'phi_s_disp_deg', 'psi_u_disp_deg', 'psi_D_disp_deg',
               'phi_s_disp_plus_deg', 'phi_s_disp_minus_deg', 'psi_u_disp_plus_deg', 'psi_u_disp_minus_deg']
EXTENT_COLUMNS = ['seed', 'kind', 'variable', 'snapshot_ms', 'left_deg', 'right_deg', 'center_deg']
VARIABLES_TOP = [('activity', 'activity_s_norm'), ('facilitation', 'facilitation_u_norm'),
                 ('depletion', 'depletion_D_norm')]
PROFILE_COLUMNS = ['seed', 'kind', 'snapshot_ms', 'angle_from_old_deg', 'activity_s_norm', 'facilitation_u_norm',
                   'depletion_D_norm', 'activity_s', 'facilitation_u_minus_U', 'depletion_D']


# ================================================================== data layer
def _centers(raw, U=U_BASELINE):
    """Axial centres (SI Eq. S9) of s_NMDA, max(u - U, 0) and max(1 - R, 0) at every stored snapshot."""
    th = raw['theta_e'].astype(float)
    s, u, R = (raw[k].astype(float) for k in ('snap_s', 'snap_u', 'snap_R'))
    return {'t': raw['snap_t'].astype(float), 'phi_s': state_axis(np.clip(s, 0., None), th),
            'psi_u': state_axis(np.maximum(u - U, 0.), th), 'psi_D': state_axis(np.maximum(1. - R, 0.), th)}


def _fold(v):
    """Average neuron n and n + N/2 (theta and theta + 180 deg): the two bumps are folded onto one axial period."""
    n = v.shape[-1] // 2
    return .5 * (v[..., :n] + v[..., n:])


def _aligned_profile(values, theta_fold, ref_deg, sign, grid=PROFILE_GRID):
    """Folded profile against the sign-aligned axial angle from ref_deg, linearly interpolated (periodic)."""
    a = sign * axial_delta(theta_fold, ref_deg)
    order = np.argsort(a)
    a, v = a[order], np.asarray(values, float)[order]
    a = np.r_[a[-1] - 180., a, a[0] + 180.]
    v = np.r_[v[-1], v, v[0]]
    return np.interp(grid, a, v)


def half_max_extent(grid, y):
    """Edges (linear interpolation) of the contiguous region around the peak where y >= max(y)/2, and the axial
    centre (SI Eq. S9) of the profile; display summary of a folded, normalized profile."""
    grid, y = np.asarray(grid, float), np.asarray(y, float)
    i, half = int(np.argmax(y)), .5 * np.nanmax(y)
    lo = i
    while lo > 0 and y[lo - 1] >= half:
        lo -= 1
    hi = i
    while hi < len(y) - 1 and y[hi + 1] >= half:
        hi += 1
    if lo == 0 or hi == len(y) - 1:
        raise ValueError('Profile above half maximum at the edge of the angular grid')
    left = np.interp(half, [y[lo - 1], y[lo]], [grid[lo - 1], grid[lo]])
    right = np.interp(half, [y[hi + 1], y[hi]], [grid[hi + 1], grid[hi]])
    centre = float(np.rad2deg(np.angle(np.sum(np.clip(y, 0, None) * np.exp(2j * np.deg2rad(grid))))) / 2)
    return float(left), float(right), centre


def state_extents(profiles):
    rows = []
    for kind, g in profiles.groupby('kind', sort=False):
        for var, col in VARIABLES_TOP:
            left, right, centre = half_max_extent(g.angle_from_old_deg, g[col])
            rows.append(dict(seed=int(g.seed.iloc[0]), kind=kind, variable=var, snapshot_ms=float(g.snapshot_ms.iloc[0]),
                             left_deg=left, right_deg=right, center_deg=centre))
    return pd.DataFrame(rows)[EXTENT_COLUMNS]


def example_data(ds: DataStore, U=U_BASELINE, delta=DELTA, snapshot=0):
    """Direction-averaged, sign-aligned profiles (first snapshot) and centre trajectories of trial ds.seed."""
    mag = f'{delta:g}'.replace('.', 'p')
    ref_raw = ds.snapshot_branch('no_shift')
    ref = _centers(ref_raw, U)
    th = ref_raw['theta_e'].astype(float)
    th_fold = th[: len(th) // 2]
    ref_angle = float(ref['phi_s'][snapshot])            # old position = activity centre without the shift
    kin, prof = [], []
    for kind in KINDS:
        per, shapes = {}, {}
        for sign, tag in ((1., 'p'), (-1., 'm')):
            raw = ds.snapshot_branch(kind, tag, mag)
            c = _centers(raw, U)
            if not np.array_equal(c['t'], ref['t']):
                raise ValueError(f'{kind} {tag}: snapshot times differ from the unshifted continuation')
            per[sign] = {k: sign * axial_delta(c[k], ref[k]) for k in ('phi_s', 'psi_u', 'psi_D')}
            s = _fold(raw['snap_s'][snapshot].astype(float))
            u = _fold(np.maximum(raw['snap_u'][snapshot].astype(float) - U, 0.))
            D = _fold(np.maximum(1. - raw['snap_R'][snapshot].astype(float), 0.))
            shapes[sign] = [_aligned_profile(x, th_fold, ref_angle, sign) for x in (s, u, D)]
        avg = {k: np.nanmean([per[1.][k], per[-1.][k]], axis=0) for k in ('phi_s', 'psi_u', 'psi_D')}
        kin.append(pd.DataFrame({
            'seed': int(ds.seed), 'kind': kind, 't_ms': ref['t'],
            'phi_s_disp_deg': avg['phi_s'], 'psi_u_disp_deg': avg['psi_u'], 'psi_D_disp_deg': avg['psi_D'],
            'phi_s_disp_plus_deg': per[1.]['phi_s'], 'phi_s_disp_minus_deg': per[-1.]['phi_s'],
            'psi_u_disp_plus_deg': per[1.]['psi_u'], 'psi_u_disp_minus_deg': per[-1.]['psi_u']}))
        mean_shape = [.5 * (a + b) for a, b in zip(shapes[1.], shapes[-1.])]
        prof.append(pd.DataFrame({
            'seed': int(ds.seed), 'kind': kind, 'snapshot_ms': float(ref['t'][snapshot]),
            'angle_from_old_deg': PROFILE_GRID,
            'activity_s_norm': mean_shape[0] / mean_shape[0].max(),
            'facilitation_u_norm': mean_shape[1] / mean_shape[1].max(),
            'depletion_D_norm': mean_shape[2] / mean_shape[2].max(),
            'activity_s': mean_shape[0], 'facilitation_u_minus_U': mean_shape[1], 'depletion_D': mean_shape[2]}))
    return pd.concat(prof, ignore_index=True)[PROFILE_COLUMNS], pd.concat(kin, ignore_index=True)[KIN_COLUMNS]


def compute_panels(ds, window=None, seed=EXAMPLE_SEED):
    """All quantities drawn in Fig. 3.  (d) is the archived confirmation-set table, unchanged since manuscript v6."""
    ex = DataStore(ds.root, ds.examples, raw_root=ds.raw_root, seed=seed)
    profiles, kinematics = example_data(ex)
    ds.used.update(ex.used)
    table = ds.table('seed15')
    sub = table[(table.condition == 'dynamic') & np.isclose(table.abs_delta_deg, DELTA)]
    return {'profiles': profiles, 'extents': state_extents(profiles), 'kinematics': kinematics,
            'd_pivot': paired_matrix(sub, GROUPS, column='shift_kind'), 'contrasts': ds.table('contrasts15')}


def export_source_data(ex, d):
    ex.csv('Fig3ABC_example_profiles', d['profiles'])
    ex.csv('Fig3ABC_example_state_extents', d['extents'])
    ex.csv('Fig3ABC_example_kinematics', d['kinematics'])
    ex.csv('Fig3D_confirmation_paired', d['d_pivot'].reset_index())
    ex.csv('Fig3D_archived_contrasts', d['contrasts'])


def load_source_data(source, window=None):
    source = Path(source)
    missing = [n for n in SOURCE_FILES if not (source / n).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing Fig. 3 source data in {source}: {missing}. '
                                'Write them with tools/export_fig3_example.py (needs the module-B branches).')
    prof = pd.read_csv(source / 'Fig3ABC_example_profiles.csv')
    kin = pd.read_csv(source / 'Fig3ABC_example_kinematics.csv')
    if list(prof.columns) != PROFILE_COLUMNS or list(kin.columns) != KIN_COLUMNS:
        raise ValueError('Fig. 3(a)-(c) source data do not have the revision-5 columns.')
    if prof.seed.nunique() != 1 or kin.seed.nunique() != 1 or prof.seed.iloc[0] != kin.seed.iloc[0]:
        raise ValueError('Fig. 3(a)-(c) source data must describe exactly one trial.')
    ext = pd.read_csv(source / 'Fig3ABC_example_state_extents.csv')
    if list(ext.columns) != EXTENT_COLUMNS:
        raise ValueError('Fig. 3(a)-(c) state-extent table does not have the revision-5 columns.')
    return {'profiles': prof, 'extents': ext, 'kinematics': kin,
            'd_pivot': pd.read_csv(source / 'Fig3D_confirmation_paired.csv').set_index('seed')[GROUPS],
            'contrasts': pd.read_csv(source / 'Fig3D_archived_contrasts.csv')}


# ================================================================== plot layer
INK, MID, LIGHT = vg.NEUTRAL['ink'], vg.NEUTRAL['mid'], vg.NEUTRAL['light']
ACT, FAC, DEP = vg.VARIABLES['activity'], vg.VARIABLES['u'], vg.VARIABLES['resource']
GUIDE = dict(color=MID, ls=':', lw=vg.LW['ref'], zorder=0)
FAC_TEXT = '#A36A00'                  # darker shade of the facilitation orange: legible as 6-pt text in print
COL_LABELS = ['Activity only', 'Activity + $u$', 'Activity + $R$']


def _key(L, x_mm, y_mm):
    """One key for both rows: a colour marks one variable, as a profile extent (top) or as its centre (bottom)."""
    handles = [(Line2D([], [], color=ACT, ls='-', lw=1.6), r'Activity $s^\mathrm{NMDA}$ (center $\varphi_s$)'),
               (Line2D([], [], color=FAC, ls='--', lw=1.35), r'Facilitation $u-U$ (center ' + vg.psi('u') + ')'),
               (Patch(facecolor=DEP, edgecolor='none'), r'Depletion $D=1-R$')]
    h, lab = zip(*handles)
    return L.fig.legend(h, lab, loc='center left', bbox_to_anchor=(x_mm / L.w, y_mm / L.h), ncol=3, frameon=False,
                        fontsize=vg.FS['legend'], handlelength=2.3, columnspacing=1.8, borderaxespad=0.)


BAR_Y = {'activity': 2., 'facilitation': 1., 'depletion': 0.}
BAR_COLOR = {'activity': ACT, 'facilitation': FAC, 'depletion': DEP}
BAR_LABEL = {
    'activity': 'NMDA gating variable',
    'facilitation': 'Facilitation',
    'depletion': 'Resource depletion',
}


def draw_figure(d):
    """(a)-(c): one column per shift -- where each state variable was 10 ms after the shift (top) and what the activity
    centre and the facilitation-profile centre did next (bottom); (d) confirmation-set statistics, ring = this trial."""
    L = vg.Layout(height_mm=87.)
    X0, W, G = 29.5, 25.8, 3.6          # columns (a)-(c): left margin fits 'NMDA gating variable' cleanly
    YB, HB = 11.5, 36.                  # bottom row: displacement time courses
    YT, HT = 65.0, 13.5                 # top row: extents of the state profiles 10 ms after the shift
    ext, kin = d['extents'], d['kinematics']
    seed = int(kin.seed.iloc[0])
    for col, kind in enumerate(KINDS):
        x0 = X0 + col * (W + G)
        # ---- top: where activity, facilitation and depletion were 10 ms after the shift -------------------------
        ax = L.axes(x0, YT, W, HT)
        for _, r in ext[ext.kind == kind].iterrows():
            ax.barh(BAR_Y[r.variable], r.right_deg - r.left_deg, left=r.left_deg, height=.56,
                    color=BAR_COLOR[r.variable], edgecolor='none', zorder=2)
        for xv in (0., DELTA):
            ax.axvline(xv, **GUIDE)
        ax.set(xlim=(-34, 44), ylim=(-.6, 2.6), xticks=[-20, 0, 20, 40], yticks=[2, 1, 0])
        ax.set_yticklabels([BAR_LABEL[v] for v in ('activity', 'facilitation', 'depletion')] if col == 0 else [])
        ax.tick_params(axis='y', length=0)
        vg.g_ticks(ax, 'x')
        ax.text(0., 1.04, 'Pre-shift ', transform=ax.get_xaxis_transform(), ha='right', va='bottom',
                fontsize=vg.FS['note'], color=MID)
        ax.text(DELTA, 1.04, r' $|\Delta\theta|$', transform=ax.get_xaxis_transform(), ha='left', va='bottom',
                fontsize=vg.FS['note'], color=MID)
        if col == 0:
            ax.text(-.24, 1.04, r'$t$ = 10 ms', transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=vg.FS['note'], color=MID)
        L.letter(x0 - 4.5, YT + HT + 6.0, 'abc'[col], va='center')
        L.text(x0 + 2.0, YT + HT + 6.0, COL_LABELS[col], ha='left', va='center')
        # ---- bottom: what the activity centre and the facilitation-profile centre did next -----------------------
        bx = L.axes(x0, YB, W, HB)
        k = kin[kin.kind == kind].sort_values('t_ms')
        t = k.t_ms.to_numpy() / 1000.
        for yv in (0., DELTA):
            bx.axhline(yv, **GUIDE)
        bx.plot(t, k.phi_s_disp_deg, color=ACT, lw=1.6, zorder=3, solid_capstyle='butt')
        bx.plot(t, k.psi_u_disp_deg, color=FAC, ls='--', lw=1.35, zorder=4)
        bx.set(xlim=(0, 2), ylim=(-1.2, 13.2), xticks=[0, .5, 1, 1.5, 2], yticks=[0, 4, 8, 12])
        vg.g_ticks(bx, 'xy')
        if col == 0:
            bx.set_ylabel('Angular displacement from\npre-shift position (°)')
        else:                           # shared y scale: ticks kept, numbers only once
            bx.tick_params(axis='y', labelleft=False)
        if col == 0:                    # direct reference-line labels
            note = dict(fontsize=vg.FS['note'], va='center')
            bx.text(1.97, DELTA + .78, r'$|\Delta\theta|$', color=MID, ha='right', **note)
            bx.text(1.97, .75, 'Pre-shift position', color=MID, ha='right', **note)
        if col == 1:
            bx.set_xlabel('Time after shift (s)')
    L.text(X0 + 1.5 * W + G, YT - 6.4, 'Angular position relative to pre-shift position (°)', ha='center', va='center')
    traj_handles = [
        Line2D([], [], color=ACT, ls='-', lw=1.6, solid_capstyle='butt',
               label=r'E-neuron activity center $\varphi_s$'),
        Line2D([], [], color=FAC, ls='--', lw=1.35,
               label='Facilitation-profile center ' + vg.psi('u')),
    ]
    vg.fig_legend(L, traj_handles, X0 + 1.5 * W + G, YB + HB + 4.3, loc='center', ncol=2,
                  columnspacing=1.6, handlelength=2.2)

    # ---- (d) confirmation set, F_rest(2 s); ring = trial of (a)-(c); same panel specification as Fig. 4(a) ---------
    XD = 129.8
    ax_d = L.axes(XD, YB, len(GROUPS) * vg.FREST_GROUP_MM, YT + HT - YB)
    vals = d['d_pivot'].to_numpy(float)
    off = vg.frest_panel(ax_d, vals, [vg.SHIFTS[k] for k in GROUPS],
                         ['Activity\nonly', 'Activity\n+ $u$', 'Activity\n+ $R$', 'All\nstates'], connect=[(0, 1)])
    row = list(d['d_pivot'].index).index(seed)
    ax_d.scatter(np.arange(3) + off[row], vals[row, :3], s=62, facecolors='none', edgecolors=INK, linewidths=.9, zorder=6)
    ax_d.scatter([2.40], [.705], s=62, facecolors='none', edgecolors=INK, linewidths=.9, zorder=6)
    ax_d.text(2.56, .705, 'trial in\n(a)–(c)', ha='left', va='center', fontsize=vg.FS['note'], color=INK,
              linespacing=1.15)
    L.letter(XD - 12.3, YT + HT + 6.0, 'd', va='center')
    return L.fig


def make_figure(ds, ex, window=None):
    d = compute_panels(ds, window)
    export_source_data(ex, d)
    with vg.style():
        fig = draw_figure(d)
        ex.save(fig, STEM, CAPTIONS[CAPTION_KEY])
