"""Figure 5: mechanism of restoration and the moving synaptic reference.

Data layer -- ``compute_panels`` / ``export_source_data``: uses the same archived Phase-16
             tables, transforms, bootstrap bands and source-data schemas as manuscript v6.
Plot layer -- ``draw_figure``: manuscript-v7 visual revision, drawn only from exported quantities.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter

from captions_cpl import CAPTIONS
from dynamics import seed_mean_band
from plot_style import panel_letter
import phase16_data as P16
import visual_grammar as vg

STEM = 'Fig5_Restoring_field_and_reference_updating'
CAPTION_KEY = 'Fig5'
SOURCE_FILES = [
    'Fig5B_clamp_time_courses.csv',
    'Fig5B_clamp_contrasts.csv',
    'Fig5C_restoring_field.csv',
    'Fig5C_small_offset_gain_summary.csv',
    'Fig5C_small_offset_gain_by_trial.csv',
    'Fig5D_full_11p25_kinematics_band.csv',
    'Fig5D_full_11p25_kinematics_by_trial.csv',
    'Fig5E_full_60pA_relative_vs_absolute.csv',
    'Fig5E_late_slopes.csv',
    'Fig5F_tau_f_trials.csv',
    'Fig5F_tau_f_summary.csv',
    'Fig5F_tau_f_contrasts.csv',
]

COND_ENC = {
    'dynamic': vg.MODELS['full'],
    'u_output_only': vg.MODELS['output'],
    'u_release_only': vg.MODELS['release'],
}
COND_ORDER = ['dynamic', 'u_output_only', 'u_release_only']
TRACE_ENC = {
    'x': dict(vg.TRACES['activity'], label=r'$\varphi_s$'),
    'psi_eff': dict(vg.TRACES['drive'], label=vg.psi(r'\mathrm{eff}')),
    'psi_u': dict(vg.TRACES['u'], label=vg.psi('u')),
    'psi_D': dict(vg.TRACES['resource'], label=vg.psi('D')),
}


# ================================================================== data layer (v6-equivalent)
def compute_panels(ds, window=None):
    """Compute exactly the quantities exported by the manuscript-v6 Figure 5 module."""
    p16 = P16.attach(ds)
    rest_ci = p16.table('A', 'restoration_mean_ci')
    p1 = p16.table('A', 'P1_paired_contrasts')
    force = p16.table('A', 'force_curve_mean_ci')
    p2s = p16.table('A', 'P2_small_delta_slopes_summary')
    p2t = p16.table('A', 'P2_small_delta_slopes_by_trial')
    kin = p16.table('B', 'kinematics')
    curves = p16.table('C', 'curves')
    cb = p16.table('C', 'P5_cluster_bootstrap')
    dtr = p16.table('D', 'restoration_2s_by_trial')
    grouped = p16.table('D', 'P6_grouped')
    key = p16.table('D', 'P6_key_contrasts')

    d = {}

    # b: same rows/order as the v6 plotting function exported.
    rows = []
    for cond in COND_ORDER:
        for mode in ('normal', 'clamp'):
            g = rest_ci[(rest_ci.condition == cond) & (rest_ci['mode'] == mode)
                        & np.isclose(rest_ci.abs_delta, 11.25)].sort_values('t_ms')
            if g.empty:
                raise ValueError(f'Missing clamp-set time course {cond}/{mode}')
            rows.append(g.assign(model=COND_ENC[cond]['label']))
    d['b_time'] = pd.concat(rows, ignore_index=True)
    d['b_contrasts'] = p1

    # c: archived means/CIs and small-offset summaries.
    d['c_field'] = force
    d['c_gain_summary'] = p2s
    d['c_gain_trials'] = p2t

    # d: same pointwise trial-bootstrap band as v6, now computed in the data layer so replay is
    # entirely source-data driven.
    kl = P16.kinematics_long(kin, 'dynamic', 'activity_shift', 11.25)
    kl_band = kl.copy()
    kl_band['delta_eff'] = -kl_band['drive_offset']
    band_rows = []
    for col in ('x', 'psi_eff', 'psi_u', 'psi_D', 'delta_eff'):
        t, m, lo, hi, n = seed_mean_band(kl_band, col, group='t_ms')
        band_rows.append(pd.DataFrame({'trace': col, 't_ms': t, 'mean_deg': m,
                                       'ci_low': lo, 'ci_high': hi, 'n_trials': n}))
    d['d_band'] = pd.concat(band_rows, ignore_index=True)
    d['d_trials'] = kl

    # e: same 0.5-s display-only running mean and archived cluster-bootstrap slopes as v6.
    g = curves[(curves.condition == 'dynamic') & np.isclose(curves.noise, 60.)].sort_values('t_s')
    smooth = P16.running_mean(g.var_drive_offset, 25)
    out = g[['t_s', 'var_drive_offset', 'msd_phi']].rename(
        columns={'var_drive_offset': 'var_delta_eff_deg2', 'msd_phi': 'msd_phi_deg2'}).copy()
    out['var_delta_eff_running_mean_0p5s_deg2'] = smooth
    d['e_curves'] = out
    d['e_slopes'] = cb[(cb.condition == 'dynamic') & np.isclose(cb.noise, 60.)
                       & cb.parameter.isin(['late_var_delta_eff_slope', 'late_msd_slope'])]

    # f: same trial rows, grouped bootstrap means/CIs and paired contrasts as v6.
    d['f_trials'] = dtr[(dtr.condition == 'dynamic') & (dtr.kind == 'activity_shift')
                        & np.isclose(dtr.tr, 20.)]
    tfs = [250., 1000., 2500.]
    rows = []
    for delta in (11.25, 22.5):
        gg = [grouped[(grouped.condition == 'dynamic') & (grouped.kind == 'activity_shift')
                      & np.isclose(grouped.abs_delta, delta) & np.isclose(grouped.tf, tf)
                      & np.isclose(grouped.tr, 20.)].iloc[0] for tf in tfs]
        rows.append(pd.DataFrame({
            'abs_delta_deg': delta,
            'tau_f_ms': tfs,
            'tau_r_ms': 20.,
            'mean_F_2s': [r.mean_F_2s for r in gg],
            'ci_low': [r.ci_low for r in gg],
            'ci_high': [r.ci_high for r in gg],
            'n_trials': [int(r.n_seeds) for r in gg],
        }))
    d['f_summary'] = pd.concat(rows, ignore_index=True)
    d['f_contrasts'] = key[key.contrast.isin(['F_tf250_minus_tf1000', 'F_tf2500_minus_tf1000'])
                           & (key.condition == 'dynamic')]
    return d


def export_source_data(ex, d):
    """Write the twelve manuscript-v6 Figure 5 source-data files unchanged in schema/order."""
    ex.csv('Fig5B_clamp_time_courses', d['b_time'])
    ex.csv('Fig5B_clamp_contrasts', d['b_contrasts'])
    ex.csv('Fig5C_restoring_field', d['c_field'])
    ex.csv('Fig5C_small_offset_gain_summary', d['c_gain_summary'])
    ex.csv('Fig5C_small_offset_gain_by_trial', d['c_gain_trials'])
    ex.csv('Fig5D_full_11p25_kinematics_band', d['d_band'])
    ex.csv('Fig5D_full_11p25_kinematics_by_trial', d['d_trials'])
    ex.csv('Fig5E_full_60pA_relative_vs_absolute', d['e_curves'])
    ex.csv('Fig5E_late_slopes', d['e_slopes'])
    ex.csv('Fig5F_tau_f_trials', d['f_trials'])
    ex.csv('Fig5F_tau_f_summary', d['f_summary'])
    ex.csv('Fig5F_tau_f_contrasts', d['f_contrasts'])


def load_source_data(source, window=None):
    """Rebuild the Figure 5 panel dictionary from its archived source-data directory."""
    source = Path(source)
    missing = [name for name in SOURCE_FILES if not (source/name).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing Fig. 5 source data in {source}: {missing}')
    return {
        'b_time': pd.read_csv(source/'Fig5B_clamp_time_courses.csv'),
        'b_contrasts': pd.read_csv(source/'Fig5B_clamp_contrasts.csv'),
        'c_field': pd.read_csv(source/'Fig5C_restoring_field.csv'),
        'c_gain_summary': pd.read_csv(source/'Fig5C_small_offset_gain_summary.csv'),
        'c_gain_trials': pd.read_csv(source/'Fig5C_small_offset_gain_by_trial.csv'),
        'd_band': pd.read_csv(source/'Fig5D_full_11p25_kinematics_band.csv'),
        'd_trials': pd.read_csv(source/'Fig5D_full_11p25_kinematics_by_trial.csv'),
        'e_curves': pd.read_csv(source/'Fig5E_full_60pA_relative_vs_absolute.csv'),
        'e_slopes': pd.read_csv(source/'Fig5E_late_slopes.csv'),
        'f_trials': pd.read_csv(source/'Fig5F_tau_f_trials.csv'),
        'f_summary': pd.read_csv(source/'Fig5F_tau_f_summary.csv'),
        'f_contrasts': pd.read_csv(source/'Fig5F_tau_f_contrasts.csv'),
    }


# ================================================================== plot layer (v7)
INK, RAW, MID, LIGHT = (vg.NEUTRAL[k] for k in ('ink', 'raw', 'mid', 'light'))


def _placeholder(ax):
    """Panel-a placeholder only; the author will redraw the final schematic."""
    ax.set_axis_off()
    ax.add_patch(Rectangle((.05, .08), .90, .82, transform=ax.transAxes, fill=False,
                           lw=.6, ls=(0, (3, 2)), edgecolor=LIGHT))
    ax.text(.5, .52, 'schematic placeholder', transform=ax.transAxes,
            ha='center', va='center', fontsize=vg.FS['label'], color=MID)
    ax.text(.5, .40, r'$\varphi$   $\mathcal{\psi}_{\mathrm{eff}}$   $\delta_{\mathrm{eff}}$',
            transform=ax.transAxes, ha='center', va='center', fontsize=vg.FS['label'], color=RAW)


def _gauss(x, mu, sigma):
    return np.exp(-.5 * ((x - mu) / sigma) ** 2)


def _schematic(ax):
    ucol, acol, dcol = vg.VARIABLES['u'], vg.VARIABLES['activity'], vg.VARIABLES['drive']
    ax.set_axis_off()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    base, old, psi, phi = .40, .29, .52, .69
    x = np.linspace(.04, .96, 400)
    u = base + .25 * _gauss(x, old, .13)
    act = base + .36 * _gauss(x, phi, .055)
    ax.plot([.04, .96], [base, base], color=MID, lw=vg.LW['axis'], solid_capstyle='butt', zorder=1)
    ax.fill_between(x, base, u, color=ucol, alpha=.16, lw=0, zorder=1)
    ax.plot(x, u, color=ucol, ls=vg.TRACES['u']['ls'], lw=vg.LW['data'], zorder=2)
    ax.fill_between(x, base, act, color=acol, alpha=.20, lw=0, zorder=2)
    ax.plot(x, act, color=acol, lw=vg.LW['data'], zorder=3)
    ax.text(.05, .79, 'retained\nfacilitation', ha='left', va='center', color=ucol,
            fontsize=vg.FS['note'], linespacing=.95)
    ax.text(.04, base - .03, 'position on ring', ha='left', va='top', color=MID, fontsize=vg.FS['note'])
    ax.plot([psi, psi], [base, .88], color=dcol, lw=.8, zorder=4)
    ax.plot([phi, phi], [base + .36, .88], color=acol, lw=.6, ls=(0, (1, 1.2)), zorder=4)
    ax.text(psi, base - .03, vg.psi(r'\mathrm{eff}'), ha='center', va='top', color=dcol,
            fontsize=vg.FS['label'])
    ax.text(phi, base - .03, r'$\varphi$', ha='center', va='top', color=acol, fontsize=vg.FS['label'])
    ax.annotate('', xy=(psi, .86), xytext=(phi, .86),
                arrowprops=dict(arrowstyle='<->', lw=.7, color=INK, shrinkA=0, shrinkB=0,
                                mutation_scale=6))
    ax.text((psi + phi) / 2, .895, r'$\delta_{\mathrm{eff}}$ (restored)', ha='center', va='bottom',
            color=INK, fontsize=vg.FS['note'])
    ax.annotate('', xy=(.30, .235), xytext=(.90, .235),
                arrowprops=dict(arrowstyle='<->', lw=.7, color=MID, shrinkA=0, shrinkB=0,
                                mutation_scale=6))
    ax.text(.60, .19, 'joint shift of $\\varphi$ and $\\mathcal{\\psi}_{\\mathrm{eff}}$:\nneutral mode', ha='center',
            va='top', color=INK, fontsize=vg.FS['note'], linespacing=1.0)


def _top_legend(fig):
    """Shared orthogonal key: model = colour/shape/line style; clamp state = marker fill/weight."""
    handles = []
    labels = []
    for key in ('full', 'output', 'release'):
        enc = vg.MODELS[key]
        handles.append(Line2D([], [], color=enc['color'], ls=enc['ls'], lw=vg.LW['data'],
                              marker=enc['marker'], markersize=3.8,
                              markerfacecolor=enc['color'], markeredgecolor=enc['color']))
        labels.append(enc['label'])
    handles += [
        Line2D([], [], color=INK, lw=1.25, marker='o', markersize=3.5,
               markerfacecolor=INK, markeredgecolor=INK),
        Line2D([], [], color=MID, lw=.65, marker='o', markersize=3.5,
               markerfacecolor='white', markeredgecolor=MID),
    ]
    labels += [r'$u$ clamp', 'free']
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.39, .985), ncol=5,
               columnspacing=.95, handlelength=1.5, handletextpad=.4, borderaxespad=0,
               fontsize=vg.FS['legend'])


def _panel_b(ax, d):
    """Clamp-set restoration F(t): clamped = filled markers, heavier line, 95% CI shading; free = open markers."""
    for cond in COND_ORDER:
        enc = COND_ENC[cond]
        for mode in ('normal', 'clamp'):
            g = d['b_time'][(d['b_time'].condition == cond) & (d['b_time']['mode'] == mode)].sort_values('t_ms')
            clamp = mode == 'clamp'
            if clamp:
                ax.fill_between(g.t_ms, g.ci_low, g.ci_high, color=enc['color'], alpha=.10, linewidth=0, zorder=1)
            markevery = [2, 10, 18] if len(g) >= 19 else None
            ax.plot(g.t_ms, g['mean'], color=enc['color'], ls=enc['ls'], lw=1.35 if clamp else .8,
                    alpha=1.0 if clamp else .8, marker=enc['marker'], markersize=4.6 if clamp else 4.0,
                    markerfacecolor=enc['color'] if clamp else 'white', markeredgecolor=enc['color'],
                    markeredgewidth=.8, markevery=markevery, zorder=3 if clamp else 2)
    vg.zero_line(ax)
    ax.text(.03, .97, r'$T_\mathrm{w} = 150$ ms', transform=ax.transAxes, ha='left', va='top',
            fontsize=vg.FS['note'], color=MID)
    ax.set(xlim=(135, 615), xticks=[200, 300, 400, 500, 600], ylim=(-.13, .82), yticks=[0, .2, .4, .6, .8],
           xlabel='Time after shift (ms)', ylabel=r'Restoration fraction $F_\mathrm{rest}(t)$')
    vg.g_ticks(ax)


def _panel_b_strip(ax, d, cond, show_x=False, show_y=False):
    """Candidate-B small multiple for one model; source data and statistics unchanged."""
    enc = COND_ENC[cond]
    for mode in ('normal', 'clamp'):
        g = d['b_time'][(d['b_time'].condition == cond) & (d['b_time']['mode'] == mode)].sort_values('t_ms')
        clamp = mode == 'clamp'
        if clamp:
            ax.fill_between(g.t_ms, g.ci_low, g.ci_high, color=enc['color'], alpha=.075, linewidth=0, zorder=1)
        markevery = [2, 10, 18] if len(g) >= 19 else None
        ax.plot(g.t_ms, g['mean'], color=enc['color'], ls=enc['ls'],
                lw=1.25 if clamp else .65, alpha=1.0 if clamp else .72,
                marker=enc['marker'], markersize=3.0 if clamp else 2.7,
                markerfacecolor=enc['color'] if clamp else 'white',
                markeredgecolor=enc['color'], markeredgewidth=.65, markevery=markevery,
                zorder=3 if clamp else 2)
    ax.axhline(0, color='#D9D9D9', lw=.55, zorder=0)
    ax.set(xlim=(150, 600), xticks=[200, 300, 400, 500, 600],
           ylim=(-.13, .80), yticks=[0, .4, .8])
    if show_x:
        ax.set_xlabel('Time after shift (ms)')
    else:
        vg.hide_x(ax, spine=True)
    if not show_y:
        ax.tick_params(axis='y', labelleft=False)
    vg.marker_key(ax, .035, .82, enc, enc['label'], transform=ax.transAxes, gap=.065, size=3.7)
    vg.offset_spines(ax, ('left',) if not show_x else ('left','bottom'))
    vg.g_ticks(ax)


def _panel_c(ax, d):
    lows = []
    for cond in COND_ORDER:
        enc = COND_ENC[cond]
        g = d['c_field'][(d['c_field'].condition == cond) & (d['c_field']['mode'] == 'clamp')].sort_values('abs_delta')
        n = d['c_field'][(d['c_field'].condition == cond) & (d['c_field']['mode'] == 'normal')].sort_values('abs_delta')
        ax.plot(n.abs_delta, n['mean'], color=enc['color'], ls=enc['ls'], lw=.65, alpha=.68,
                marker=enc['marker'], markersize=2.7, markerfacecolor='white',
                markeredgecolor=enc['color'], markeredgewidth=.6, zorder=1)
        ax.errorbar(g.abs_delta, g['mean'], yerr=[g['mean']-g.ci_low, g.ci_high-g['mean']],
                    color=enc['color'], ls=enc['ls'], lw=1.2, marker=enc['marker'], markersize=3.6,
                    markerfacecolor=enc['color'], markeredgecolor=enc['color'], markeredgewidth=.5,
                    capsize=1.8, elinewidth=.7, capthick=.7, zorder=3)
        lows.append(float(g.ci_low.min()))
    ax.axhline(0, color='#D9D9D9', lw=.55, zorder=0)
    ax.set(xlim=(0, 47), xticks=[0, 10, 20, 30, 40], ylim=(-40, 65),
           yticks=[-40, -20, 0, 20, 40, 60], xlabel=r'Imposed mismatch $|\Delta\theta|$ (°)',
           ylabel=r'Restoring velocity (° s$^{-1}$)')
    vg.offset_spines(ax); vg.g_ticks(ax)


def _panel_d(ax, d, markers=True, ax_sub=None):
    """Displacement of phi_s, psi_eff, psi_u, psi_D from the old position (profile-snapshot set),
    with optional lower strip ``ax_sub`` showing delta_eff = d_ax(phi_s, psi_eff)."""
    names = {'x': r'$\varphi_s$ (activity)', 'psi_eff': vg.psi(r'\mathrm{eff}') + ' (reference)',
             'psi_u': vg.psi('u') + ' (facilitation)', 'psi_D': vg.psi('D') + ' (depletion)'}
    handles = []
    for trace in ('x', 'psi_eff', 'psi_u', 'psi_D'):
        enc = TRACE_ENC[trace]
        g = d['d_band'][d['d_band'].trace == trace].sort_values('t_ms')
        if trace in ('x', 'psi_eff'):
            ax.fill_between(g.t_ms / 1000, g.ci_low, g.ci_high, color=enc['color'], alpha=.12, linewidth=0, zorder=1)
        kw = dict(color=enc['color'], ls=enc['ls'], lw=enc.get('lw', 1.3), alpha=enc.get('alpha', 1.0))
        if trace == 'x':
            kw['solid_capstyle'] = 'butt'
        if markers:
            kw.update(marker=enc['marker'], markersize=4.2, markerfacecolor=vg.marker_face(enc),
                      markeredgecolor=enc['color'], markeredgewidth=.8)
        z = 2 if trace == 'psi_D' else (3 if trace == 'x' else (4 if trace == 'psi_u' else 5))
        ax.plot(g.t_ms / 1000, g.mean_deg, markevery=[24, 74, 123] if markers else None, zorder=z, **kw)
        handles.append(Line2D([], [], label=names[trace], **kw))
    ax.axhline(11.25, ls=':', lw=vg.LW['ref'], color=MID, zorder=0)
    vg.right_tag(ax, 11.25, r'$|\Delta\theta|$')
    ax.set(xlim=(0, 2.5), xticks=[0, .5, 1, 1.5, 2, 2.5], ylim=(-.6, 12.6), yticks=[0, 3, 6, 9, 12],
           xlabel='Time after shift (s)', ylabel='Displacement from\nold position (°)' if ax_sub is not None else 'Displacement from old position (°)')
    ax.legend(handles=handles, loc='lower right', ncol=1, handlelength=2.4, labelspacing=.25, borderaxespad=.3)
    vg.g_ticks(ax, 'xy')
    if ax_sub is not None:
        vg.hide_x(ax)
        band = d['d_band']
        g_de = band[band.trace == 'delta_eff'].sort_values('t_ms')
        if g_de.empty and d.get('d_trials') is not None:
            kl_b = d['d_trials'].copy()
            kl_b['delta_eff'] = -kl_b['drive_offset']
            t, m, lo, hi, n = seed_mean_band(kl_b, 'delta_eff', group='t_ms')
            g_de = pd.DataFrame({'t_ms': t, 'mean_deg': m, 'ci_low': lo, 'ci_high': hi, 'n_trials': n})
        t_s = g_de.t_ms.to_numpy(float) / 1000
        dcol = vg.VARIABLES['drive']
        ax_sub.axhspan(-.5, .5, color='#E7E7E7', lw=0, zorder=0)
        vg.zero_line(ax_sub)
        ax_sub.fill_between(t_s, g_de.ci_low, g_de.ci_high, color=dcol, alpha=.16, linewidth=0, zorder=1)
        ax_sub.plot(t_s, g_de.mean_deg, color=dcol, ls='-', lw=1.2, zorder=3)
        ax_sub.set(xlim=(0, 2.5), xticks=[0, .5, 1, 1.5, 2, 2.5],
                   ylim=(-.7, max(2.8, 1.1 * float(g_de.ci_high.max()))), yticks=[0, 2],
                   xlabel='Time after shift (s)', ylabel=r'$\delta_\mathrm{eff}$ (°)')
        vg.g_ticks(ax_sub, 'xy')


def _panel_e(ax1, ax2, d):
    g = d['e_curves'].sort_values('t_s')
    ax1.plot(g.t_s, g.var_delta_eff_running_mean_0p5s_deg2,
             color=vg.VARIABLES['activity'], lw=vg.LW['data'])
    ax2.plot(g.t_s, g.msd_phi_deg2, color=vg.VARIABLES['activity'], lw=vg.LW['data'])
    ax1.set(xlim=(0, 10), xticks=[0, 2, 4, 6, 8, 10], ylim=(0, .42), yticks=[0, .2, .4],
            ylabel='Var($\\delta_{\\mathrm{eff}}$)\n(deg$^2$)')
    ax2.set(xlim=(0, 10), xticks=[0, 2, 4, 6, 8, 10], ylim=(0, 300), yticks=[0, 100, 200, 300],
            xlabel='Elapsed time (s)', ylabel='MSD($\\varphi_s$)\n(deg$^2$)')
    vg.hide_x(ax1, spine=True)
    vg.offset_spines(ax1, ('left',)); vg.offset_spines(ax2)
    vg.g_ticks(ax1); vg.g_ticks(ax2)


def _panel_f(ax, d):
    tfs = np.array([250., 1000., 2500.])
    specs = {
        11.25: {'marker': 'o', 'ls': '-', 'mfc': vg.MODELS['full']['color'], 'scale': .955},
        22.5: {'marker': '^', 'ls': '--', 'mfc': 'white', 'scale': 1.047},
    }
    trials = d['f_trials']
    summary = d['f_summary']
    for delta in (11.25, 22.5):
        sp = specs[delta]
        pv = (trials[np.isclose(trials.abs_delta, delta)]
              .pivot(index='seed', columns='tf', values='F_2s').reindex(columns=tfs))
        if pv.isna().any().any() or len(pv) != 8:
            raise ValueError('Incomplete timescale-scaling trials')
        xs = tfs * sp['scale']
        for _, row in pv.iterrows():
            ax.plot(xs, row.to_numpy(float), color=LIGHT, ls=sp['ls'], lw=.48, alpha=.72, zorder=1)
            ax.scatter(xs, row.to_numpy(float), s=8, marker=sp['marker'], facecolor='white',
                       edgecolor=MID, linewidth=.45, alpha=.75, zorder=2)
        g = summary[np.isclose(summary.abs_delta_deg, delta)].set_index('tau_f_ms').loc[tfs]
        m = g.mean_F_2s.to_numpy(float); lo = g.ci_low.to_numpy(float); hi = g.ci_high.to_numpy(float)
        ax.errorbar(xs, m, yerr=[m-lo, hi-m], color=vg.MODELS['full']['color'], ls=sp['ls'], lw=1.15,
                    marker=sp['marker'], markersize=4.1, markerfacecolor=sp['mfc'],
                    markeredgecolor=vg.MODELS['full']['color'], markeredgewidth=.8,
                    capsize=1.8, elinewidth=.7, capthick=.7, zorder=4)
    ax.axhline(0, color='#D9D9D9', lw=.55, zorder=0)
    ax.set_xscale('log')
    ax.set(xlim=(200, 3100), ylim=(-.22, .66), yticks=[-.2, 0, .2, .4, .6],
           xlabel=r'$\tau_f$ (ms); $\tau_r$ = 20 ms', ylabel=r'$F_{\mathrm{rest}}(2\,\mathrm{s})$')
    ax.set_xticks(tfs, ['250', '1,000', '2,500'])
    ax.xaxis.set_minor_formatter(ScalarFormatter())
    ax.tick_params(axis='x', which='minor', bottom=False, labelbottom=False)
    handles = [
        Line2D([], [], color=vg.MODELS['full']['color'], ls='-', marker='o', markersize=4,
               markerfacecolor=vg.MODELS['full']['color'], label='11.25°'),
        Line2D([], [], color=vg.MODELS['full']['color'], ls='--', marker='^', markersize=4,
               markerfacecolor='white', markeredgecolor=vg.MODELS['full']['color'], label='22.5°'),
    ]
    ax.legend(handles=handles, loc='upper left', ncol=1, borderaxespad=.3)
    vg.offset_spines(ax); vg.g_ticks(ax)


def draw_figure(d, panel_b_variant='A', panel_a='schematic'):
    if panel_b_variant not in ('A', 'B'):
        raise ValueError("panel_b_variant must be 'A' or 'B'")
    if panel_a not in ('schematic', 'placeholder'):
        raise ValueError("panel_a must be 'schematic' or 'placeholder'")
    L = vg.Layout(height_mm=134.)
    fig = L.fig

    x1, x2, x3 = 13., 68., 127.
    w1, w2, w3 = 40., 44., 43.
    y_top, h_top = 82., 41.
    y_bot, h_bot = 12., 49.

    axa = L.axes(x1, y_top, w1, h_top)
    (_schematic if panel_a == 'schematic' else _placeholder)(axa)

    if panel_b_variant == 'A':
        axb = L.axes(x2, y_top, w2, h_top)
        _panel_b(axb, d)
    else:
        bys = [111.5, 97.0, 82.5]
        bax = []
        for i, (cond, yy) in enumerate(zip(COND_ORDER, bys)):
            a = L.axes(x2, yy, w2, 10.5, sharex=bax[0] if bax else None, sharey=bax[0] if bax else None)
            bax.append(a)
            _panel_b_strip(a, d, cond, show_x=(i == 2), show_y=True)
        L.text(x2-8.0, y_top+h_top/2, r'Restoration fraction $F_\mathrm{rest}(t)$', rotation=90,
               ha='center', va='center')

    axc = L.axes(x3, y_top, w3, h_top)
    _panel_c(axc, d)

    axd = L.axes(x1, y_bot, w1, h_bot)
    _panel_d(axd, d)

    axe1 = L.axes(x2, 39., w2, 22.)
    axe2 = L.axes(x2, y_bot, w2, 20., sharex=axe1)
    _panel_e(axe1, axe2, d)

    axf = L.axes(x3, y_bot, w3, h_bot)
    _panel_f(axf, d)

    _top_legend(fig)

    for s, x, y in [('a', 3., 126.5), ('b', x2-10., 126.5), ('c', x3-10., 126.5),
                    ('d', 3., 66.), ('e', x2-10., 66.), ('f', x3-10., 66.)]:
        L.letter(x, y, panel_letter(s))
    return fig


def make_figure(ds, ex, window=None):
    d = compute_panels(ds, window)
    export_source_data(ex, d)
    with vg.style():
        fig = draw_figure(d)
        ex.save(fig, STEM, CAPTIONS[CAPTION_KEY])
