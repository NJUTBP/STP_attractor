"""Manuscript (CPL revision R1) Supplementary Fig. S10: efficacy statistics beyond the mean.

Panels
  (a) reference point: relative difference from the full model in five drive statistics (T2/P2, 4 trials)
  (b) 42 settings: output-only frequency error against its loss of within-cycle drive modulation
  (c) 42 settings: total E->I drive relative to the full model under mean-efficacy matching
  (d) P2b drive-matched controls (10 paired trials) -- drawn only when the P2b results exist; otherwise the
      figure is a single row of panels (a)-(c) with the legend SUPP_CAPTION_VARIANTS['10_without_p2b']

Inputs (build mode, results tree):
  results/revision_r1/t2_efficacy/per_trial_metrics.csv                    P2 statistics (exact reconstruction)
  results/revision_r1/t2_efficacy/verdict.json                             archived Spearman correlation
  results/revision_r1/p2b_drive_matched/analysis/{per_trial.csv, p2b_verdict.json}   (optional; not smoke)
  source_data/FigS2_theta.csv, source_data/FigS6_phase10_combo_summary.csv frozen tables of Figs. S1 and S3
Replay mode reads only the RevS10* source data (tools/replay_supp_cpl.py); the derived tables of (a)-(c) are
recomputed there from RevS10_efficacy_per_trial.csv and must agree to 1e-9.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

from runtime import context
from data_io import unique
import visual_grammar as vg
from supp_captions_cpl import SUPP_CAPTIONS, SUPP_CAPTION_VARIANTS

HERE = Path(__file__).resolve().parent
STEM = 'FigS10_efficacy_statistics'
CAPTION = SUPP_CAPTIONS[10]
CAPTION_WITHOUT_P2B = SUPP_CAPTION_VARIANTS['10_without_p2b']
PREFIX = 'RevS10'
# Panel (d), the drive-matched controls, is not part of the manuscript figure (their results are reported in the
# text of Sec. S10).  The P2b tables are still read, verified and exported; set True to draw panel (d) again.
SHOW_DRIVE_MATCHED_PANEL = False
MID = vg.NEUTRAL['mid']
REFERENCE = 'tf_1000_tr_20'

# Model without facilitation in either route (gain-rescued depletion); not in the main-text grammar.
NEITHER = {'color': '#D55E00', 'marker': 'v', 'mfc': None, 'ls': (0, (5, 1.5)), 'label': 'Neither'}
ENC = {'full': vg.MODELS['full'], 'release': vg.MODELS['release'], 'mean_release': vg.MODELS['mean_release'],
       'output': vg.MODELS['output'], 'neither': NEITHER, 'static': vg.MODELS['static']}
LABEL_TO_KEY = {'dynamic': 'full', 'u_release_only': 'release', 'mean_u_release': 'mean_release',
                'u_output_only': 'output', 'gain_rescued_depletion': 'neither', 'dynamic_static': 'static'}
SHORT = {'release': 'Release', 'mean_release': 'Mean rel.', 'output': 'Output', 'neither': 'Neither',
         'static': 'Static', 'full': 'Full'}
STATS = [('M2_cv_p', 'Efficacy CV'), ('M4_modulation_depth', 'Cycle modulation'),
         ('M5_release_mean', 'Release per spike'), ('M4_drive_mean', r'Mean contribution $\mathscr{D}$'),
         ('M3_drive_width_deg', 'Profile width')]
LONG = {'release': 'Release-only', 'mean_release': 'Mean-release', 'output': 'Output-only', 'neither': 'Neither',
        'static': 'Static'}
PER_TRIAL_COLUMNS = ['combo_id', 'label', 'seed', 'f_proxy_hz', *[c for c, _ in STATS]]
P2B_MODELS = [('full', 'full'), ('release_only', 'release'), ('output_only', 'output'), ('static', 'static')]

REPLAY_READ = {
    f'{PREFIX}_efficacy_per_trial.csv': 'per_trial',
    f'{PREFIX}_settings.csv': 'settings',
    f'{PREFIX}A_reference_relative.csv': 'relative',
    f'{PREFIX}B_output_error_vs_modulation.csv': 'scatter',
    f'{PREFIX}B_spearman.csv': 'spearman',
    f'{PREFIX}C_drive_ratio.csv': 'drive_ratio',
}
REPLAY_OPTIONAL = {
    f'{PREFIX}D_p2b_per_trial.csv': 'p2b_per_trial',
    f'{PREFIX}D_p2b_summary.csv': 'p2b_summary',
}


# ============================================================================== derived tables
def reference_relative(pt: pd.DataFrame) -> pd.DataFrame:
    ref = pt[pt.combo_id == REFERENCE].groupby('label')[[c for c, _ in STATS]].mean()
    rows = []
    for label in ('u_release_only', 'mean_u_release', 'u_output_only', 'gain_rescued_depletion'):
        for col, _ in STATS:
            rows.append({'model': LABEL_TO_KEY[label], 'statistic': col,
                         'full_mean': float(ref.loc['dynamic', col]), 'model_mean': float(ref.loc[label, col]),
                         'relative_difference_percent': float(100. * (ref.loc[label, col] / ref.loc['dynamic', col] - 1.))})
    return pd.DataFrame(rows)


def _setting_means(pt):
    return pt.groupby(['combo_id', 'label'])[['f_proxy_hz', 'M4_modulation_depth', 'M4_drive_mean']].mean().unstack('label')


def output_scatter(pt: pd.DataFrame, settings: pd.DataFrame) -> pd.DataFrame:
    g = _setting_means(pt)
    out = pd.DataFrame({
        'combo_id': g.index,
        'abs_frequency_error_hz': (g[('f_proxy_hz', 'u_output_only')] - g[('f_proxy_hz', 'dynamic')]).abs().to_numpy(),
        'abs_modulation_difference': (g[('M4_modulation_depth', 'u_output_only')] -
                                      g[('M4_modulation_depth', 'dynamic')]).abs().to_numpy()})
    out = out.merge(settings, on='combo_id', how='left', validate='one_to_one')
    if out[['tau_facilitation_ms', 'tau_recovery_ms']].isna().any().any():
        raise ValueError('S10(b): every setting needs its time constants')
    return out.sort_values(['tau_facilitation_ms', 'tau_recovery_ms']).reset_index(drop=True)


def drive_ratio(pt: pd.DataFrame) -> pd.DataFrame:
    g = _setting_means(pt)['M4_drive_mean']
    rows = []
    for label in ('u_release_only', 'mean_u_release', 'u_output_only', 'gain_rescued_depletion', 'dynamic_static'):
        r = (g[label] / g['dynamic'])
        rows += [{'combo_id': c, 'model': LABEL_TO_KEY[label], 'drive_ratio': float(v)} for c, v in r.items()]
    return pd.DataFrame(rows)


def settings_table(combo: pd.DataFrame) -> pd.DataFrame:
    return combo[['combo_id', 'tau_facilitation_ms', 'tau_recovery_ms']].sort_values(
        ['tau_facilitation_ms', 'tau_recovery_ms']).reset_index(drop=True)


# ============================================================================== data
def _p2b(ds):
    """P2b tables, or (None, None) while the pre-registered run has not been analysed (smoke output is ignored)."""
    try:
        vpath = ds.path('revision_r1', 'p2b_drive_matched/analysis/p2b_verdict.json')
    except FileNotFoundError:
        return None, None
    v = json.loads(Path(vpath).read_text(encoding='utf-8'))
    if v.get('smoke', False):
        return None, None
    ds.record(vpath, 'P2b pre-registered verdict')
    pt = ds.csv('revision_r1', 'p2b_drive_matched/analysis/per_trial.csv',
                columns=['model', 'seed', 'frequency_hz', 'theta', 'drive_D', 'c1_efficacy'])
    pt = pt[['model', 'seed', 'frequency_hz', 'theta', 'drive_D', 'c1_efficacy']].sort_values(['model', 'seed'])
    unique(pt, ['model', 'seed'], 'P2b per_trial')
    theta = pd.read_csv(ds.record(HERE / 'source_data' / 'FigS2_theta.csv', 'frozen source data of Fig. S1'))
    combo = pd.read_csv(ds.record(HERE / 'source_data' / 'FigS6_phase10_combo_summary.csv',
                                  'frozen source data of Fig. S3'))
    ref = combo[combo.combo_id == REFERENCE].iloc[0]
    c1 = {'full': float(theta[theta.condition == 'dynamic'].dominant_frequency_hz.mean()),
          'static': float(theta[theta.condition == 'static_gain'].dominant_frequency_hz.mean()),
          'release_only': float(ref.u_release_only_frequency_hz), 'output_only': float(ref.u_output_only_frequency_hz)}
    rows = []
    for model, _ in P2B_MODELS:
        s = v['summary'][model]
        cal = v.get('calibrated', {}).get(model, {})
        rows.append({'model': model, 'mean_frequency_hz': float(s['mean_frequency_hz']),
                     'theta_fraction': float(s['theta_fraction']), 'drive_over_target': float(s['drive_over_target']),
                     'c1_mean_frequency_hz': c1[model], 'calibrated_param': cal.get('param', np.nan),
                     'calibration_valid': cal.get('valid', True), 'verdict': v['verdict']['verdict'],
                     'static_label': v['verdict']['static_label']})
    return pt.reset_index(drop=True), pd.DataFrame(rows)


def compute_panels(ds):
    pt = ds.csv('revision_r1', 't2_efficacy/per_trial_metrics.csv', columns=PER_TRIAL_COLUMNS)
    pt = pt[pt.label.isin(LABEL_TO_KEY)][PER_TRIAL_COLUMNS].sort_values(['combo_id', 'label', 'seed'])
    unique(pt, ['combo_id', 'label', 'seed'], 'T2 per_trial_metrics')
    n = pt.groupby(['combo_id', 'label']).size()
    if pt.combo_id.nunique() != 42 or not (n == 4).all():
        raise ValueError('S10: expected 42 settings x 6 models x 4 trials')
    combo = pd.read_csv(ds.record(HERE / 'source_data' / 'FigS6_phase10_combo_summary.csv',
                                  'frozen source data of Fig. S3'))
    settings = settings_table(combo)
    sp = ds.json('revision_r1', 't2_efficacy/verdict.json')['rule_inputs']['spearman_outputonly_abs_ferr_vs_abs_dmod']
    d = {'per_trial': pt.reset_index(drop=True), 'settings': settings,
         'spearman': pd.DataFrame([{'rho': float(sp['rho']), 'p': float(sp['p']), 'n_settings': int(sp['n_settings'])}])}
    d.update(recompute_tables(d))
    d['p2b_per_trial'], d['p2b_summary'] = _p2b(ds)
    return d


def recompute_tables(d):
    return {'relative': reference_relative(d['per_trial']), 'scatter': output_scatter(d['per_trial'], d['settings']),
            'drive_ratio': drive_ratio(d['per_trial'])}


def recompute(d, source):
    """Derived tables recomputed by tools/replay_supp_cpl.py from the frozen per-trial table."""
    t = recompute_tables(d)
    names = {v: Path(k).stem for k, v in REPLAY_READ.items()}
    return {names[k]: v for k, v in t.items()}


def export_source_data(ex, d):
    for name, key in REPLAY_READ.items():
        ex.csv(Path(name).stem, d[key])
    if d.get('p2b_per_trial') is not None:
        for name, key in REPLAY_OPTIONAL.items():
            ex.csv(Path(name).stem, d[key])


def _draw_p2b(d):
    return SHOW_DRIVE_MATCHED_PANEL and d.get('p2b_per_trial') is not None


def caption_for(d):
    return CAPTION if _draw_p2b(d) else CAPTION_WITHOUT_P2B


# ============================================================================== drawing
def _scatter_kw(enc, size=None):
    open_ = enc.get('mfc') == 'none'
    return dict(s=size or vg.MS['scatter'], marker=enc['marker'], facecolors=vg.marker_face(enc),
                edgecolors=enc['color'] if open_ else 'white', linewidths=.8 if open_ else .4, alpha=.92)


def _panel_a(ax, rel):
    """Statistics on the y axis (top to bottom), difference from the full model on x."""
    models = ['release', 'mean_release', 'output', 'neither']
    off = np.linspace(-.27, .27, len(models))
    yy = np.arange(len(STATS))[::-1].astype(float)
    for k, key in enumerate(models):
        q = rel[rel.model == key].set_index('statistic').loc[[c for c, _ in STATS]]
        ax.scatter(q.relative_difference_percent, yy - off[k], zorder=3, **_scatter_kw(ENC[key], vg.MS['scatter'] - 6))
    vg.zero_line(ax, 0., vertical=True)
    for x in (-10., 10.):
        ax.axvline(x, color=MID, ls=':', lw=vg.LW['ref'], zorder=0)
    lo = float(rel.relative_difference_percent.min())
    ax.set(ylim=(-.6, len(STATS) - .4), xlim=(min(-60., 1.08 * lo), 15.), xticks=[-60, -40, -20, 0],
           xlabel='Difference from full (%)')
    # Script D (U+1D49F), the symbol used for D in the text, from matplotlib's bundled STIX fonts;
    # only the $...$ part uses STIX, the rest of each label keeps the figure's sans font.
    ax.set_yticks(yy, [lab for _, lab in STATS], math_fontfamily='stixsans')
    ax.tick_params(axis='y', length=0)
    vg.g_ticks(ax, 'x')


def _tau_r_colors(values):
    levels = np.array(sorted(set(np.round(np.asarray(values, float), 6))))
    import matplotlib.pyplot as plt
    cols = plt.get_cmap('cividis')(np.linspace(.08, .92, len(levels)))
    return levels, ListedColormap(cols)


def _panel_b(L, ax, sc_df, spearman, cbar_box):
    levels, cmap = _tau_r_colors(sc_df.tau_recovery_ms)
    idx = np.searchsorted(levels, np.round(sc_df.tau_recovery_ms.to_numpy(float), 6))
    norm = BoundaryNorm(np.arange(len(levels) + 1) - .5, len(levels))
    sc = ax.scatter(sc_df.abs_modulation_difference, sc_df.abs_frequency_error_hz, c=idx, cmap=cmap, norm=norm,
                    s=vg.MS['scatter'] - 4, marker='^', edgecolors='white', linewidths=.4, zorder=3)
    ax.set(xlim=(0, 1.08 * float(sc_df.abs_modulation_difference.max())),
           ylim=(0, 1.22 * float(sc_df.abs_frequency_error_hz.max())),
           xlabel='|Modulation depth: output-only − full|', ylabel='|Frequency error| of\noutput-only (Hz)')
    r = spearman.iloc[0]
    ax.text(.03, .985, rf'Spearman $r_\mathrm{{s}}$ = {r.rho:.2f}', transform=ax.transAxes, ha='left', va='top',
            fontsize=vg.FS['note'], color=vg.NEUTRAL['ink'])
    cax = L.axes(*cbar_box)
    cb = L.fig.colorbar(sc, cax=cax, ticks=np.arange(len(levels)))
    cb.ax.set_yticklabels([f'{v:g}' for v in levels])
    cb.ax.tick_params(labelsize=vg.FS['tick'], length=1.6, width=vg.LW['axis'])
    cb.outline.set_linewidth(vg.LW['axis'])
    cb.set_label(r'$\tau_\mathrm{r}$ (ms)', fontsize=vg.FS['label'], labelpad=2.)
    vg.g_ticks(ax, 'xy')


def _panel_c(ax, dr):
    """Models on the y axis (top to bottom), total drive relative to the full model on x; bars, medians."""
    models = ['release', 'mean_release', 'output', 'neither', 'static']
    yy = np.arange(len(models))[::-1].astype(float)
    for y, key in zip(yy, models):
        v = dr[dr.model == key].drive_ratio.to_numpy(float)
        ax.scatter(v, y + np.linspace(-.22, .22, v.size), zorder=3, **_scatter_kw(ENC[key], vg.MS['scatter'] - 12))
        ax.plot([np.median(v)] * 2, [y - .3, y + .3], color=vg.NEUTRAL['ink'], lw=1.6, solid_capstyle='butt', zorder=4)
    vg.zero_line(ax, 1., vertical=True)
    ax.set(ylim=(-.6, len(models) - .4), xlim=vg.pad(float(dr.drive_ratio.min()), float(dr.drive_ratio.max()), .08),
           xlabel=r'$\mathscr{D}_\mathrm{model}/\mathscr{D}_\mathrm{full}$')
    ax.xaxis.label.set_math_fontfamily('stixsans')   # script D as in panel (a) and the text
    ax.set_yticks(yy, [LONG[k] for k in models])
    ax.tick_params(axis='y', length=0)
    vg.g_ticks(ax, 'x')


def _panel_d(ax, pt, summ):
    for k, (model, key) in enumerate(P2B_MODELS):
        enc = ENC[key]
        f = pt[pt.model == model].sort_values('seed').frequency_hz.to_numpy(float)
        ax.scatter(k + np.linspace(-.14, .14, f.size), f, zorder=3, **_scatter_kw(enc, vg.MS['scatter'] - 6))
        ax.plot([k - .2, k + .2], [np.nanmean(f)] * 2, color=enc['color'], lw=2.0, solid_capstyle='butt', zorder=4)
        c1 = float(summ[summ.model == model].c1_mean_frequency_hz.iloc[0])
        ax.plot([k + .34], [c1], ls='none', marker='D', mfc='white', mec=vg.NEUTRAL['raw'], mew=.8,
                ms=vg.MS['key'] - .4, zorder=3)
    hi = float(max(pt.frequency_hz.max(), summ.c1_mean_frequency_hz.max()))
    ax.set(xlim=(-.55, len(P2B_MODELS) - .4), ylim=(0, 1.12 * hi), ylabel='Dominant frequency (Hz)')
    ax.set_xticks(range(len(P2B_MODELS)), [SHORT[k] for _, k in P2B_MODELS])
    vg.side_bar(ax, 4., 12., '4–12 Hz')
    vg.categorical(ax)
    vg.g_ticks(ax, 'y')


def draw_figure(d):
    keys = ['full', 'release', 'mean_release', 'output', 'neither', 'static']
    handles = [vg.model_handle(ENC[k], line=False) for k in keys]
    if _draw_p2b(d):
        L = vg.Layout(104)
        TOP_Y, TOP_H, BOT_Y, BOT_H = 60., 30., 13., 30.
        vg.fig_legend(L, handles, 87., 100.6, loc='center', columnspacing=1.7, handlelength=1.2)
        _panel_a(L.axes(33., TOP_Y, 50., TOP_H), d['relative'])
        _panel_b(L, L.axes(103., TOP_Y, 53., TOP_H), d['scatter'], d['spearman'], (159., TOP_Y, 2.4, TOP_H))
        _panel_c(L.axes(33., BOT_Y, 50., BOT_H), d['drive_ratio'])
        _panel_d(L.axes(103., BOT_Y, 64., BOT_H), d['p2b_per_trial'], d['p2b_summary'])
        letters = {'a': (1.5, TOP_Y + TOP_H + 2.4), 'b': (88.5, TOP_Y + TOP_H + 2.4),
                   'c': (1.5, BOT_Y + BOT_H + 2.4), 'd': (88.5, BOT_Y + BOT_H + 2.4)}
    else:
        L = vg.Layout(62)
        Y, H = 13., 36.
        vg.fig_legend(L, handles[1:], 87., 58.6, loc='center', columnspacing=1.7, handlelength=1.2)
        _panel_a(L.axes(28., Y, 34., H), d['relative'])
        _panel_b(L, L.axes(77., Y, 33., H), d['scatter'], d['spearman'], (112.5, Y, 2.2, H))
        _panel_c(L.axes(147., Y, 25., H), d['drive_ratio'])
        letters = {'a': (1.5, Y + H + 2.4), 'b': (65., Y + H + 2.4), 'c': (125., Y + H + 2.4)}
    for k, (x, y) in letters.items():
        L.letter(x, y, k, va='center')
    return L.fig


def build_figure(ds=None, ex=None):
    ds, ex = context(ds, ex)
    d = compute_panels(ds)
    export_source_data(ex, d)
    with vg.style():
        fig = draw_figure(d)
        return ex.save(fig, STEM, caption_for(d))


if __name__ == '__main__':
    build_figure()
