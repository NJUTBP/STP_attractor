"""Manuscript (CPL revision R1) Supplementary Fig. S4: parameter sensitivity within the model.

Panels
  (a) dominant frequency against the baseline release probability U (P3 U sweep, tau_f = 1 s, tau_r = 20 ms)
  (b) absolute frequency error of the route-isolated models and the mean-release control
  (c) F_rest(2 s) at U = 0.3 and 0.5 (P3 restoration stage) with the confirmation-set means at U = 0.4
  (d) F_rest(2 s) of the full model in the timescale-scaling set (Phase-16 module D)

Inputs (build mode, results tree):
  results/revision_r1/p3_u_sensitivity/analysis/p3_per_u.csv          P3 frequency stage (prespecified)
  results/revision_r1/p3_u_sensitivity/analysis/p3_restoration.csv    P3 restoration stage
  source_data/FigS12_seed_metrics.csv                                 frozen per-trial values of Fig. S4
  bundled_data/phase16/module_D                                       identity-checked module-D tables
Replay mode reads only the RevS9* source data written by export_source_data (tools/replay_supp_cpl.py).
Nothing here re-estimates an archived inferential statistic: means and CIs are drawn as archived.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

from runtime import context
import visual_grammar as vg
from supp_captions_cpl import SUPP_CAPTIONS

HERE = Path(__file__).resolve().parent
STEM = 'FigS4_parameter_sensitivity'
CAPTION = SUPP_CAPTIONS[4]
PREFIX = 'RevS9'
MID = vg.NEUTRAL['mid']

# Frozen source data read by tools/replay_supp_cpl.py: {file: key in the drawing dictionary}.
REPLAY_READ = {
    f'{PREFIX}A_u_sweep.csv': 'u_sweep',
    f'{PREFIX}C_u_restoration.csv': 'restoration',
    f'{PREFIX}C_confirmation_reference.csv': 'confirmation',
    f'{PREFIX}D_timescales.csv': 'timescales',
    f'{PREFIX}D_key_contrasts.csv': 'key_contrasts',
}
U_COLUMNS = ['u', 'full_frequency_hz', 'u_release_only_frequency_hz', 'mean_u_release_frequency_hz',
             'u_output_only_frequency_hz', 'static_frequency_hz', 'release_abs_error_hz',
             'mean_release_abs_error_hz', 'output_abs_error_hz', 'full_theta_fraction', 'p_match',
             'mean_release_u', 'interpretable', 'calibration_valid']
REST_COLUMNS = ['u', 'condition', 'contrast', 'F2s_mean', 'ci_low', 'ci_high', 'n']   # p3_restoration.csv
ROUTES = [('dynamic', 'full'), ('u_output_only', 'output'), ('u_release_only', 'release')]
FREQ = [('static', 'static_frequency_hz'), ('output', 'u_output_only_frequency_hz'),
        ('mean_release', 'mean_u_release_frequency_hz'), ('release', 'u_release_only_frequency_hz'),
        ('full', 'full_frequency_hz')]
ERRORS = [('output', 'output_abs_error_hz'), ('mean_release', 'mean_release_abs_error_hz'),
          ('release', 'release_abs_error_hz')]


# ============================================================================== data
def confirmation_reference(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    """Means of F_rest(2 s) after 11.25-deg activity-only shifts in the confirmation set (Fig. S4 source data)."""
    rows = []
    for cond, _ in ROUTES:
        q = seed_metrics[(seed_metrics.condition == cond) & (seed_metrics.shift_kind == 'activity_shift') &
                         np.isclose(seed_metrics.abs_delta_deg, 11.25)]
        if len(q) != 16 or q.seed.duplicated().any():
            raise ValueError(f'Confirmation set: expected 16 distinct trials for {cond}, found {len(q)}')
        rows.append({'u': 0.4, 'condition': cond, 'F2s_mean': float(q.R_2000ms.mean()), 'n': int(len(q))})
    return pd.DataFrame(rows)


def compute_panels(ds):
    u = ds.csv('revision_r1', 'p3_u_sensitivity/analysis/p3_per_u.csv', columns=U_COLUMNS)[U_COLUMNS]
    u = u.sort_values('u').reset_index(drop=True)
    if not (u.interpretable.astype(bool).all() and u.calibration_valid.astype(bool).all()):
        raise ValueError('P3: every U value must be interpretable with valid calibrations before it is drawn')
    rest = ds.csv('revision_r1', 'p3_u_sensitivity/analysis/p3_restoration.csv', columns=REST_COLUMNS)
    rest = rest.assign(row=rest.condition.where(rest.condition.notna(), rest.contrast))
    rest = rest[['u', 'row', 'F2s_mean', 'ci_low', 'ci_high', 'n']].sort_values(['u', 'row']).reset_index(drop=True)
    if rest.row.isna().any():
        raise ValueError('P3 restoration: every row needs a condition or a contrast label')
    sm_path = ds.record(HERE / 'source_data' / 'FigS12_seed_metrics.csv',
                        'frozen source data of Fig. S4 (confirmation set)')
    conf = confirmation_reference(pd.read_csv(sm_path))
    import phase16_data
    st = phase16_data.attach(ds)
    key = st.table('D', 'P6_key_contrasts')                      # triggers the module-D identity check
    mci_path = st.module_dir('D') / 'analysis' / 'D_restoration_2s_mean_ci.csv'
    ds.record(mci_path, st.identity)
    mci = pd.read_csv(mci_path)
    ts = mci[(mci.condition == 'dynamic') & (mci.kind == 'activity_shift') & np.isclose(mci.abs_delta, 11.25)]
    ts = ts[['tf', 'tr', 'mean', 'ci_low', 'ci_high', 'n']].sort_values(['tf', 'tr']).reset_index(drop=True)
    if len(ts) != 5:
        raise ValueError(f'Module D: expected five (tf, tr) settings, found {len(ts)}')
    kc = key[(key.condition == 'dynamic') & np.isclose(key.abs_delta, 11.25) & key.family.isin(['tau_f', 'tau_r'])]
    kc = kc[['family', 'contrast', 'mean', 'ci_low', 'ci_high', 'n_pairs']].reset_index(drop=True)
    return {'u_sweep': u, 'restoration': rest, 'confirmation': conf, 'timescales': ts, 'key_contrasts': kc}


def export_source_data(ex, d):
    for name, key in REPLAY_READ.items():
        ex.csv(Path(name).stem, d[key])


def recompute(d, source):
    """Tables that tools/replay_supp_cpl.py recomputes from other frozen files (must agree to 1e-9)."""
    seed = pd.read_csv(Path(source) / 'FigS12_seed_metrics.csv')
    return {f'{PREFIX}C_confirmation_reference': confirmation_reference(seed)}


# ============================================================================== drawing
def _line(ax, x, y, key, **kw):
    enc = vg.MODELS[key]
    return ax.plot(x, y, color=enc['color'], ls=enc['ls'], marker=enc['marker'], mfc=vg.marker_face(enc),
                   mec=enc['color'], mew=.8, ms=kw.pop('ms', vg.MS['line']), lw=kw.pop('lw', vg.LW['data']), **kw)


def _panel_a(ax, u):
    for key, col in FREQ:
        _line(ax, u.u, u[col], key, zorder=4 if key == 'full' else 3,
              ms=vg.MS['line'] - (.6 if key in ('release', 'mean_release') else 0))
    vg.zero_line(ax, .4, vertical=True)
    ax.set(xlim=(.17, .63), xticks=[.2, .3, .4, .5, .6], ylim=(0, 24), yticks=[0, 5, 10, 15, 20],
           xlabel='Baseline utilization $U$', ylabel='Dominant frequency (Hz)')
    vg.side_bar(ax, 4., 12., '4–12 Hz')
    vg.g_ticks(ax, 'xy')


def _panel_b(ax, u):
    for key, col in ERRORS:
        _line(ax, u.u, u[col], key, zorder=3, ms=vg.MS['line'] - (.4 if key != 'output' else 0))
    for y, ls, tag in ((1., '--', '1 Hz'), (2., ':', '2 Hz')):
        ax.axhline(y, color=MID, ls=ls, lw=vg.LW['ref'], zorder=0)
        vg.right_tag(ax, y, tag)
    top = float(u.output_abs_error_hz.max())
    ax.set(xlim=(.17, .63), xticks=[.2, .3, .4, .5, .6], ylim=(-.04 * 1.12 * top, 1.12 * top),
           yticks=np.arange(0., 1.12 * top, 2.), xlabel='Baseline utilization $U$',
           ylabel='|Error| relative to full (Hz)')
    vg.g_ticks(ax, 'xy')


def _panel_c(ax, rest, conf):
    off = {'full': -.022, 'output': 0., 'release': .022}
    for cond, key in ROUTES:
        enc = vg.MODELS[key]
        r = rest[rest.row == cond].sort_values('u')
        x = r.u.to_numpy(float) + off[key]
        m = r.F2s_mean.to_numpy(float)
        ax.errorbar(x, m, yerr=[m - r.ci_low.to_numpy(float), r.ci_high.to_numpy(float) - m], fmt=enc['marker'],
                    color=enc['color'], mfc=vg.marker_face(enc), mec=enc['color'], mew=.8, ms=vg.MS['line'],
                    elinewidth=.9, capsize=2., capthick=.9, zorder=3)
        c = conf[conf.condition == cond]
        ax.plot(c.u + off[key], c.F2s_mean, ls='none', marker=enc['marker'], mfc='white', mec=enc['color'],
                mew=.9, ms=vg.MS['line'], zorder=3)
    vg.zero_line(ax)
    lo = float(min(rest.ci_low.min(), conf.F2s_mean.min()))
    hi = float(max(rest.ci_high.max(), conf.F2s_mean.max()))
    ax.set(xlim=(.25, .55), xticks=[.3, .4, .5], ylim=vg.pad(lo, hi, .08),
           xlabel='Baseline utilization $U$', ylabel=r'$F_\mathrm{rest}$(2 s), 11.25°')
    vg.g_ticks(ax, 'xy')


def _panel_d(ax, ts):
    enc = vg.MODELS['full']
    tf = ts[np.isclose(ts.tr, 20.)].sort_values('tf')
    tr = ts[np.isclose(ts.tf, 1000.)].sort_values('tr')
    groups = [(np.arange(3, dtype=float), tf, tf.tf), (np.arange(3, dtype=float) + 3.7, tr, tr.tr)]
    for x, g, _ in groups:
        m = g['mean'].to_numpy(float)
        # CPL revision 4 (C): no connecting line -- the settings sit on a categorical, non-uniform tau axis.
        ax.errorbar(x, m, yerr=[m - g.ci_low.to_numpy(float), g.ci_high.to_numpy(float) - m], fmt=enc['marker'],
                    color=enc['color'], mfc=vg.marker_face(enc), mec=enc['color'], mew=.8, ms=vg.MS['line'],
                    elinewidth=.9, capsize=2., capthick=.9, zorder=3)
    vg.zero_line(ax)
    ticks = np.concatenate([x for x, _, _ in groups])
    labels = [f'{v:g}' for v in np.concatenate([lab.to_numpy(float) for _, _, lab in groups])]
    ax.set_xticks(ticks, labels)
    lo, hi = float(ts.ci_low.min()), float(ts.ci_high.max())
    ax.set(xlim=(-.5, 6.2), ylim=vg.pad(min(lo, 0.), hi, .08), ylabel=r'$F_\mathrm{rest}$(2 s), full model')
    tr_ = blended_transform_factory(ax.transData, ax.transAxes)
    for xc, text in ((1., r'$\tau_\mathrm{f}$ (ms), $\tau_\mathrm{r}$ = 20 ms'),
                     (4.7, r'$\tau_\mathrm{r}$ (ms), $\tau_\mathrm{f}$ = 1 s')):
        ax.text(xc, -.215, text, transform=tr_, ha='center', va='top', fontsize=vg.FS['label'])
    vg.g_ticks(ax, 'y')


def draw_figure(d):
    """2 x 2 grid on the S1 grid (columns at 19 and 103 mm, 64 mm wide), one shared model key on top."""
    L = vg.Layout(101)
    fig = L.fig
    LX, RX, W = 19., 103., 64.
    TOP_Y, TOP_H = 59., 29.
    BOT_Y, BOT_H = 14., 29.
    vg.fig_legend(L, [vg.model_handle(k) for k in ('full', 'release', 'mean_release', 'output', 'static')],
                  87., 97.6, loc='center', columnspacing=2.0, handlelength=2.6)
    _panel_a(L.axes(LX, TOP_Y, W, TOP_H), d['u_sweep'])
    _panel_b(L.axes(RX, TOP_Y, W, TOP_H), d['u_sweep'])
    _panel_c(L.axes(LX, BOT_Y, W, BOT_H), d['restoration'], d['confirmation'])
    _panel_d(L.axes(RX, BOT_Y, W, BOT_H), d['timescales'])
    for k, (x, y) in {'a': (1.5, TOP_Y + TOP_H + 2.4), 'b': (88.5, TOP_Y + TOP_H + 2.4),
                      'c': (1.5, BOT_Y + BOT_H + 2.4), 'd': (88.5, BOT_Y + BOT_H + 2.4)}.items():
        L.letter(x, y, k, va='center')
    return fig


def build_figure(ds=None, ex=None):
    ds, ex = context(ds, ex)
    d = compute_panels(ds)
    export_source_data(ex, d)
    with vg.style():
        fig = draw_figure(d)
        return ex.save(fig, STEM, CAPTION)


if __name__ == '__main__':
    build_figure()
