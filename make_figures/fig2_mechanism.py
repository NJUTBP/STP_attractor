"""Main Fig. 2(d)-(f): release-fraction mechanism of the oscillation frequency (former Supplementary Fig. S11).

CPL revision 3 (2026-10-27).  The three panels of the former Supplementary Fig. S11 are drawn in the bottom row of
main Fig. 2, reordered along the mechanism chain of the text:

  (d) E->I drive of one neuron after its spike at the reference frequency        (former S11c)
  (e) frequency F(rho, tau_r, p) of networks with a fixed release fraction rho    (former S11a)
  (f) self-consistent fixed point f = F(u+(f), tau_r, p) against the full model   (former S11b)

Data.  The frozen RevS11* tables in make_suppfigures/source_data (file names unchanged for provenance).  Before
anything is drawn:
  * their SHA256 must equal the "S11" entry of make_suppfigures/source_data/REVISION_SOURCES.json, and
  * the reference points and the drive kernel are recomputed from FigS2_theta.csv, FigS6_phase10_combo_summary.csv
    and the stored kernel parameters and must agree with the frozen tables to 1e-9
    (the same checks tools/replay_supp_cpl.py applied to S11).
No simulation is run and no table is written.

Drawing.  A verbatim port of supp_rev_s11 (_panel_c -> panel_drive_kernel, _panel_a -> panel_fixed_release,
_panel_b -> panel_fixed_point): colours, limits, legends and tick formats are unchanged; the tau_r = 80 ms
marker is set to a downward triangle ('v') and marker sizes in panels (e) and (f) are enlarged so that the
three tau_r series ('o', 's', 'v') are clearly legible and visually distinct from one another and from the
upward triangle ('^') of the output-only model at the reference point.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

import visual_grammar as vg

BASE = Path(__file__).resolve().parent
DEFAULT_MECH_SOURCE = BASE.parent / 'make_suppfigures' / 'source_data'
PREFIX = 'RevS11'
FILES = {
    f'{PREFIX}A_fixed_release.csv': 'cells',
    f'{PREFIX}A_reference_points.csv': 'reference',
    f'{PREFIX}B_fixed_point_sweep.csv': 'fixed_point',
    f'{PREFIX}B_fixed_point_u.csv': 'fixed_point_u',
    f'{PREFIX}C_kernel_parameters.csv': 'kernel_parameters',
    f'{PREFIX}C_drive_kernel.csv': 'kernel',
}
RECOMPUTE_INPUTS = ('FigS2_theta.csv', 'FigS6_phase10_combo_summary.csv')
MANIFEST = 'REVISION_SOURCES.json'
MANIFEST_KEY = 'S11'          # historical key of these tables in REVISION_SOURCES.json

MID = vg.NEUTRAL['mid']
REFERENCE = 'tf_1000_tr_20'
U_REF = 0.4
# NMDA kinetics of Table S1 (config/phase4_causal.toml): x decays with tau_x, ds/dt = -s/tau_NMDA + alpha x (1 - s).
NMDA = {'tau_x_ms': 2.0, 'alpha_per_ms': 0.5, 'tau_nmda_ms': 100.0, 'tau_r_ms': 20.0, 'dt_ms': 0.05, 'n_cycles': 25}
TAU_R_ROWS = (5., 20., 80.)
# tau_r colours: the 6-level cividis scale of Fig. S10(b) (tau_r = 5, 10, 20, 40, 80, 160 ms), levels 0, 2, 4.
# Marker for tau_r = 80 ms uses downward triangle ('v') instead of diamond ('D') to improve shape contrast
# against circle ('o') and square ('s') while avoiding collision with the output-only upward triangle ('^').
# Per-series size adjustments ('ms_pad' for line/legend markers in pt, 's_pad' for scatter area in pt^2)
# enlarge all symbols and give the downward triangle extra area so its optical size matches the circle and square.
_CIVIDIS6 = None
TAU_R_STYLE = {
    5.:  dict(marker='o', ls=':',  level=0, ms_pad=0.6, s_pad=8.),
    20.: dict(marker='s', ls='-',  level=2, ms_pad=0.6, s_pad=8.),
    80.: dict(marker='v', ls='--', level=4, ms_pad=1.3, s_pad=16.),
}


def tau_r_color(tr):
    global _CIVIDIS6
    if _CIVIDIS6 is None:
        import matplotlib.pyplot as plt
        _CIVIDIS6 = plt.get_cmap('cividis')(np.linspace(.08, .92, 6))
    return _CIVIDIS6[TAU_R_STYLE[float(tr)]['level']]


# ============================================================================== derived tables (verbatim from S11)
def reference_tables(theta: pd.DataFrame, combo: pd.DataFrame):
    """Reference-point values of the main text used to mark panel (e)."""
    ref = combo[combo.combo_id == REFERENCE]
    if len(ref) != 1:
        raise ValueError('Fig. S3 summary: the reference setting must appear once')
    ref = ref.iloc[0]
    points = pd.DataFrame([
        {'model': 'output_only', 'rho': U_REF, 'frequency_hz': float(ref.u_output_only_frequency_hz),
         'source': 'Fig. S3 summary, reference setting (release fraction = U)'},
        {'model': 'full', 'rho': float(ref.full_mean_spike_release_u), 'frequency_hz': float(ref.full_frequency_hz),
         'source': 'Fig. S3 summary, reference setting (spike-weighted mean release fraction)'}])
    return points


def kernel_parameters(points: pd.DataFrame, theta: pd.DataFrame) -> pd.DataFrame:
    f_ref = float(theta[theta.condition == 'dynamic'].dominant_frequency_hz.mean())
    rho_full = float(points[points.model == 'full'].rho.iloc[0])
    return pd.DataFrame([{**NMDA, 'frequency_hz': f_ref, 'rho_full': rho_full, 'rho_output': U_REF}])


def drive_kernel(par: pd.DataFrame) -> pd.DataFrame:
    """Steady-state NMDA gating of one neuron that fires once per period (midpoint rule as in the simulator), and
    the drive it carries under continuous-gain coupling, s(t)[1 - rho exp(-t/tau_r)], or event-scaled coupling."""
    p = par.iloc[0]
    T = 1000. / float(p.frequency_hz)
    dt, tx, tn, a = float(p.dt_ms), float(p.tau_x_ms), float(p.tau_nmda_ms), float(p.alpha_per_ms)
    n = int(round(T / dt))
    x = s = 0.
    ss = np.empty(n)
    for _ in range(int(p.n_cycles)):
        x += 1.
        for k in range(n):
            k1 = -s / tn + a * x * (1 - s)
            sh = s + .5 * dt * k1
            xh = x * np.exp(-.5 * dt / tx)
            k2 = -sh / tn + a * xh * (1 - sh)
            x *= np.exp(-dt / tx)
            s = s + dt * k2
            ss[k] = s
    t = np.arange(1, n + 1) * dt
    smax = ss.max()
    rec = np.exp(-t / float(p.tau_r_ms))
    return pd.DataFrame({'t_ms': t, 'event_scaled': ss / smax,
                         'continuous_rho_full': ss * (1 - float(p.rho_full) * rec) / smax,
                         'continuous_rho_output': ss * (1 - float(p.rho_output) * rec) / smax})


# ============================================================================== data + checks
def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _frames_agree(new: pd.DataFrame, old: pd.DataFrame, tol=1e-9) -> str | None:
    """None if equal (numbers to rtol = atol = tol after the CSV round trip), otherwise a short reason."""
    if list(new.columns) != list(old.columns):
        return f'columns {list(new.columns)} != {list(old.columns)}'
    if new.shape != old.shape:
        return f'shape {new.shape} != {old.shape}'
    for col in new.columns:
        a, b = new[col], old[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            if not np.allclose(a.to_numpy(float), b.to_numpy(float), rtol=tol, atol=tol, equal_nan=True):
                return f'column {col} differs beyond {tol:g}'
        elif not (a.astype(str).to_numpy() == b.astype(str).to_numpy()).all():
            return f'column {col} differs'
    return None


def load_mechanism_data(source=None, verify=False) -> dict:
    """Load source tables of Fig. 2(d)-(f) from make_suppfigures/source_data."""
    source = Path(source).resolve() if source else DEFAULT_MECH_SOURCE.resolve()
    missing = [n for n in list(FILES) if not (source / n).is_file()]
    if missing:
        raise FileNotFoundError(f'Fig. 2(d)-(f): missing source tables in {source}: {missing}. '
                                'Point --mechanism-source at make_suppfigures/source_data.')
    d = {key: pd.read_csv(source / name) for name, key in FILES.items()}
    d['source'] = str(source)
    return d


# ============================================================================== drawing (verbatim port of S11)
def _ref_marker(ax, x, y, key, ms=None):
    # CPL revision 4 (C): black edge, like the black rings that mark the reference point in panel (c); it also keeps
    # the large full-model circle apart from the small navy circles of tau_r = 5 ms.
    enc = vg.MODELS[key]
    ax.plot([x], [y], ls='none', marker=enc['marker'], mfc=vg.marker_face(enc), mec=vg.NEUTRAL['ink'],
            mew=.9, ms=ms or vg.MS['line'] + 2.2, zorder=5)


def panel_fixed_release(ax, cells, ref):
    """(e)  former S11a."""
    handles = []
    for tr in TAU_R_ROWS:
        c = cells[np.isclose(cells.tau_r, tr)].sort_values('rho')
        st, col = TAU_R_STYLE[tr], tau_r_color(tr)
        ax.plot(c.rho, c.freq_hz, color=col, ls=st['ls'], marker=st['marker'], mfc=col, mec=col, mew=.6,
                ms=vg.MS['line'] + st['ms_pad'], lw=vg.LW['data'] - .1, zorder=3)
        handles.append(Line2D([], [], color=col, ls=st['ls'], marker=st['marker'], mfc=col, mec=col,
                              ms=vg.MS['key'] + st['ms_pad'], lw=vg.LW['data'] - .1, label=rf'$\tau_\mathrm{{r}}$ = {tr:g} ms'))
    for _, r in ref.iterrows():
        _ref_marker(ax, r.rho, r.frequency_hz, 'output' if r.model == 'output_only' else 'full')
    hi = float(cells.freq_hz.max())
    ax.set(xlim=(.25, 1.), xticks=[.3, .5, .7, .9], ylim=(0, 1.08 * hi),
           xlabel=r'Fixed release fraction $\rho$', ylabel='Dominant frequency (Hz)')
    ax.legend(handles=handles, loc='lower left', handlelength=1.8, labelspacing=.3)
    vg.side_bar(ax, 4., 12., '4–12 Hz')
    vg.g_ticks(ax, 'xy')


def panel_fixed_point(ax, fp, fu):
    """(f)  former S11b."""
    ok = np.isfinite(fp.f_pred_hz)
    for tr in TAU_R_ROWS:
        q = fp[ok & np.isclose(fp.tau_r_ms, tr)]
        st, col = TAU_R_STYLE[tr], tau_r_color(tr)
        ax.scatter(q.f_full_hz, q.f_pred_hz, s=vg.MS['scatter'] + st['s_pad'], marker=st['marker'], facecolors=col,
                   edgecolors='white', linewidths=.45, zorder=3)
    okb = np.isfinite(fu.f_pred_fixed_point_hz)
    ax.scatter(fu.f_full_hz[okb], fu.f_pred_fixed_point_hz[okb], s=vg.MS['scatter'] + 16., marker='s',
               facecolors='none', edgecolors=vg.NEUTRAL['ink'], linewidths=.9, zorder=4)
    lim = (3.5, 11.5)
    ax.plot(lim, lim, color=MID, ls=':', lw=vg.LW['ref'], zorder=0)
    ax.set(xlim=lim, ylim=lim, xticks=[4, 6, 8, 10], yticks=[4, 6, 8, 10], xlabel='Full-model frequency (Hz)',
           ylabel='Fixed point\n' + r'$f = F[u^+(f)]$ (Hz)')
    # CPL revision 4 (C): one key for both symbol sets; the filled sweep markers were explained by free text before.
    from matplotlib.legend_handler import HandlerTuple
    sweep = tuple(Line2D([], [], ls='none', marker=TAU_R_STYLE[tr]['marker'], mfc=tau_r_color(tr), mec='white',
                         mew=.4, ms=vg.MS['key'] + TAU_R_STYLE[tr]['ms_pad']) for tr in TAU_R_ROWS)
    u_sweep = Line2D([], [], ls='none', marker='s', mfc='none', mec=vg.NEUTRAL['ink'], ms=vg.MS['key'] + .8, mew=.9)
    ax.legend(handles=[sweep, u_sweep], labels=[r'$(\tau_\mathrm{f}, \tau_\mathrm{r})$ sweep', '$U$ sweep'],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=.25)}, loc='upper left', handlelength=2.6)
    vg.g_ticks(ax, 'xy')


def panel_drive_kernel(ax, k):
    """(d)  former S11c."""
    full, out = vg.MODELS['full'], vg.MODELS['output']
    ax.plot(k.t_ms, k.event_scaled, color=vg.NEUTRAL['raw'], ls='-', lw=vg.LW['data'], label='No depletion')
    ax.plot(k.t_ms, k.continuous_rho_output, color=out['color'], ls=out['ls'], lw=vg.LW['data'],
            label=r'$\rho$ = 0.4')
    ax.plot(k.t_ms, k.continuous_rho_full, color=full['color'], ls=full['ls'], lw=vg.LW['data'] + .1,
            label=r'$\rho$ = 0.89')
    ax.set(xlim=(0, float(k.t_ms.max())), xticks=[0, 50, 100, 150], ylim=(0, 1.08),
           xlabel='Time after a spike (ms)', ylabel='$c(t)$ of one\nneuron (normalized)')
    # CPL revision 4: the key uses the manuscript font like every other label (it was forced to DejaVu Sans).
    ax.legend(loc='upper right', ncol=1, fontsize=vg.FS['legend'],
              handlelength=1.4, handletextpad=.4, labelspacing=.15, borderpad=.25)
    vg.g_ticks(ax, 'xy')


def draw_row(L, m, boxes, letter_y, letter_x):
    """Bottom row of Fig. 2.  boxes / letter_x: dicts keyed 'd', 'e', 'f' (mm, bottom-left origin)."""
    panel_drive_kernel(L.axes(*boxes['d']), m['kernel'])
    panel_fixed_release(L.axes(*boxes['e']), m['cells'], m['reference'])
    panel_fixed_point(L.axes(*boxes['f']), m['fixed_point'], m['fixed_point_u'])
    for key in ('d', 'e', 'f'):
        L.letter(letter_x[key], letter_y, key, va='center')