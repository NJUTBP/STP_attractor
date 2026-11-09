#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect archived Phase-13 90-pA trials and export one *descriptive* Fig. 1 trial.

Never runs the simulator. Never modifies canonical results, existing source_data,
reference-condition settings, or other figures.

    python make_figures/tools/export_fig1_noise90.py --scan
    python make_figures/tools/export_fig1_noise90.py --seed 100

Use --results-root to override make_figures/settings.py's local RESULTS_ROOT.
The default combo tf_1000_tr_20 is the paper's STP reference *parameter set*;
90 pA is used for this Fig. 1 illustration only, not as the analysis reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import hilbert

MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

import fig1_model
from dynamics import welch_proxy, theta_band
from settings import RESULTS_ROOT

PHASE13 = Path('phase13_attractor_confirmation/combos')
DEFAULT_COMBO = 'tf_1000_tr_20'
NOISE_DIR = 'noise_090'
CONDITION = 'dynamic'
NOISE_PA = 90.
SEED_NAME = 'seed_{:04d}.npz'
REQUIRED = ('spike_times_e_ms', 'spike_angles_e_deg',
            'spike_times_i_ms', 'spike_angles_i_deg',
            'theta_e_deg', 'condition', 'seed')
# E/I spikes are sufficient for raster screening. The inhibitory-current trace
# is a separate observable; it may have been archived over a shorter interval.
# Its PSD is derived only if that interval supports an honest Welch estimate.



def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def raw_directory(results_root: Path, combo: str) -> Path:
    """One explicit, nonrecursive, audit-friendly Phase-13 path."""
    if combo != DEFAULT_COMBO:
        raise ValueError(f'Fig. 1 90-pA representative protocol fixes combo={DEFAULT_COMBO}; got {combo!r}.')
    return Path(results_root).expanduser().resolve() / PHASE13 / combo / NOISE_DIR / CONDITION


def discover_seeds(folder: Path) -> list[int]:
    if not folder.is_dir():
        raise FileNotFoundError(f'Phase-13 90-pA dynamic trial folder not found:\n  {folder}\n'
                                'Set --results-root to the folder containing phase13_attractor_confirmation.')
    seeds = []
    for path in sorted(folder.glob('seed_*.npz')):
        suffix = path.stem.removeprefix('seed_')
        if suffix.isdecimal() and path.name == SEED_NAME.format(int(suffix)):
            seeds.append(int(suffix))
    if not seeds:
        raise FileNotFoundError(f'No seed_NNNN.npz archival records in {folder}')
    return seeds


def _scalar(arr, name, path):
    if np.asarray(arr).size != 1:
        raise ValueError(f'{path}: {name} must be scalar, got shape {np.shape(arr)}')
    value = np.asarray(arr).item()
    return value.decode('utf-8') if isinstance(value, bytes) else value


def load_original_trial(folder: Path, seed: int) -> tuple[dict, dict]:
    """Reject wrong seed/mode/noise and truncated/non-original records.

    The noise folder is explicitly fixed to 090. When the original schema does
    not expose noise_pa, that limitation is stated in provenance; no intensity
    is silently inserted into the original trial arrays.
    """
    path = folder / SEED_NAME.format(seed)
    if not path.is_file():
        raise FileNotFoundError(f'No archived Phase-13 seed {seed}: {path}')
    with np.load(path, allow_pickle=False) as obj:
        data = {k: obj[k] for k in obj.files}
    missing = sorted(set(REQUIRED) - set(data))
    if missing:
        raise ValueError(f'{path}: not a complete Fig. 1 raw recording; missing keys: {missing}. '
                         'A derived summary is not enough to reconstruct spike rasters.')
    got_seed = int(_scalar(data['seed'], 'seed', path))
    condition = str(_scalar(data['condition'], 'condition', path))
    if got_seed != seed or condition != CONDITION:
        raise ValueError(f'{path}: expected seed={seed}, condition=dynamic, got {got_seed}, {condition}.')
    sidecar = path.with_suffix('.json')
    sidecar_meta = json.loads(sidecar.read_text(encoding='utf-8')) if sidecar.is_file() else {}
    noise_values = []
    if 'noise_pa' in data:
        noise_values.append(('raw NPZ', float(_scalar(data['noise_pa'], 'noise_pa', path))))
    if 'noise_pa' in sidecar_meta:
        noise_values.append(('raw sidecar JSON', float(sidecar_meta['noise_pa'])))
    if any(not np.isfinite(value) or not np.isclose(value, NOISE_PA, rtol=0, atol=1e-5)
           for source, value in noise_values):
        raise ValueError(f'{path}: noise metadata {noise_values} disagree with 90 pA.')
    noise_evidence = ('; '.join(source for source, value in noise_values) + '; noise_090 directory'
                      if noise_values else 'noise_090 directory ONLY: no explicit noise_pa in raw/sidecar')
    # The legacy Fig. 1 replay assumes these specific anchor parameters.
    # Reject conflicting metadata rather than replaying a different model as the full model.
    for key, expected in {'tau_f_ms': 1000., 'tau_r_ms': 20., 'U': .4, 'dt_ms': .02}.items():
        values = []
        if key in data:
            values.append(float(_scalar(data[key], key, path)))
        if key in sidecar_meta:
            values.append(float(sidecar_meta[key]))
        if any(not np.isclose(x, expected, atol=1e-7, rtol=1e-7) for x in values):
            raise ValueError(f'{path}: {key} metadata {values} differ from Fig. 1 anchor {expected}.')

    # An archived 90-pA trial may contain a shortened I_inh recording. This
    # is NOT a reason to reject the original spike raster. The current is
    # inventoried separately, with no fabricated samples or extrapolation.
    current_record = describe_current(data)
    angles = np.asarray(data['theta_e_deg'], dtype=float)
    if (angles.shape != (1024,) or not np.isfinite(angles).all()
            or np.any(np.diff(angles) <= 0) or angles[0] < 0 or angles[-1] >= 360):
        raise ValueError(f'{path}: missing/incorrect canonical 1024-cell E ring angles.')
    for population in ('e', 'i'):
        st = np.asarray(data[f'spike_times_{population}_ms'], dtype=float)
        a = np.asarray(data[f'spike_angles_{population}_deg'], dtype=float)
        if not (st.ndim == a.ndim == 1 and len(st) == len(a) and np.isfinite(st).all()
                and np.isfinite(a).all() and ((st >= 0) & (st <= 8000)).all()
                and ((a >= 0) & (a < 360)).all()):
            raise ValueError(f'{path}: invalid {population.upper()} spike times/angles.')
    info = {'raw_path': str(path.resolve()), 'raw_sha256': sha256(path),
            'raw_sidecar_path': str(sidecar.resolve()) if sidecar.is_file() else None,
            'raw_sidecar_sha256': sha256(sidecar) if sidecar.is_file() else None,
            'seed': seed, 'condition': condition, 'noise_pa': NOISE_PA,
            'noise_evidence': noise_evidence, 'combo': DEFAULT_COMBO,
            'current_record': current_record,
            'plot_time_axis_ms': [0, 8000], 'current_trace_is_not_guaranteed_full_8s': True}
    return data, info


def describe_current(raw: dict) -> dict:
    """Explicitly report what Phase-13 archived, independent of spike events.

    Units are taken from the array names; this routine never fabricates a
    0–8 s current trace. A regular, sufficiently sampled >=256-point current
    segment inside 1–8 s can provide an *interval-labelled* Welch PSD.
    """
    keys = set(raw)
    if not {'sample_times_ms', 'plfp_gaba_e'} <= keys:
        return {'available': False,
                'reason': 'missing sample_times_ms and/or plfp_gaba_e',
                'psd_available': False}
    t = np.asarray(raw['sample_times_ms'], dtype=float)
    x = np.asarray(raw['plfp_gaba_e'], dtype=float)
    result = {'available': True, 'n_current_samples': int(len(t)),
              'sample_time_shape': list(t.shape), 'current_shape': list(x.shape)}
    if t.ndim != 1 or x.ndim != 1 or len(t) != len(x) or len(t) < 3:
        return {**result, 'psd_available': False,
                'reason': 'current/time arrays must be 1D and have equal length >=3'}
    if not (np.isfinite(t).all() and np.isfinite(x).all()):
        return {**result, 'psd_available': False,
                'reason': 'non-finite timestamps/current samples'}
    result.update({'recorded_start_ms': float(t[0]),
                   'recorded_end_ms': float(t[-1])})
    dt = np.diff(t)
    if not np.all(dt > 0):
        return {**result, 'psd_available': False,
                'reason': 'timestamps are not strictly increasing'}
    median_dt = float(np.median(dt))
    result['sample_dt_ms'] = median_dt
    if not np.allclose(dt, median_dt, atol=1e-5, rtol=1e-5):
        return {**result, 'psd_available': False,
                'reason': 'nonuniform sampling: no implicit interpolation'}
    # Millisecond semantics are explicit in the field name. A source accidentally
    # saved in seconds must be identified/converted by the author, not guessed.
    if median_dt < 0.1 and t[-1] < 20.:
        return {**result, 'psd_available': False,
                'reason': 'timestamps may be in seconds despite _ms key; inspect raw schema'}
    fs = 1000. / median_dt
    result['sampling_hz'] = fs
    valid = (t >= 1000. - 1e-8) & (t <= 8000. + 1e-8)
    segment = t[valid]
    result['psd_sample_count'] = int(valid.sum())
    if len(segment):
        result['psd_window_s'] = [float(segment[0]/1000.), float(segment[-1]/1000.)]
    if fs <= 24.:
        return {**result, 'psd_available': False,
                'reason': f'sampling rate {fs:g} Hz does not resolve 4–12 Hz theta'}
    if len(segment) < 256:
        return {**result, 'psd_available': False,
                'reason': f'only {len(segment)} current samples in 1–8 s; Welch requires >=256'}
    return {**result, 'psd_available': True,
            'reason': 'PSD uses only the actually archived I_inh interval'}


def measured_psd(raw: dict) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """Welch from actual Phase-13 current data, without extending its window.

    Uses the legacy Hamming/256/128/1024 settings at the actual recorded fs.
    When data are a complete 200-Hz 1–8 s trace, this reproduces welch_proxy.
    """
    inventory = describe_current(raw)
    if not inventory['psd_available']:
        raise ValueError('Current trace insufficient for full Fig. 1d PSD: '
                         + inventory.get('reason', 'unavailable')
                         + '. The E/I raster is still usable for --scan.')
    t = np.asarray(raw['sample_times_ms'], dtype=float)
    x = np.asarray(raw['plfp_gaba_e'], dtype=float)
    valid = (t >= 1000. - 1e-8) & (t <= 8000. + 1e-8)
    fs = float(inventory['sampling_hz'])
    f, p = signal.welch(x[valid], fs=fs,
                        window=signal.get_window('hamming', 256),
                        nperseg=256, noverlap=128, nfft=1024)
    usable = (f > 0) & (f <= 50)
    peak = float(f[usable][np.argmax(p[usable])])
    return f, p, peak, inventory


def recorded_theta(raw: dict) -> np.ndarray:
    """Bandpass only on archived current, at its measured sampling frequency."""
    inventory = describe_current(raw)
    if not inventory['psd_available']:
        raise ValueError(inventory.get('reason', 'theta current unavailable'))
    x = np.asarray(raw['plfp_gaba_e'], dtype=float)
    sos = signal.butter(3, [4., 12.], btype='bandpass',
                        fs=inventory['sampling_hz'], output='sos')
    return signal.sosfiltfilt(sos, x)


def axial_difference(angles, center_deg):
    return (np.asarray(angles) - center_deg + 90.) % 180. - 90.


def descriptive_diagnostics(raw: dict, seed: int) -> dict:
    """Diagnostics for *visual selection*, not a new quantitative paper endpoint.

    No numeric cutoff is used to select a trial. Reviewer inspects the contact
    sheet and later the full figure, with the seed fixed across all its panels.
    """
    st = np.asarray(raw['spike_times_e_ms'], float)
    ang = np.asarray(raw['spike_angles_e_deg'], float)
    use = (st >= 1000) & (st <= 8000)
    st, ang = st[use], ang[use]
    if len(st) < 20:
        raise ValueError(f'seed={seed}: too few persistent-period E spikes to depict a bump')
    m2 = np.exp(2j * np.deg2rad(ang)).mean()
    center = float((np.degrees(np.angle(m2)) / 2) % 180)
    dist = np.abs(axial_difference(ang, center))
    inside = dist <= 35.
    # Tail is the complement of the two ±35° fields on the 180° axial circle.
    outside = ~inside
    inventory = describe_current(raw)
    peak = float('nan')
    ratio = float('nan')
    phase_lock = float('nan')
    phase_count = 0
    if inventory['psd_available']:
        f, p, peak, inventory = measured_psd(raw)
        freq_mask = (f >= 1) & (f <= 50)
        theta_mask = (f >= 4) & (f <= 12)
        ratio = float(np.sum(p[theta_mask]) / max(np.sum(p[freq_mask]), 1e-30))
        # ONLY interpolate phases for spikes truly inside the sampled current
        # range; np.interp() otherwise copies an endpoint phase (false locking).
        filtered = recorded_theta(raw)
        phase = np.unwrap(np.angle(hilbert(filtered)))
        theta_t = np.asarray(raw['sample_times_ms'], dtype=float)
        within_t = st[inside]
        # 200-ms edge guard protects against Hilbert/bandpass edge artifacts.
        interior = (within_t >= theta_t[0] + 200.) & (within_t <= theta_t[-1] - 200.)
        phase_count = int(interior.sum())
        if phase_count >= 10:
            spike_phases = np.interp(within_t[interior], theta_t, phase)
            phase_lock = float(abs(np.exp(1j * spike_phases).mean()))
    # Short epochs: stability of the 180°-periodic position of the main bump.
    centers = []
    for a, b in zip(np.arange(1000., 7500., 500.), np.arange(1500., 8000., 500.)):
        chunk = ang[(st >= a) & (st < b)]
        if len(chunk) >= 20:
            z = np.mean(np.exp(2j * np.deg2rad(chunk)))
            if abs(z) > 0.05:
                centers.append(float((np.degrees(np.angle(z)) / 2) % 180))
    max_shift = float(max(abs(axial_difference(np.array(centers), center)))) if centers else float('nan')
    peak_value = round(float(peak), 4) if np.isfinite(peak) else None
    ratio_value = round(ratio, 5) if np.isfinite(ratio) else None
    phase_value = round(phase_lock, 5) if np.isfinite(phase_lock) else None
    shift_value = round(max_shift, 3) if np.isfinite(max_shift) else None
    return {'seed': seed, 'noise_pa': NOISE_PA, 'combo': DEFAULT_COMBO,
            'n_e_spikes_1_8s': int(len(st)), 'bump_center_axial_deg': round(center, 3),
            'axial_concentration': round(float(abs(m2)), 5),
            'fraction_e_spikes_within_35deg_bump': round(float(inside.mean()), 5),
            'n_off_bump_e_spikes_1_8s': int(outside.sum()),
            'off_bump_rate_hz_per_E_neuron': round(float(outside.sum() / (1024 * 7)), 5),
            'max_halfsecond_axial_center_shift_deg': shift_value,
            'dominant_frequency_hz': peak_value,
            'theta_4_12Hz_psd_fraction_1_50Hz': ratio_value,
            'within_bump_spike_theta_phase_concentration': phase_value,
            'phase_matched_spikes_n': phase_count,
            'psd_available': bool(inventory['psd_available']),
            'psd_window_s': inventory.get('psd_window_s'),
            'current_start_ms': inventory.get('recorded_start_ms'),
            'current_end_ms': inventory.get('recorded_end_ms'),
            'current_sampling_hz': inventory.get('sampling_hz'),
            'psd_note': inventory.get('reason')}


def render_contact_sheet(trials: list[tuple[int, dict, dict]], output: Path):
    count = len(trials)
    ncol = min(4, count)
    nrow = math.ceil(count / ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.0*ncol, 2.9*nrow), squeeze=False)
    for ax in axs.flat:
        ax.set_visible(False)
    for ax, (seed, raw, metrics) in zip(axs.flat, trials):
        ax.set_visible(True)
        t = np.asarray(raw['spike_times_e_ms'], float)
        ang = np.asarray(raw['spike_angles_e_deg'], float)
        mask = (t >= 1000) & (t <= 8000)
        ax.scatter(t[mask] / 1000, ang[mask], s=.45, alpha=.68, linewidths=0, color='#181818', rasterized=True)
        ax.set(xlim=(1, 8), ylim=(0, 360), yticks=[0, 180, 360], xticks=[1, 3, 5, 7], xlabel='Time (s)')
        peak_label = (f"{metrics['dominant_frequency_hz']:.2f} Hz"
                      if metrics['psd_available'] else 'PSD not available')
        ax.set_title(f"seed {seed:04d} | {peak_label} | "
                     f"bump {metrics['fraction_e_spikes_within_35deg_bump']:.2f}", fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle('Archived 90 pA / dynamic / tf_1000_tr_20 — actual E spikes, no simulated additions', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .98])
    fig.savefig(output, dpi=170)
    plt.close(fig)


def scan(folder: Path, output: Path):
    seeds = discover_seeds(folder)
    rows, thumbnails, failures = [], [], []
    for seed in seeds:
        try:
            raw, info = load_original_trial(folder, seed)
            metrics = descriptive_diagnostics(raw, seed)
            metrics['raw_sha256'] = info['raw_sha256']
            metrics['noise_evidence'] = info['noise_evidence']
            rows.append(metrics)
            thumbnails.append((seed, raw, metrics))
            peak_display = (f"{metrics['dominant_frequency_hz']:.2f} Hz"
                            if metrics['psd_available'] else 'unavailable')
            print(f"seed={seed:04d}: bump_fraction={metrics['fraction_e_spikes_within_35deg_bump']:.3f}, "
                  f"off_bump_E={metrics['n_off_bump_e_spikes_1_8s']}, "
                  f"PSD_peak={peak_display}; "
                  f"I_inh={info['current_record'].get('recorded_start_ms', 'NA')}.."
                  f"{info['current_record'].get('recorded_end_ms', 'NA')} ms", flush=True)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            failures.append({'seed': seed, 'error': str(exc)})
            print(f'seed={seed:04d} skipped (invalid raw): {exc}', file=sys.stderr, flush=True)
    if not rows:
        raise ValueError('No Phase-13 90-pA trial with readable E/I spike events found. Inspect errors above.')
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values('seed').to_csv(output/'noise90_seed_metrics.csv', index=False)
    render_contact_sheet(thumbnails, output/'noise90_raster_contact_sheet.png')
    summary = {'source_directory': str(folder), 'run_type': 'archived_raw_only_no_simulator',
               'valid_seeds': [x['seed'] for x in rows], 'failures': failures,
               'decision': 'no_automatic_selection; inspect localization and theta locking visually',
               'note': 'The per-trial diagnostics are descriptive visual QC, not new inferential endpoints.'}
    (output/'noise90_scan.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print('Saved review raster + CSV + scan.json:', output.resolve())
    if failures:
        print(f'Warning: {len(failures)} archives could not be used; see noise90_scan.json', file=sys.stderr)


def render_selected_seed_qc(raw: dict, metrics: dict, window: tuple[float, float], output: Path):
    """Detailed review: recorded E/I rasters, inhibitory current, PSD, theta phases.

    Only for screening a *candidate*. There are no artificial background spikes.
    """
    from scipy import signal
    t_e = np.asarray(raw['spike_times_e_ms'], float) / 1000.
    a_e = np.asarray(raw['spike_angles_e_deg'], float)
    t_i = np.asarray(raw['spike_times_i_ms'], float) / 1000.
    a_i = np.asarray(raw['spike_angles_i_deg'], float)
    # Export requires an actual PSD-capable trace, but it need not be 0–8 s.
    f, psd, _, current_info = measured_psd(raw)
    t = np.asarray(raw['sample_times_ms'], float) / 1000.
    inh = np.asarray(raw['plfp_gaba_e'], float)
    theta_current = recorded_theta(raw)
    start, end = window
    fig = plt.figure(figsize=(12.6, 8.6), facecolor='white')
    gs = fig.add_gridspec(3, 3, width_ratios=[3., 3., 1.3], height_ratios=[1.7, 1.12, 1.12],
                          left=.07, right=.97, top=.87, bottom=.08, wspace=.37, hspace=.54)
    overview = fig.add_subplot(gs[0, :2])
    marginal = fig.add_subplot(gs[0, 2], sharey=overview)
    ezoom = fig.add_subplot(gs[1, 0])
    izoom = fig.add_subplot(gs[1, 1], sharex=ezoom)
    powers = fig.add_subplot(gs[1, 2])
    currents = fig.add_subplot(gs[2, :2])
    phases = fig.add_subplot(gs[2, 2])
    overview.scatter(t_e, a_e, s=.75, linewidths=0, color='#191919', rasterized=True)
    overview.axvspan(start, end, color='#cddcde', alpha=.32, zorder=0)
    overview.set(xlim=(0, 8), ylim=(0, 360), yticks=[0, 90, 180, 270, 360],
                 xlabel='Time (s)', ylabel='E-cell angle (deg)', title='Actual E-spike raster: 0–8 s')
    # A robust *unsmoothed* per-neuron marginal, with zeros for silent neurons.
    theta = np.asarray(raw['theta_e_deg'], float)
    rate_counts, _ = np.histogram(a_e[(t_e >= 1.) & (t_e <= 8.)], bins=np.r_[theta, 360.])
    marginal.plot(rate_counts / 7., theta, color='#253b48', lw=.8)
    marginal.set(xlabel='Rate (Hz)', title='E-rate marginal')
    marginal.tick_params(labelleft=False, axis='y')
    for ax, pop, spikes_t, spikes_a, color in [
        (ezoom, 'E', t_e, a_e, '#171717'), (izoom, 'I', t_i, a_i, '#555555')
    ]:
        keep = (spikes_t >= start) & (spikes_t <= end)
        ax.scatter(spikes_t[keep], spikes_a[keep], s=1.2 if pop == 'E' else .7,
                   linewidths=0, color=color, rasterized=True)
        ax.set(xlim=window, ylim=(0, 360), xlabel='Time (s)',
               ylabel=f'{pop}-cell angle (deg)', title=f'{pop} raster: {start:g}–{end:g} s')
    powers.plot(f, psd, color='#31617b', lw=1.1)
    powers.axvspan(4, 12, color='#cddcde', alpha=.3)
    psd_label = current_info['psd_window_s']
    powers.set(xlim=(0, 30), xlabel='Frequency (Hz)', ylabel='PSD (pA²/Hz)',
               title=f'I_inh PSD ({psd_label[0]:.2f}–{psd_label[1]:.2f} s)')
    select = (t >= start) & (t <= end)
    if select.any():
        currents.plot(t[select], inh[select] - inh[(t >= 1) & (t <= 8)].mean(),
                      lw=.65, color='#777777', label='Demeaned I_inh')
        currents.plot(t[select], theta_current[select], lw=1., color='#31617b', label='4–12 Hz')
    else:
        currents.text(.5, .5, 'No archived I_inh samples in the E/I zoom interval',
                      transform=currents.transAxes, ha='center', va='center', fontsize=9)
    currents.set(xlim=window, xlabel='Time (s)', ylabel='Current (pA)',
                 title='Current: raw and 4–12-Hz visualization')
    if select.any():
        currents.legend(loc='upper right', fontsize=8, frameon=False)
    mid = float(metrics['bump_center_axial_deg'])
    near = (t_e >= max(1.5, t[0] + .2)) & (t_e <= min(7.5, t[-1] - .2)) & (np.abs(axial_difference(a_e, mid)) <= 35)
    ph_signal = np.angle(signal.hilbert(theta_current))
    # For event phases interpolate the continuous UNWRAPPED Hilbert phase.
    ph_unwrapped = np.unwrap(ph_signal)
    spike_phases = np.interp(t_e[near], t, ph_unwrapped) if np.any(near) else np.array([])
    spike_phases = (spike_phases + np.pi) % (2*np.pi) - np.pi
    phases.hist(spike_phases, bins=np.linspace(-np.pi, np.pi, 17), color='#31617b', linewidth=.1)
    phases.set(xlim=(-np.pi, np.pi), xticks=[-np.pi, 0, np.pi], xticklabels=['−π', '0', 'π'],
               xlabel='I_inh theta phase (rad)', ylabel='E spike count', title='Within-bump E spikes')
    fig.suptitle('CANDIDATE, NOT APPROVED — genuine recorded spikes; do not accept if bump/theta degrades\n'
                 f"Phase 13, dynamic, 90 pA, seed {metrics['seed']:04d} | "
                 f"axial concentration {metrics['axial_concentration']:.2f} | "
                 f"off-bump E spikes {metrics['n_off_bump_e_spikes_1_8s']} | "
                 f"peak {metrics['dominant_frequency_hz']:.2f} Hz",
                 fontsize=11, y=.975)
    fig.savefig(output, dpi=220, facecolor='white')
    plt.close(fig)


def approve_trial(source: Path, confirm_bump: bool, confirm_theta: bool,
                  confirm_background: bool, confirm_reference: bool, confirm_full: bool):
    """Author explicitly approves visual criteria; freeze approval in sidecar metadata.

    The original raw and exported scientific files must still match their hashes.
    """
    source = source.expanduser().resolve()
    if source == (MAIN/'source_data').resolve() or (MAIN/'source_data').resolve() in source.parents:
        raise ValueError('Cannot approve/modify the frozen 0-pA source_data directory.')
    d = fig1_model.load_source_data(source)
    meta = d.get('representative_metadata')
    if meta is None:
        raise ValueError('Not a dedicated 90-pA candidate with Fig1_representative_trial.json.')
    if meta.get('selection_status') != 'candidate_pending_visual_review':
        raise ValueError('Expected an unapproved Fig. 1 90-pA candidate; do not approve twice.')
    if not all((confirm_bump, confirm_theta, confirm_background, confirm_reference, confirm_full)):
        raise ValueError('Author must explicitly confirm bump, theta, sparse background, '
                         'same reference model except noise, and the complete original 0–8-s protocol.')
    if meta['diagnostics']['n_off_bump_e_spikes_1_8s'] <= 0:
        raise ValueError('There are zero off-bump spikes; Fig. 1 visual criteria not satisfied.')
    original = Path(meta['raw_path'])
    if not original.is_file() or sha256(original) != meta['raw_sha256']:
        raise ValueError('Original Phase-13 NPZ is missing or changed since export.')
    sidecar = meta.get('raw_sidecar_path')
    if sidecar and (not Path(sidecar).is_file() or sha256(sidecar) != meta['raw_sidecar_sha256']):
        raise ValueError('Original raw sidecar was changed since export.')
    qc = source/'Fig1_noise90_QC.png'
    if not qc.is_file() or sha256(qc) != meta['qc_sha256']:
        raise ValueError('Candidate QC image changed since export; review is not verifiable.')
    # Keep 5 source-data files byte-identical; only the approval JSON changes.
    meta['selection_status'] = 'visually_approved_by_author'
    meta['approved_utc'] = datetime.now(timezone.utc).isoformat()
    meta['manual_checks'] = {
        'clear_localized_bump': True, 'clear_theta_organization': True,
        'sparse_not_dense_outside_bump_activity': True,
        'same_reference_model_except_noise': True, 'full_8s_original_trial': True}
    path = source/'Fig1_representative_trial.json'
    tmp = source/'Fig1_representative_trial.json.pending'
    tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    tmp.replace(path)
    fig1_model.load_source_data(source)  # verify approved manifest too
    print('Explicitly approved Fig. 1 90-pA illustrative seed:', meta['seed'], '|', source)
    print('No change to the quantitative zero-noise reference or other figures.')


def export_trial(folder: Path, seed: int, output: Path, window: tuple[float, float],
                 overwrite=False, cue_confirmed=False):
    """Complete re-export of Fig. 1 B–F from one 90-pA seed, without mixing 0-pA panels."""
    output = Path(output).expanduser().resolve()
    canonical_source = (MAIN/'source_data').resolve()
    if output == canonical_source or canonical_source in output.parents or output == MAIN.resolve():
        raise ValueError(f'Refusing to overwrite canonical reference Fig. 1 files: {canonical_source}')
    if output == folder or folder in output.parents:
        raise ValueError('Refusing to write into the Phase-13 archival result tree.')
    if output.exists():
        if not overwrite:
            raise FileExistsError(f'Output exists: {output}. Choose a new output or use --overwrite.')
        label = output/'Fig1_representative_trial.json'
        if not label.is_file():
            raise ValueError('Output directory lacks Fig. 1 representative-trial marker; refusing overwrite.')
        existing = json.loads(label.read_text(encoding='utf-8'))
        if existing.get('noise_pa') != NOISE_PA or existing.get('purpose') != 'fig1_descriptive_only':
            raise ValueError('Existing folder is not marked as an isolated Fig. 1 90-pA export.')
    raw, info = load_original_trial(folder, seed)
    metrics = descriptive_diagnostics(raw, seed)
    # Preserve legacy compute -> export -> load -> draw separation.
    f, power, peak, current_info = measured_psd(raw)
    d = fig1_model.compute_panels(raw, window=window,
                                  psd_override=(f, power, peak))
    if not np.isclose(d['d_peak_hz'], metrics['dominant_frequency_hz'], atol=.0002):
        raise RuntimeError('PSD peak differs between raw diagnostics and Fig. 1 data layer.')
    output.parent.mkdir(parents=True, exist_ok=True)
    # A temporary Export writes exactly the source filenames expected by v10/CPL.
    with tempfile.TemporaryDirectory(prefix='fig1_noise90_export_', dir=output.parent) as tmp:
        tmp_path = Path(tmp)
        class SourceExporter:
            def __init__(self, root): self.root = root
            def csv(self, stem, table):
                table.to_csv(self.root/f'{stem}.csv', index=False)
            def npz(self, stem, **arrays):
                np.savez_compressed(self.root/f'{stem}.npz', **arrays)
        stage = tmp_path/'source_data'
        stage.mkdir()
        fig1_model.export_source_data(SourceExporter(stage), d)
        files = {name: sha256(stage/name) for name in fig1_model.SOURCE_FILES}
        qc = stage/'Fig1_noise90_QC.png'
        render_selected_seed_qc(raw, metrics, window, qc)
        metadata = {
            'schema': 'Fig1_representative_trial_v1', 'purpose': 'fig1_descriptive_only',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'noise_pa': NOISE_PA, 'seed': seed, 'condition': CONDITION, 'combo': DEFAULT_COMBO,
            'source_experiment': 'phase13_attractor_confirmation',
            'reference_condition_unchanged': True,
            'full_trial_is_one_seed_for_panels_b_to_f': True,
            'raw_path': info['raw_path'], 'raw_sha256': info['raw_sha256'],
            'noise_evidence': info['noise_evidence'], 'window_s': list(window),
            'raw_sidecar_path': info['raw_sidecar_path'],
            'raw_sidecar_sha256': info['raw_sidecar_sha256'],
            'cue_0_to_0p5_verified': bool(cue_confirmed),
            'psd_peak_hz': float(d['d_peak_hz']),
            'psd_window_s': current_info['psd_window_s'],
            'current_record': current_info,
            'psd_method': 'Welch: archived current only, Hamming 256, overlap 128, nfft 1024',
            'stp_replay_assumes_rest_initial_state_at_zero': True,
            'diagnostics': metrics,
            'source_data_sha256': files, 'qc_sha256': sha256(qc),
            'selection_policy': 'illustrative_only; user_visually_approves_bump_localization_and_theta; no_automatic_seed_pick',
            'selection_status': 'exported',
        }
        (stage/'Fig1_representative_trial.json').write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
        # Re-read all 5 files through the exact production Fig. 1 source-data loader.
        loaded = fig1_model.load_source_data(stage, window=window)
        assert np.isclose(loaded['d_peak_hz'], d['d_peak_hz'])
        # Stage and commit only after fully validated. Back up an existing approved export.
        backup = None
        try:
            if output.exists():
                backup = output.with_name(output.name + '.previous')
                if backup.exists():
                    raise FileExistsError(f'Old backup already exists, remove/rename manually: {backup}')
                output.rename(backup)
            stage.rename(output)
        except Exception:
            if backup is not None and backup.is_dir() and not output.exists():
                backup.rename(output)
            raise
        # The pre-existing export, if any, is preserved as *.previous rather than deleted.
    print(f'Exported real Phase-13 noise90 seed={seed:04d}: {output}')
    print(f"PSD peak = {d['d_peak_hz']:.4f} Hz; off-bump E spikes (1–8 s) = "
          f"{metrics['n_off_bump_e_spikes_1_8s']}; source SHA256 recorded.")
    print('This is an *unapproved candidate* until the raster and theta structure are inspected.')
    print('Open detailed E/I/PSD/theta QC:', output/'Fig1_noise90_QC.png')
    return output


def inspect_seed(folder: Path, seed: int):
    """Print archive inventory without requiring PSD or other full-panel inputs."""
    path = folder / SEED_NAME.format(seed)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    print('ARCHIVE:', path)
    print('RAW FILE SHA256:', sha256(path))
    for k, a in sorted(raw.items()):
        a = np.asarray(a)
        desc = f'  {k}: shape={a.shape}, dtype={a.dtype}'
        if a.ndim == 1 and a.size and np.issubdtype(a.dtype, np.number):
            desc += f', min={np.min(a):.6g}, max={np.max(a):.6g}'
        elif a.size == 1:
            desc += f', value={a.item()}'
        print(desc)
    report = describe_current(raw)
    print('CURRENT TRACE/PSD INVENTORY:', json.dumps(report, indent=2, ensure_ascii=False))
    print('IMPORTANT: a missing/short current trace does not invalidate archived spike rasters.')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    choices = ap.add_mutually_exclusive_group(required=True)
    choices.add_argument('--scan', action='store_true', help='Inspect available noise_090/dynamic archival seeds; write raster contact sheet and QC CSV.')
    choices.add_argument('--inspect-seed', type=int, metavar='SEED',
                         help='Print raw NPZ fields, shapes, exact stored time ranges and current PSD suitability.')
    choices.add_argument('--seed', type=int, help='Export this chosen archival seed (for example 100, 105, ...).')
    ap.add_argument('--results-root', type=Path, default=RESULTS_ROOT,
                    help='Local results folder containing phase13_attractor_confirmation.')
    ap.add_argument('--combo', default=DEFAULT_COMBO, help='Locked to tf_1000_tr_20 for this Fig. 1 protocol.')
    ap.add_argument('--review-output', type=Path, default=MAIN/'fig1_noise90_review')
    ap.add_argument('--output', type=Path, default=MAIN/'source_data_fig1_noise90')
    ap.add_argument('--window', type=float, nargs=2, default=(5., 6.2), metavar=('START_S', 'END_S'))
    ap.add_argument('--overwrite', action='store_true', help='Version an existing dedicated noise90 export as *.previous.')
    ap.add_argument('--cue-confirmed', action='store_true', default=True, help='Include the 0-0.5 s cue marker.')
    args = ap.parse_args(argv)
    if not 1. <= args.window[0] < args.window[1] <= 8. or args.window[1] - args.window[0] < .05:
        ap.error('Window must lie in [1,8] s and be at least 50 ms long.')
    try:
        folder = raw_directory(args.results_root, args.combo)
        if args.scan:
            scan(folder, args.review_output)
        elif args.inspect_seed is not None:
            inspect_seed(folder, args.inspect_seed)
        else:
            if args.seed not in discover_seeds(folder):
                raise FileNotFoundError(f'No seed={args.seed} under {folder}. Use --scan to list valid seeds.')
            export_trial(folder, args.seed, args.output, tuple(args.window), args.overwrite,
                         cue_confirmed=args.cue_confirmed)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        ap.exit(2, f'INPUT ERROR: {exc}\n')


if __name__ == '__main__':
    main()
