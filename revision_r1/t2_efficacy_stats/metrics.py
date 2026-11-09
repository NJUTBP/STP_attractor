"""T2 metrics M1-M7 for one trial, plus the frequency, spikes-per-cycle and u+ checks used by T4."""
from __future__ import annotations

import numpy as np
from scipy import signal

from revision_r1.t2_efficacy_stats.reconstruct import (
    TEMPORAL, U_DYNAMIC, efficacy, neuron_index, replay_events, replay_s_nmda, sample_uR,
)

N_PHASE_BINS = 50
ACF_MAX_LAG_MS = 500.0


def u_plus_periodic(f_hz: float, U: float, tau_f_ms: float, spikes_per_cycle: float = 1.0) -> float:
    """Periodic fixed point of u+ for evenly spaced spikes (T4 formula)."""
    if not np.isfinite(f_hz) or f_hz <= 0 or not np.isfinite(spikes_per_cycle) or spikes_per_cycle <= 0:
        return float("nan")
    T = 1000.0 / (f_hz * spikes_per_cycle)
    E = np.exp(-T / tau_f_ms)
    return float(U * (1.0 + (1.0 - U) * (1.0 - E)) / (1.0 - (1.0 - U) * E))


def axial_width_deg(weights, theta_deg) -> float:
    w = np.clip(np.asarray(weights, float), 0.0, None)
    tot = w.sum()
    if tot <= 0:
        return float("nan")
    R = np.abs(np.sum(w * np.exp(2j * np.deg2rad(theta_deg)))) / tot
    R = min(max(R, 1e-12), 1.0)
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(R)) / 2.0))


def psd_frequency_from_proxy(raw: dict, start_ms: float = 1000.0, end_ms: float = 8000.0) -> float:
    """The paper's estimate: Welch (Hamming 256, overlap 128, nfft 1024) of the 200-Hz I_inh proxy."""
    t = np.asarray(raw["sample_times_ms"], float)
    x = np.asarray(raw["plfp_gaba_e"], float)
    m = (t >= start_ms) & (t <= end_ms)
    if m.sum() < 256:
        return float("nan")
    f, p = signal.welch(x[m], fs=200.0, window=signal.get_window("hamming", 256), nperseg=256,
                        noverlap=128, nfft=1024)
    sel = (f > 0) & (f <= 50)
    return float(f[sel][np.argmax(p[sel])])


def dominant_frequency(x, fs: float, fmin: float = 1.0, fmax: float = 50.0) -> float:
    x = np.asarray(x, float) - np.mean(x)
    nper = min(len(x), int(round(fs * 1.28)))
    if nper < 16:
        return float("nan")
    f, p = signal.welch(x, fs=fs, window="hamming", nperseg=nper, noverlap=nper // 2, nfft=max(4 * nper, 1024))
    m = (f >= fmin) & (f <= fmax)
    return float(f[m][np.argmax(p[m])]) if np.any(m) else float("nan")


def acf(x, max_lag: int) -> np.ndarray:
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    if n < 2 or np.allclose(x, 0.0):
        return np.full(max_lag + 1, np.nan)
    F = np.fft.rfft(x, 2 * n)
    c = np.fft.irfft(F * np.conj(F))[: max_lag + 1]
    return c / c[0]


def cycle_waveform(x, fs: float, f0: float, nbins: int = N_PHASE_BINS):
    """Phase-binned waveform of x / mean(x) - 1 (phase from a band-pass around the trace's own f0)."""
    x = np.asarray(x, float)
    mu = x.mean()
    if not np.isfinite(f0) or f0 <= 0 or mu == 0 or len(x) < 3 * fs / f0:
        return np.full(nbins, np.nan), float("nan")
    lo, hi = 0.6 * f0, min(1.6 * f0, 0.45 * fs)
    sos = signal.butter(3, [lo, hi], btype="bandpass", fs=fs, output="sos")
    ph = np.angle(signal.hilbert(signal.sosfiltfilt(sos, x - mu)))
    b = np.clip(((ph + np.pi) / (2 * np.pi) * nbins).astype(int), 0, nbins - 1)
    wave = np.array([x[b == k].mean() if np.any(b == k) else np.nan for k in range(nbins)]) / mu - 1.0
    return wave, float(np.nanmax(wave) - np.nanmin(wave))


def _wavg(x, w):
    x = np.asarray(x, float); w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.sum(w[ok] * x[ok]) / np.sum(w[ok])) if np.any(ok) else float("nan")


def _wsd(x, w):
    m = _wavg(x, w)
    return float(np.sqrt(_wavg((np.asarray(x, float) - m) ** 2, w))) if np.isfinite(m) else float("nan")


def compute_trial_metrics(raw: dict, *, mode: str, U: float, tau_f: float, tau_r: float, w_out: np.ndarray,
                          theta_e: np.ndarray, ni: int, window=(1000.0, 7000.0), sample_ms: float = 1.0,
                          dt_ms: float = 0.02, nmda=None) -> tuple[dict, dict]:
    nmda = nmda or {"tau_x": 2.0, "tau_nmda": 100.0, "alpha": 0.5}
    t0, t1 = float(window[0]), float(window[1])
    t = np.arange(t0 + sample_ms, t1 + 1e-9, sample_ms)
    ne = len(theta_e)
    k = float(raw.get("temporal_gain_scale", 1.0)) if mode in TEMPORAL else 1.0
    ov = float(raw.get("resource_release_u_override", np.nan))
    p_static = float(raw.get("static_gain", np.nan))
    frozen = raw.get("mean_efficacy_by_e")
    ids_all = neuron_index(raw, ne)
    ts_all = np.asarray(raw["spike_times_e_ms"], float)
    active = np.unique(ids_all[ts_all + dt_ms <= t1 + 1e-9])
    if active.size == 0:
        raise ValueError("no E spikes before the window end")
    ev = None
    if mode in TEMPORAL:
        ev = replay_events(raw, mode, U=U, tau_f=tau_f, tau_r=tau_r, t_end_ms=t1,
                           release_override=ov if mode == "mean_u_release" else None, dt_ms=dt_ms, ne=ne)
        u_s, r_s = sample_uR(ev, t, active, U=U, tau_f=tau_f, tau_r=tau_r, u_dynamic=mode in U_DYNAMIC)
    else:
        u_s = np.full((len(t), len(active)), U)
        r_s = np.ones_like(u_s)
    s_s = replay_s_nmda(raw, t, active, dt_ms=dt_ms, ne=ne, **nmda)
    p_s = efficacy(mode, u_s, r_s, U=U, k=k, p_static=p_static, frozen=None if frozen is None else frozen[active])
    p_rest = efficacy(mode, np.full(ne, U), np.ones(ne), U=U, k=k, p_static=p_static, frozen=frozen)
    w = np.asarray(w_out, float)[active]
    d = s_s * p_s
    den = float(np.sum(s_s * w))
    row: dict = {"n_active": int(active.size), "window_start_ms": t0, "window_end_ms": t1, "sample_ms": sample_ms}
    # M1 drive-weighted mean efficacy
    row["M1_p_drive_weighted"] = float(np.sum(d * w) / den) if den > 0 else float("nan")
    # M2 temporal variability of p_i(t)
    mean_p = p_s.mean(0); var_p = p_s.var(0)
    cv_p = np.sqrt(var_p) / np.where(mean_p > 0, mean_p, np.nan)
    wt = w * s_s.mean(0)
    row["M2_var_p"] = _wavg(var_p, wt)
    row["M2_cv_p"] = _wavg(cv_p, wt)
    # M3 spatial profiles
    pbar = np.asarray(p_rest, float).copy(); pbar[active] = mean_p
    dbar = np.zeros(ne); dbar[active] = d.mean(0)
    row["M3_drive_peak"] = float(dbar.max())
    row["M3_drive_width_deg"] = axial_width_deg(dbar, theta_e)
    row["M3_p_spatial_sd"] = _wsd(mean_p, wt)
    # M4 population E->I drive  D(t) = sum_i w_i s_i p_i / N_I  (= mean over I cells of the NMDA input)
    fs = 1000.0 / sample_ms
    D = np.sum(d * w, axis=1) / float(ni)
    row["M4_drive_mean"] = float(D.mean()); row["M4_drive_sd"] = float(D.std())
    row["M4_drive_cv"] = float(D.std() / D.mean()) if D.mean() > 0 else float("nan")
    row["M4_drive_freq_hz"] = dominant_frequency(D, fs)
    ac = acf(D, int(round(ACF_MAX_LAG_MS / sample_ms)))
    lag1 = int(round(1000.0 / row["M4_drive_freq_hz"] / sample_ms)) if np.isfinite(row["M4_drive_freq_hz"]) else -1
    row["M4_acf_at_one_period"] = float(ac[lag1]) if 0 < lag1 < len(ac) else float("nan")
    zc = np.flatnonzero(ac < 0)
    row["M4_acf_first_zero_ms"] = float(zc[0] * sample_ms) if zc.size else float("nan")
    wave, depth = cycle_waveform(D, fs, row["M4_drive_freq_hz"])
    row["M4_modulation_depth"] = depth
    # M5 spike-triggered efficacy and release
    if ev is not None:
        win = (ev.t_spike >= t0) & (ev.t_spike < t1)
        wi = np.asarray(w_out, float)[ev.ids[win]]
        p_pre = efficacy(mode, ev.u_pre[win], ev.r_pre[win], U=U, k=k)
        rel = ev.rho[win] * ev.r_pre[win]
        row["M5_p_pre_mean"] = _wavg(p_pre, wi); row["M5_p_pre_sd"] = _wsd(p_pre, wi)
        row["M5_release_mean"] = _wavg(rel, wi); row["M5_release_sd"] = _wsd(rel, wi)
        row["M5_rho_spike_mean"] = float(np.mean(ev.rho[win])) if np.any(win) else float("nan")
        row["u_plus_spike_mean"] = float(np.mean(ev.u_post[win])) if np.any(win) and mode in U_DYNAMIC else float("nan")
    else:
        for key in ("M5_p_pre_mean", "M5_p_pre_sd", "M5_release_mean", "M5_release_sd", "M5_rho_spike_mean", "u_plus_spike_mean"):
            row[key] = float("nan")
    # M6 gating-efficacy covariance share
    cov = d.mean(0) - s_s.mean(0) * mean_p
    row["M6_cov_share"] = float(np.sum(w * cov) / np.sum(w * d.mean(0))) if np.sum(w * d.mean(0)) > 0 else float("nan")
    # M7 instantaneous across-neuron correlation of p and s
    live = s_s.mean(0) > 1e-3          # exclude neurons that fired only in the initial transient
    row["M7_n_neurons"] = int(live.sum())
    s_l, p_l = s_s[:, live], p_s[:, live]
    sc = s_l - s_l.mean(1, keepdims=True); pc = p_l - p_l.mean(1, keepdims=True)
    num = np.sum(sc * pc, axis=1); dd = np.sqrt(np.sum(sc ** 2, axis=1) * np.sum(pc ** 2, axis=1))
    row["M7_ps_corr"] = float(np.nanmean(np.where(dd > 0, num / np.where(dd > 0, dd, 1.0), np.nan))) if np.any(dd > 0) else float("nan")
    # Extras: paper frequency, spikes per cycle, u+ fixed point (T4 pre-check), stored matching values
    last = float(np.asarray(raw["sample_times_ms"])[-1]) if np.size(raw["sample_times_ms"]) else float("nan")
    row["f_proxy_hz"] = psd_frequency_from_proxy(raw, 1000.0, min(8000.0, last))
    inw = (ts_all >= t0) & (ts_all < t1)
    n_neur = np.unique(ids_all[inw]).size
    cycles = row["f_proxy_hz"] * (t1 - t0) / 1000.0
    row["spikes_per_active_neuron_per_cycle"] = float(inw.sum() / n_neur / cycles) if n_neur and cycles > 0 else float("nan")
    row["u_plus_fixed_point_pred"] = (u_plus_periodic(row["f_proxy_hz"], U, tau_f, row["spikes_per_active_neuron_per_cycle"])
                                      if mode in U_DYNAMIC else float("nan"))
    gn, gd = raw.get("gain_match_numerator", np.nan), raw.get("gain_match_denominator", np.nan)
    row["npz_p_match"] = float(gn / gd) if gd and np.isfinite(gd) and gd > 0 else float("nan")
    rc = raw.get("resource_release_u_count", 0.0)
    row["npz_release_u_mean"] = float(raw.get("resource_release_u_sum", np.nan) / rc) if rc and rc > 0 else float("nan")
    arrays = {"pbar": pbar.astype(np.float32), "dbar": dbar.astype(np.float32), "acf": ac.astype(np.float32),
              "waveform": wave.astype(np.float32)}
    # v3 additions (ADDENDUM_V3.md): R-, event-amplitude and spikes-per-cycle distributions
    from revision_r1.t2_efficacy_stats.event_stats import event_distribution_metrics
    ev_row, ev_arrays = event_distribution_metrics(ev, raw, mode=mode, U=U, t0=t0, t1=t1, ne=ne)
    row.update({f"V3_{k}": v for k, v in ev_row.items()})
    arrays.update(ev_arrays)
    return row, arrays
