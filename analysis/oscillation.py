from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np
from scipy import signal


@dataclass(frozen=True)
class OscillationDiagnostics:
    acf_best_frequency_hz: float
    acf_best_correlation: float
    acf_first_peak_frequency_hz: float
    acf_first_peak_correlation: float
    spectral_peak_frequencies_hz: tuple[float, ...]
    spectral_peak_relative_power: tuple[float, ...]

    def spectral_peaks_json(self) -> str:
        return json.dumps(
            [
                {"frequency_hz": f, "relative_to_max": p}
                for f, p in zip(self.spectral_peak_frequencies_hz, self.spectral_peak_relative_power)
            ]
        )


def normalized_autocorrelation(x: np.ndarray, fs_hz: float, max_lag_s: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        raise ValueError("Not enough samples for autocorrelation.")
    x = x - np.mean(x)
    corr = signal.correlate(x, x, mode="full", method="fft")[x.size - 1 :]
    if corr[0] <= 0:
        return np.arange(min(x.size, int(max_lag_s * fs_hz) + 1)) / fs_hz, np.zeros(
            min(x.size, int(max_lag_s * fs_hz) + 1)
        )
    corr = corr / corr[0]
    n = min(corr.size, int(round(max_lag_s * fs_hz)) + 1)
    lags_s = np.arange(n, dtype=float) / float(fs_hz)
    return lags_s, corr[:n]


def diagnose_oscillation(
    x: np.ndarray,
    fs_hz: float,
    frequency_hz: np.ndarray,
    psd: np.ndarray,
    *,
    min_frequency_hz: float = 2.0,
    max_frequency_hz: float = 50.0,
    acf_prominence: float = 0.05,
    spectral_prominence_fraction: float = 0.02,
    max_spectral_peaks: int = 8,
) -> OscillationDiagnostics:
    """Cross-check a PSD-dominant frequency with ACF periodicity and local spectral peaks.

    This is deliberately diagnostic rather than a replacement for the phase-4 primary PSD endpoint.
    It is intended to flag cases where the largest PSD peak is a harmonic of a slower waveform.
    """
    lags_s, acf = normalized_autocorrelation(x, fs_hz, max_lag_s=1.0 / min_frequency_hz)
    lag_mask = (lags_s >= 1.0 / max_frequency_hz) & (lags_s <= 1.0 / min_frequency_hz)
    candidate_idx = np.flatnonzero(lag_mask)
    if candidate_idx.size == 0:
        raise ValueError("No ACF lags in requested frequency range.")

    seg = acf[candidate_idx]
    peaks, props = signal.find_peaks(seg, prominence=acf_prominence)
    if peaks.size:
        global_peak_idx = candidate_idx[peaks]
        best_local = int(np.argmax(acf[global_peak_idx]))
        best_idx = int(global_peak_idx[best_local])
        first_idx = int(global_peak_idx[0])
    else:
        best_idx = int(candidate_idx[int(np.argmax(seg))])
        first_idx = best_idx

    def freq_from_idx(i: int) -> float:
        return float(1.0 / lags_s[i]) if lags_s[i] > 0 else float("nan")

    freq = np.asarray(frequency_hz, dtype=float)
    power = np.asarray(psd, dtype=float)
    search = (freq >= min_frequency_hz) & (freq <= max_frequency_hz)
    search_idx = np.flatnonzero(search)
    if search_idx.size:
        local_power = power[search_idx]
        max_power = float(np.max(local_power)) if local_power.size else 0.0
        prominence = spectral_prominence_fraction * max_power
        pks, _ = signal.find_peaks(local_power, prominence=prominence)
        peak_idx = search_idx[pks]
        if peak_idx.size == 0:
            peak_idx = np.asarray([search_idx[int(np.argmax(local_power))]])
        order = np.argsort(power[peak_idx])[::-1][:max_spectral_peaks]
        peak_idx = peak_idx[order]
        rel = power[peak_idx] / max_power if max_power > 0 else np.zeros(peak_idx.size)
        # Return in ascending frequency order for easy harmonic inspection.
        order_f = np.argsort(freq[peak_idx])
        peak_idx = peak_idx[order_f]
        rel = rel[order_f]
        peak_freqs = tuple(float(v) for v in freq[peak_idx])
        peak_rel = tuple(float(v) for v in rel)
    else:
        peak_freqs, peak_rel = tuple(), tuple()

    return OscillationDiagnostics(
        acf_best_frequency_hz=freq_from_idx(best_idx),
        acf_best_correlation=float(acf[best_idx]),
        acf_first_peak_frequency_hz=freq_from_idx(first_idx),
        acf_first_peak_correlation=float(acf[first_idx]),
        spectral_peak_frequencies_hz=peak_freqs,
        spectral_peak_relative_power=peak_rel,
    )
