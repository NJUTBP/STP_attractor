import numpy as np
from scipy import signal

from analysis.oscillation import diagnose_oscillation


def test_acf_recovers_seven_hz_periodicity():
    fs = 200.0
    t = np.arange(0.0, 8.0, 1.0 / fs)
    x = np.sin(2 * np.pi * 7.0 * t) + 0.25 * np.sin(2 * np.pi * 14.0 * t)
    f, p = signal.welch(x, fs=fs, nperseg=256, noverlap=128, nfft=1024)
    d = diagnose_oscillation(x, fs, f, p)
    assert abs(d.acf_best_frequency_hz - 7.0) < 0.5
    assert any(abs(v - 7.0) < 0.5 for v in d.spectral_peak_frequencies_hz)
