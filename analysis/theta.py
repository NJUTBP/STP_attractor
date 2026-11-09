from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import signal

from config.schema import AnalysisConfig, SimulationConfig
from model.result import SimulationResult


@dataclass(frozen=True)
class ThetaSummary:
    seed: int
    condition: str
    noise_pa: float
    dominant_frequency_hz: float
    normalized_peak_power: float
    dominant_in_theta_band: bool


@dataclass(frozen=True)
class PSDResult:
    frequency_hz: np.ndarray
    psd: np.ndarray


def _integral(y: np.ndarray, x: np.ndarray) -> float:
    if y.size < 2:
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def analyze_theta(
    result: SimulationResult,
    sim_cfg: SimulationConfig,
    ana_cfg: AnalysisConfig,
) -> tuple[ThetaSummary, PSDResult]:
    mask = (result.sample_times_ms >= ana_cfg.theta_start_ms) & (result.sample_times_ms <= ana_cfg.theta_end_ms)
    x = result.plfp_gaba_e[mask]
    if x.size < 8:
        raise ValueError("Not enough pLFP samples in theta analysis window.")

    nperseg = min(ana_cfg.welch_window_samples, x.size)
    noverlap = min(ana_cfg.welch_overlap_samples, max(0, nperseg - 1))
    nfft = max(ana_cfg.welch_nfft, nperseg)
    freq, psd = signal.welch(
        x,
        fs=sim_cfg.sampling_rate_hz,
        window=signal.get_window("hamming", nperseg),
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
    )

    search = (freq > 0.0) & (freq <= ana_cfg.max_psd_hz)
    if not np.any(search):
        raise ValueError("PSD frequency grid has no non-DC bins in requested band.")
    idxs = np.flatnonzero(search)
    peak_idx = idxs[int(np.argmax(psd[search]))]
    f_dom = float(freq[peak_idx])

    full = (freq >= 0.0) & (freq <= ana_cfg.max_psd_hz)
    around = (freq >= max(0.0, f_dom - ana_cfg.peak_half_width_hz)) & (
        freq <= min(ana_cfg.max_psd_hz, f_dom + ana_cfg.peak_half_width_hz)
    )
    denom = _integral(psd[full], freq[full])
    numer = _integral(psd[around], freq[around])
    q_peak = numer / denom if denom > 0 else 0.0

    in_theta = ana_cfg.theta_low_hz < f_dom <= ana_cfg.theta_high_hz
    summary = ThetaSummary(
        seed=result.seed,
        condition=result.condition,
        noise_pa=result.noise_pa,
        dominant_frequency_hz=f_dom,
        normalized_peak_power=float(q_peak),
        dominant_in_theta_band=bool(in_theta),
    )
    return summary, PSDResult(freq, psd)
