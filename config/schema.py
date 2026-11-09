from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

ConditionName = Literal["dynamic", "facilitation_only", "depletion_only", "baseline", "static_gain", "frozen_profile"]


@dataclass(frozen=True)
class NetworkConfig:
    ne: int = 1024
    ni: int = 256

    cm_e_pf: float = 500.0
    cm_i_pf: float = 200.0
    e_exc_mv: float = 0.0
    e_inh_mv: float = -70.0
    e_leak_mv: float = -70.0
    threshold_mv: float = -50.0
    reset_mv: float = -60.0
    tau_ref_e_ms: float = 2.0
    tau_ref_i_ms: float = 1.0
    g_leak_e_ns: float = 25.0
    g_leak_i_ns: float = 20.0

    sigma_ei_deg: float = 18.0
    j_ei_pos: float = 1.6
    ie_offset_deg: float = 90.0
    sigma_ie_deg: float = 30.0

    magnesium_mm: float = 1.0
    g_nmda_ei_ns: float = 0.8
    g_gaba_ie_ns: float = 2.4
    g_gaba_ii_ns: float = 0.04

    tau_gaba_ms: float = 10.0
    tau_x_ms: float = 2.0
    tau_nmda_ms: float = 100.0
    alpha_nmda_khz: float = 0.5

    background_e_pa: float = 750.0
    background_i_pa: float = 325.0
    ms_current_i_pa: float = -100.0

    tau_noise_ms: float = 2.0

    def validate(self) -> None:
        if self.ne <= 0 or self.ni <= 0:
            raise ValueError("Network sizes must be positive.")
        if self.ne != 4 * self.ni:
            raise ValueError("This implementation assumes NE == 4 * NI, matching the reference ring model.")


@dataclass(frozen=True)
class STFConfig:
    u_baseline: float = 0.4
    tau_facilitation_ms: float = 1000.0
    tau_recovery_ms: float = 20.0

    def validate(self) -> None:
        if not 0.0 <= self.u_baseline <= 1.0:
            raise ValueError("u_baseline must lie in [0, 1].")
        if self.tau_facilitation_ms <= 0 or self.tau_recovery_ms <= 0:
            raise ValueError("STF time constants must be positive.")


@dataclass(frozen=True)
class SimulationConfig:
    dt_ms: float = 0.02
    duration_ms: float = 8000.0
    cue_start_ms: float = 0.0
    cue_end_ms: float = 500.0
    cue_centers_deg: tuple[float, float] = (90.0, 270.0)
    cue_amplitude_pa: float = 200.0
    cue_half_width_deg: float = 35.15625  # legacy d=100 neurons at NE=1024
    sampling_rate_hz: int = 200

    def validate(self) -> None:
        if self.dt_ms <= 0 or self.duration_ms <= 0:
            raise ValueError("dt_ms and duration_ms must be positive.")
        if not 0 <= self.cue_start_ms < self.cue_end_ms <= self.duration_ms:
            raise ValueError("Cue interval must lie inside the simulation interval.")
        sample_interval_ms = 1000.0 / self.sampling_rate_hz
        ratio = sample_interval_ms / self.dt_ms
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("sampling interval must be an integer multiple of dt_ms.")


@dataclass(frozen=True)
class AnalysisConfig:
    pav_window_ms: float = 400.0
    pav_step_ms: float = 5.0
    pav_target_time_ms: float = 7000.0
    target_axis_deg: float = 90.0
    localization_reference_start_ms: float = 1000.0
    localization_reference_end_ms: float = 2000.0
    localization_target_start_ms: float = 6500.0
    localization_target_end_ms: float = 7500.0
    localization_ratio_cutoff: float = 0.5
    localization_failure_fraction_cutoff: float = 0.2

    theta_start_ms: float = 1000.0
    theta_end_ms: float = 8000.0
    theta_low_hz: float = 4.0
    theta_high_hz: float = 12.0
    max_psd_hz: float = 50.0
    welch_window_samples: int = 256
    welch_overlap_samples: int = 128
    welch_nfft: int = 1024
    peak_half_width_hz: float = 1.0

    gain_match_start_ms: float = 1000.0
    gain_match_end_ms: float = 7000.0

    bootstrap_repeats: int = 10_000


@dataclass(frozen=True)
class ExperimentConfig:
    attractor_noise_pa: float = 60.0
    theta_noise_pa: float = 0.0
    attractor_seeds: tuple[int, ...] = tuple(range(20))
    theta_seeds: tuple[int, ...] = tuple(range(10))
    conditions: tuple[ConditionName, ...] = ("dynamic", "baseline", "static_gain")


@dataclass(frozen=True)
class ProjectConfig:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    stf: STFConfig = field(default_factory=STFConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def validate(self) -> None:
        self.network.validate()
        self.stf.validate()
        self.simulation.validate()
        if self.analysis.theta_end_ms > self.simulation.duration_ms:
            raise ValueError("theta_end_ms cannot exceed simulation duration.")
        if self.analysis.pav_target_time_ms > self.simulation.duration_ms:
            raise ValueError("pav_target_time_ms cannot exceed simulation duration.")

    def to_dict(self) -> dict:
        return asdict(self)
