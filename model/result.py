from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SimulationResult:
    condition: str
    seed: int
    noise_pa: float
    static_gain: float | None
    temporal_gain_scale: float
    resource_release_u_override: float | None
    theta_e_deg: np.ndarray
    spike_times_e_ms: np.ndarray
    spike_angles_e_deg: np.ndarray
    spike_times_i_ms: np.ndarray
    spike_angles_i_deg: np.ndarray
    sample_times_ms: np.ndarray
    plfp_gaba_e: np.ndarray
    mean_efficacy_by_e: np.ndarray
    gain_match_numerator: float
    gain_match_denominator: float
    resource_release_u_sum: float = 0.0
    resource_release_u_count: int = 0
    # Optional Phase-14 STP spatial-trace recorder. Empty arrays keep old results
    # and all non-trace simulations backward compatible.
    stp_trace_times_ms: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    stp_u_axis_deg: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    stp_u_resultant: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    stp_u_mass: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    stp_r_axis_deg: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    stp_r_resultant: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    stp_r_mass: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))

    @property
    def p_match_contribution(self) -> tuple[float, float]:
        return self.gain_match_numerator, self.gain_match_denominator

    @property
    def mean_resource_release_u(self) -> float:
        if self.resource_release_u_count <= 0:
            return float("nan")
        return self.resource_release_u_sum / self.resource_release_u_count
