from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from config.schema import STFConfig

# Full and mechanism-isolation modes whose efficacy changes in time.
TEMPORAL_CONDITIONS = frozenset({
    "dynamic",
    "facilitation_only",
    "depletion_only",
    "u_output_only",
    "u_release_only",
    "mean_u_release",
})

RESOURCE_MODES = frozenset({
    "dynamic",
    "depletion_only",
    "u_output_only",
    "u_release_only",
    "mean_u_release",
})

U_DYNAMIC_MODES = frozenset({
    "dynamic",
    "facilitation_only",
    "u_output_only",
    "u_release_only",
})


@dataclass
class DynamicSTFState:
    u: np.ndarray
    resources: np.ndarray

    @classmethod
    def initialize(cls, n: int, cfg: STFConfig) -> "DynamicSTFState":
        return cls(
            u=np.full(n, cfg.u_baseline, dtype=float),
            resources=np.ones(n, dtype=float),
        )

    @property
    def efficacy(self) -> np.ndarray:
        return self.u * self.resources

    def efficacy_for_mode(self, mode: str, cfg: STFConfig) -> np.ndarray:
        if mode in {"dynamic", "u_output_only"}:
            return self.u * self.resources
        if mode == "facilitation_only":
            return self.u
        if mode in {"depletion_only", "u_release_only", "mean_u_release"}:
            return cfg.u_baseline * self.resources
        raise ValueError(f"Unknown temporal STP mode: {mode}")

    def half_step_recovery(self, dt_ms: float, cfg: STFConfig) -> tuple[np.ndarray, np.ndarray]:
        fac = np.exp(-dt_ms / (2.0 * cfg.tau_facilitation_ms))
        rec = np.exp(-dt_ms / (2.0 * cfg.tau_recovery_ms))
        u_half = cfg.u_baseline + (self.u - cfg.u_baseline) * fac
        r_half = 1.0 + (self.resources - 1.0) * rec
        return u_half, r_half

    def recover_full_step(self, dt_ms: float, cfg: STFConfig) -> None:
        fac = np.exp(-dt_ms / cfg.tau_facilitation_ms)
        rec = np.exp(-dt_ms / cfg.tau_recovery_ms)
        self.u[:] = cfg.u_baseline + (self.u - cfg.u_baseline) * fac
        self.resources[:] = 1.0 + (self.resources - 1.0) * rec

    def on_spikes(
        self,
        spikes: np.ndarray,
        cfg: STFConfig,
        mode: str = "dynamic",
        *,
        release_u_override: float | None = None,
    ) -> tuple[float, int]:
        """Apply spike-triggered STP updates.

        Returns the sum and count of the actual release probabilities used to
        deplete resources on this time step.  This lets Phase 10 construct a
        spike-event-weighted mean-release control without saving full u(t)
        trajectories.

        Phase-10 mechanism modes:
          u_output_only: dynamic u multiplies postsynaptic efficacy, but R is
                         depleted with fixed baseline U.
          u_release_only: dynamic u controls R depletion, but postsynaptic
                          efficacy uses U*R (up to a matched scalar).
          mean_u_release: R is depleted with a fixed externally supplied mean
                          release probability; postsynaptic efficacy uses U*R.
        """
        if not np.any(spikes):
            return 0.0, 0
        if mode not in TEMPORAL_CONDITIONS:
            raise ValueError(f"Unknown temporal STP mode: {mode}")

        if mode in U_DYNAMIC_MODES:
            self.u += spikes * cfg.u_baseline * (1.0 - self.u)

        release_sum = 0.0
        release_count = 0
        if mode in RESOURCE_MODES:
            if mode in {"dynamic", "u_release_only"}:
                release_u: np.ndarray | float = self.u
            elif mode == "mean_u_release":
                if release_u_override is None or not np.isfinite(release_u_override):
                    raise ValueError("mean_u_release requires finite release_u_override")
                release_u = float(release_u_override)
            else:  # depletion_only and u_output_only
                release_u = cfg.u_baseline

            if np.isscalar(release_u):
                p = float(release_u)
                self.resources -= spikes * self.resources * p
                release_count = int(np.count_nonzero(spikes))
                release_sum = p * release_count
            else:
                self.resources -= spikes * self.resources * release_u
                used = np.asarray(release_u)[spikes]
                release_count = int(used.size)
                release_sum = float(np.sum(used))

        np.clip(self.u, 0.0, 1.0, out=self.u)
        np.clip(self.resources, 0.0, 1.0, out=self.resources)
        return release_sum, release_count


def temporal_efficacy_from_components(
    mode: str,
    u: np.ndarray,
    resources: np.ndarray,
    stf_cfg: STFConfig,
) -> np.ndarray:
    if mode in {"dynamic", "u_output_only"}:
        return u * resources
    if mode == "facilitation_only":
        return u
    if mode in {"depletion_only", "u_release_only", "mean_u_release"}:
        return stf_cfg.u_baseline * resources
    raise ValueError(f"Unknown temporal STP mode: {mode}")


def efficacy_for_condition(
    condition: str,
    stf_state: DynamicSTFState | None,
    stf_cfg: STFConfig,
    static_gain: float | None,
    frozen_profile: np.ndarray | None = None,
) -> np.ndarray | float:
    if condition in TEMPORAL_CONDITIONS:
        if stf_state is None:
            raise ValueError(f"{condition} condition requires DynamicSTFState")
        return stf_state.efficacy_for_mode(condition, stf_cfg)
    if condition == "baseline":
        return stf_cfg.u_baseline
    if condition == "static_gain":
        if static_gain is None:
            raise ValueError("static_gain condition requires p_match")
        return float(static_gain)
    if condition == "frozen_profile":
        if frozen_profile is None:
            raise ValueError("frozen_profile condition requires a per-E-neuron efficacy profile")
        return frozen_profile
    raise ValueError(f"Unknown E->I efficacy condition: {condition}")
