from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
try:
    import tomllib          # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib # Python 3.10 及以下


@dataclass(frozen=True)
class Phase8SweepConfig:
    tau_facilitation_ms: tuple[float, ...]
    tau_recovery_ms: tuple[float, ...]
    seeds: tuple[int, ...]
    theta_noise_pa: float = 0.0
    attractor_noise_pa: float = 60.0
    temporal_modes: tuple[str, ...] = ("dynamic", "facilitation_only", "depletion_only")
    run_frozen_profile: bool = True
    stationary_abs_shift_cutoff_deg: float = 10.0

    def validate(self) -> None:
        if not self.tau_facilitation_ms or not self.tau_recovery_ms:
            raise ValueError("Phase-8 sweep grids must be non-empty.")
        if not self.seeds:
            raise ValueError("Phase-8 sweep needs at least one seed.")
        if any(x <= 0 for x in self.tau_facilitation_ms + self.tau_recovery_ms):
            raise ValueError("STP time constants must be positive.")
        allowed = {"dynamic", "facilitation_only", "depletion_only"}
        unknown = set(self.temporal_modes) - allowed
        if unknown:
            raise ValueError(f"Unknown temporal modes: {sorted(unknown)}")
        if "dynamic" not in self.temporal_modes:
            raise ValueError("Phase-8 requires the full dynamic mode.")

    @property
    def combos(self) -> tuple[tuple[float, float], ...]:
        return tuple(product(self.tau_facilitation_ms, self.tau_recovery_ms))

    def expected_run_count(self) -> int:
        # Baseline is shared across the whole grid: theta + attractor, one run per seed each.
        baseline = 2 * len(self.seeds)
        per_combo_temporal = 2 * len(self.temporal_modes) * len(self.seeds)  # theta + attractor temporal runs
        per_combo_matched = 2 * len(self.temporal_modes) * len(self.seeds)   # theta + attractor static matched
        per_combo_frozen = len(self.seeds) if self.run_frozen_profile else 0 # theta only
        return baseline + len(self.combos) * (per_combo_temporal + per_combo_matched + per_combo_frozen)


def load_phase8_sweep(path: str | Path) -> Phase8SweepConfig:
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)
    data = dict(raw.get("sweep", {}))
    required = {"tau_facilitation_ms", "tau_recovery_ms", "seeds"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"Missing Phase-8 sweep keys: {sorted(missing)}")
    allowed = {
        "tau_facilitation_ms", "tau_recovery_ms", "seeds", "theta_noise_pa",
        "attractor_noise_pa", "temporal_modes", "run_frozen_profile",
        "stationary_abs_shift_cutoff_deg",
    }
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown Phase-8 sweep keys: {sorted(unknown)}")
    for key in ("tau_facilitation_ms", "tau_recovery_ms", "seeds", "temporal_modes"):
        if key in data:
            data[key] = tuple(data[key])
    cfg = Phase8SweepConfig(**data)
    cfg.validate()
    return cfg
