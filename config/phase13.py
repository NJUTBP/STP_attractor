from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
try:
    import tomllib          # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib # Python 3.10 及以下


_ALLOWED_MODES = ("dynamic", "facilitation_only", "depletion_only")


@dataclass(frozen=True)
class Phase13ConfirmatoryConfig:
    tau_facilitation_ms: tuple[float, ...]
    tau_recovery_ms: tuple[float, ...]
    seeds: tuple[int, ...]
    noise_levels_pa: tuple[float, ...]
    temporal_modes: tuple[str, ...] = ("dynamic",)
    stationary_abs_shift_cutoff_deg: float = 10.0
    localization_failure_fraction_cutoff: float = 0.2
    bootstrap_repeats: int = 10_000
    primary_tau_facilitation_ms: float = 1000.0
    primary_tau_recovery_ms: float = 20.0
    primary_mode: str = "dynamic"

    def validate(self) -> None:
        if not self.tau_facilitation_ms or not self.tau_recovery_ms:
            raise ValueError("Phase-13 parameter grids must be non-empty.")
        if any(x <= 0 for x in self.tau_facilitation_ms + self.tau_recovery_ms):
            raise ValueError("STP time constants must be positive.")
        if not self.seeds:
            raise ValueError("Phase-13 requires confirmatory seeds.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Phase-13 seeds must be unique.")
        if 0.0 not in self.noise_levels_pa:
            raise ValueError("noise_levels_pa must include 0 pA.")
        if tuple(sorted(set(self.noise_levels_pa))) != tuple(self.noise_levels_pa):
            raise ValueError("noise_levels_pa must be unique and sorted.")
        unknown = set(self.temporal_modes) - set(_ALLOWED_MODES)
        if unknown:
            raise ValueError(f"Unknown temporal modes: {sorted(unknown)}")
        if self.primary_tau_facilitation_ms not in self.tau_facilitation_ms:
            raise ValueError("Primary tau_f must be in the grid.")
        if self.primary_tau_recovery_ms not in self.tau_recovery_ms:
            raise ValueError("Primary tau_r must be in the grid.")
        if self.primary_mode not in self.temporal_modes:
            raise ValueError("primary_mode must be in temporal_modes.")

    @property
    def combos(self) -> tuple[tuple[float, float], ...]:
        return tuple(product(self.tau_facilitation_ms, self.tau_recovery_ms))

    def expected_run_count(self) -> int:
        return (
            len(self.combos)
            * len(self.temporal_modes)
            * 2
            * len(self.seeds)
            * len(self.noise_levels_pa)
        )


def load_phase13_confirmatory(path: str | Path) -> Phase13ConfirmatoryConfig:
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)
    data = dict(raw.get("confirmatory", {}))
    required = {"tau_facilitation_ms", "tau_recovery_ms", "seeds", "noise_levels_pa"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"Missing Phase-13 config keys: {sorted(missing)}")
    for key in ("tau_facilitation_ms", "tau_recovery_ms", "seeds", "noise_levels_pa", "temporal_modes"):
        if key in data:
            data[key] = tuple(data[key])
    cfg = Phase13ConfirmatoryConfig(**{
        k: v for k, v in data.items()
        if k in Phase13ConfirmatoryConfig.__dataclass_fields__
    })
    cfg.validate()
    return cfg
