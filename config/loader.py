from __future__ import annotations

from dataclasses import fields
from pathlib import Path
try:
    import tomllib          # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib # Python 3.10 及以下

from .schema import (
    AnalysisConfig,
    ExperimentConfig,
    NetworkConfig,
    ProjectConfig,
    STFConfig,
    SimulationConfig,
)


def _build(cls, data: dict | None):
    data = data or {}
    valid = {f.name for f in fields(cls)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)

    # TOML arrays become lists; dataclasses use tuples for immutable experiment metadata.
    exp = dict(raw.get("experiment", {}))
    for key in ("attractor_seeds", "theta_seeds", "conditions"):
        if key in exp:
            exp[key] = tuple(exp[key])
    sim = dict(raw.get("simulation", {}))
    if "cue_centers_deg" in sim:
        sim["cue_centers_deg"] = tuple(sim["cue_centers_deg"])

    cfg = ProjectConfig(
        network=_build(NetworkConfig, raw.get("network")),
        stf=_build(STFConfig, raw.get("stf")),
        simulation=_build(SimulationConfig, sim),
        analysis=_build(AnalysisConfig, raw.get("analysis")),
        experiment=_build(ExperimentConfig, exp),
    )
    cfg.validate()
    return cfg
