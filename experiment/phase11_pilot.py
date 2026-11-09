from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from config.schema import ProjectConfig
from io_utils import save_result, write_json
from model.simulator import run_segment
from model.state import (
    aligned_roll_bins,
    apply_named_shift,
    load_checkpoint,
    save_checkpoint,
)

# Frozen Phase-10 operating-point scales (tf=1000, tr=20). Do not refit.
ANCHOR_FULL_P_MATCH = 0.7042313040351903
ANCHOR_MEAN_SPIKE_RELEASE_U = 0.888714275170488
ANCHOR_SCALES = {
    "dynamic": {"temporal_gain_scale": 1.0, "resource_release_u_override": None, "static_gain": None},
    "static_gain": {
        "temporal_gain_scale": 1.0,
        "resource_release_u_override": None,
        "static_gain": ANCHOR_FULL_P_MATCH,
    },
    "u_output_only": {
        "temporal_gain_scale": 0.8721143384065277,
        "resource_release_u_override": None,
        "static_gain": None,
    },
    "u_release_only": {
        "temporal_gain_scale": 2.1329119922475366,
        "resource_release_u_override": None,
        "static_gain": None,
    },
    "mean_u_release": {
        "temporal_gain_scale": 2.133982993354528,
        "resource_release_u_override": ANCHOR_MEAN_SPIKE_RELEASE_U,
        "static_gain": None,
    },
}

PILOT_MODES = tuple(ANCHOR_SCALES)
PILOT_DELTA_DEG = 33.75
PILOT_SEEDS = tuple(range(8))
PILOT_CHECKPOINT_MS = 3000.0
PILOT_JITTER_MS = 250.0
PILOT_END_MS = 8000.0
SHIFT_KINDS_STATEFUL = ("no_shift", "activity_shift", "co_shift", "trace_shift")
SHIFT_KINDS_STATIC = ("no_shift", "activity_shift")


def seed_jitter_ms(seed: int) -> float:
    return float((seed * 37) % 251)


def checkpoint_ms_for_seed(seed: int) -> float:
    return PILOT_CHECKPOINT_MS + seed_jitter_ms(seed)


def mode_kwargs(condition: str) -> dict:
    kw = dict(ANCHOR_SCALES[condition])
    return {
        "temporal_gain_scale": float(kw["temporal_gain_scale"]),
        "resource_release_u_override": kw["resource_release_u_override"],
        "static_gain": kw["static_gain"],
    }


@dataclass(frozen=True)
class PilotBranch:
    condition: str
    seed: int
    shift_kind: str
    delta_deg: float
    checkpoint_ms: float

    @property
    def branch_id(self) -> str:
        sign = "p" if self.delta_deg >= 0 else "m"
        mag = f"{abs(self.delta_deg):g}".replace(".", "p")
        return f"{self.condition}__seed{self.seed:04d}__{self.shift_kind}__{sign}{mag}"


def iter_pilot_branches() -> list[PilotBranch]:
    out: list[PilotBranch] = []
    for seed in PILOT_SEEDS:
        t0 = checkpoint_ms_for_seed(seed)
        for condition in PILOT_MODES:
            kinds = SHIFT_KINDS_STATIC if condition == "static_gain" else SHIFT_KINDS_STATEFUL
            for kind in kinds:
                deltas = (0.0,) if kind == "no_shift" else (PILOT_DELTA_DEG, -PILOT_DELTA_DEG)
                for delta in deltas:
                    out.append(PilotBranch(condition, seed, kind, float(delta), t0))
    return out


def checkpoint_path(root: Path, condition: str, seed: int) -> Path:
    return root / "checkpoints" / condition / f"seed_{seed:04d}.pkl"


def branch_path(root: Path, branch: PilotBranch) -> Path:
    return root / "branches" / branch.condition / f"{branch.branch_id}.npz"


def _prehistory_task(payload: dict) -> str:
    from config import load_config

    cfg = load_config(payload["config_path"])
    out = Path(payload["ckpt_path"])
    if payload["resume"] and out.exists():
        return str(out)
    kw = mode_kwargs(payload["condition"])
    _result, state = run_segment(
        cfg,
        condition=payload["condition"],
        seed=int(payload["seed"]),
        noise_pa=0.0,
        end_ms=float(payload["checkpoint_ms"]),
        **kw,
    )
    save_checkpoint(state, out)
    return str(out)


def _branch_task(payload: dict) -> str:
    from config import load_config

    cfg = load_config(payload["config_path"])
    out = Path(payload["out_path"])
    if payload["resume"] and out.exists():
        return str(out)
    state = load_checkpoint(payload["ckpt_path"])
    kind = payload["shift_kind"]
    delta = float(payload["delta_deg"])
    if kind == "no_shift" or abs(delta) < 1e-12:
        branched = state.copy()
    else:
        e_bins, i_bins = aligned_roll_bins(delta, cfg.network.ne, cfg.network.ni)
        branched = apply_named_shift(state, kind, e_bins, i_bins)
    kw = mode_kwargs(payload["condition"])
    result, _ = run_segment(
        cfg,
        condition=payload["condition"],
        seed=int(payload["seed"]),
        noise_pa=0.0,
        start_state=branched,
        end_ms=float(payload["end_ms"]),
        **kw,
    )
    save_result(result, out)
    return str(out)


def _run_pool(tasks: list[dict], workers: int, label: str, fn) -> None:
    total = len(tasks)
    if workers <= 1:
        for i, task in enumerate(tasks, 1):
            fn(task)
            if i % 10 == 0 or i == total:
                print(f"[{label}] {i}/{total}")
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(fn, task) for task in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            fut.result()
            if i % 10 == 0 or i == total:
                print(f"[{label}] {i}/{total}")


def run_phase11_anchor_pilot(
    cfg: ProjectConfig,
    *,
    output_dir: str | Path,
    config_path: str,
    workers: int = 1,
    resume: bool = True,
) -> dict:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    branches = iter_pilot_branches()
    pre = []
    seen = set()
    for b in branches:
        key = (b.condition, b.seed)
        if key in seen:
            continue
        seen.add(key)
        pre.append(
            {
                "config_path": config_path,
                "condition": b.condition,
                "seed": b.seed,
                "checkpoint_ms": b.checkpoint_ms,
                "ckpt_path": str(checkpoint_path(root, b.condition, b.seed)),
                "resume": resume,
            }
        )
    print(f"Phase-11 prehistories: {len(pre)}")
    _run_pool(pre, workers, "checkpoint", _prehistory_task)

    jobs = []
    for b in branches:
        jobs.append(
            {
                "config_path": config_path,
                "condition": b.condition,
                "seed": b.seed,
                "shift_kind": b.shift_kind,
                "delta_deg": b.delta_deg,
                "end_ms": PILOT_END_MS,
                "ckpt_path": str(checkpoint_path(root, b.condition, b.seed)),
                "out_path": str(branch_path(root, b)),
                "resume": resume,
            }
        )
    print(f"Phase-11 branches: {len(jobs)}")
    _run_pool(jobs, workers, "branch", _branch_task)

    manifest = {
        "experiment": "phase11_anchor_pilot",
        "combo": "tf_1000_tr_20",
        "delta_deg": PILOT_DELTA_DEG,
        "seeds": list(PILOT_SEEDS),
        "modes": list(PILOT_MODES),
        "prehistory_count": len(pre),
        "branch_count": len(branches),
        "scales": ANCHOR_SCALES,
        "branches": [asdict(b) | {"branch_id": b.branch_id} for b in branches],
        "completed": True,
    }
    write_json(manifest, root / "experiment_manifest.json")
    return manifest
