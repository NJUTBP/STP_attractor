from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from config.schema import ProjectConfig
from experiment.phase11_pilot import PILOT_SEEDS, checkpoint_ms_for_seed, checkpoint_path, mode_kwargs
from io_utils import save_result, write_json
from model.simulator import run_segment
from model.state import aligned_roll_bins, apply_named_shift, load_checkpoint, save_checkpoint

# P1/Phase-14 is deliberately anchor-only and mechanism-focused. It reuses the
# Phase-11 anchor checkpoint design rather than opening another parameter grid.
PHASE14_CONDITIONS = ("dynamic", "static_gain", "u_output_only", "u_release_only")
PHASE14_DELTAS_DEG = (5.625, 11.25, 22.5)
PHASE14_SURGERY_DELTAS_DEG = (11.25, 22.5)
PHASE14_END_AFTER_CHECKPOINT_MS = 2500.0

# Every condition gets a multi-displacement activity shift. Full dynamic alone
# receives selective state surgery, because the scientific question is which
# Full-STP state variable carries the spatial reference.
DYNAMIC_SURGERIES = ("activity_u_shift", "activity_r_shift", "co_shift")


@dataclass(frozen=True)
class TetherBranch:
    condition: str
    seed: int
    shift_kind: str
    delta_deg: float
    checkpoint_ms: float

    @property
    def branch_id(self) -> str:
        if self.shift_kind == "no_shift":
            return f"{self.condition}__seed{self.seed:04d}__no_shift"
        sign = "p" if self.delta_deg >= 0 else "m"
        mag = f"{abs(self.delta_deg):g}".replace(".", "p")
        return f"{self.condition}__seed{self.seed:04d}__{self.shift_kind}__{sign}{mag}"


def iter_phase14_branches() -> list[TetherBranch]:
    out: list[TetherBranch] = []
    for seed in PILOT_SEEDS:
        t0 = checkpoint_ms_for_seed(seed)
        for condition in PHASE14_CONDITIONS:
            out.append(TetherBranch(condition, seed, "no_shift", 0.0, t0))
            for delta_mag in PHASE14_DELTAS_DEG:
                for delta in (delta_mag, -delta_mag):
                    out.append(TetherBranch(condition, seed, "activity_shift", float(delta), t0))
        # Selective Full-STP state surgery at the two informative non-tiny shifts.
        for kind in DYNAMIC_SURGERIES:
            for delta_mag in PHASE14_SURGERY_DELTAS_DEG:
                for delta in (delta_mag, -delta_mag):
                    out.append(TetherBranch("dynamic", seed, kind, float(delta), t0))

    ids = [b.branch_id for b in out]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Phase-14 branch construction produced duplicate branch IDs")
    return out


def branch_path(root: Path, branch: TetherBranch) -> Path:
    return root / "branches" / branch.condition / f"{branch.branch_id}.npz"


def _ensure_checkpoint(payload: dict) -> str:
    from config import load_config

    out = Path(payload["ckpt_path"])
    if payload["resume"] and out.exists():
        return str(out)
    cfg = load_config(payload["config_path"])
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

    out = Path(payload["out_path"])
    if payload["resume"] and out.exists():
        return str(out)
    cfg = load_config(payload["config_path"])
    state = load_checkpoint(payload["ckpt_path"])
    kind = str(payload["shift_kind"])
    delta = float(payload["delta_deg"])
    if kind == "no_shift":
        branched = state.copy()
    else:
        e_bins, i_bins = aligned_roll_bins(delta, cfg.network.ne, cfg.network.ni)
        branched = apply_named_shift(state, kind, e_bins, i_bins)

    condition = str(payload["condition"])
    kw = mode_kwargs(condition)
    result, _ = run_segment(
        cfg,
        condition=condition,
        seed=int(payload["seed"]),
        noise_pa=0.0,
        start_state=branched,
        end_ms=float(payload["end_ms"]),
        record_stp_trace=(condition != "static_gain"),
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


def run_phase14_tether(
    cfg: ProjectConfig,
    *,
    output_dir: str | Path,
    config_path: str,
    checkpoint_dir: str | Path | None = None,
    workers: int = 1,
    resume: bool = True,
) -> dict:
    if abs(cfg.stf.tau_facilitation_ms - 1000.0) > 1e-9 or abs(cfg.stf.tau_recovery_ms - 20.0) > 1e-9:
        raise ValueError("Phase-14 tether experiment is frozen to the tf=1000 ms, tr=20 ms anchor")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    ckpt_root = Path(checkpoint_dir) if checkpoint_dir else root
    branches = iter_phase14_branches()

    pre = []
    seen = set()
    for b in branches:
        key = (b.condition, b.seed)
        if key in seen:
            continue
        seen.add(key)
        pre.append({
            "config_path": config_path,
            "condition": b.condition,
            "seed": b.seed,
            "checkpoint_ms": b.checkpoint_ms,
            "ckpt_path": str(checkpoint_path(ckpt_root, b.condition, b.seed)),
            "resume": resume,
        })
    print(f"Phase-14 checkpoints: {len(pre)} (dir={ckpt_root})")
    _run_pool(pre, workers, "checkpoint", _ensure_checkpoint)

    jobs = []
    for b in branches:
        jobs.append({
            "config_path": config_path,
            "condition": b.condition,
            "seed": b.seed,
            "shift_kind": b.shift_kind,
            "delta_deg": b.delta_deg,
            "end_ms": b.checkpoint_ms + PHASE14_END_AFTER_CHECKPOINT_MS,
            "ckpt_path": str(checkpoint_path(ckpt_root, b.condition, b.seed)),
            "out_path": str(branch_path(root, b)),
            "resume": resume,
        })
    print(f"Phase-14 branches: {len(jobs)}")
    _run_pool(jobs, workers, "branch", _branch_task)

    manifest = {
        "experiment": "phase14_attractor_tether",
        "scientific_goal": (
            "Measure multi-displacement restoring dynamics and identify whether u or R carries the Full-STP spatial trace."
        ),
        "anchor": {"tau_facilitation_ms": 1000.0, "tau_recovery_ms": 20.0},
        "conditions": list(PHASE14_CONDITIONS),
        "seeds": list(PILOT_SEEDS),
        "activity_shift_deltas_deg": list(PHASE14_DELTAS_DEG),
        "dynamic_surgery_deltas_deg": list(PHASE14_SURGERY_DELTAS_DEG),
        "dynamic_surgeries": list(DYNAMIC_SURGERIES),
        "continuation_ms": PHASE14_END_AFTER_CHECKPOINT_MS,
        "branch_count": len(branches),
        "checkpoint_count": len(pre),
        "recorded_stp_trace": True,
        "branches": [asdict(b) | {"branch_id": b.branch_id} for b in branches],
        "completed": True,
    }
    write_json(manifest, root / "experiment_manifest.json")
    return manifest
