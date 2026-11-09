from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from experiment.phase11_pilot import (
    PILOT_SEEDS,
    checkpoint_ms_for_seed,
    checkpoint_path,
    mode_kwargs,
)
from io_utils import save_result, write_json
from model.simulator import run_segment
from model.state import aligned_roll_bins, apply_named_shift, load_checkpoint, save_checkpoint

MODULE_A_CONDITIONS = ("dynamic", "static_gain", "u_release_only", "u_output_only")
MODULE_B_CONDITIONS = ("dynamic", "static_gain", "u_release_only")
NOISE_PA = (0.0, 15.0, 30.0, 60.0)
NOISE_DRAWS = tuple(range(8))
DIFFUSION_HORIZON_MS = 2000.0
SMALL_DELTA_DEG = 11.25
MODULE_B_SHIFTS = ("no_shift", "activity_shift", "co_shift")
MODULE_B_END_MS = 8000.0


@dataclass(frozen=True)
class DiffusionJob:
    condition: str
    seed: int
    noise_pa: float
    noise_draw: int
    checkpoint_ms: float
    end_ms: float

    @property
    def job_id(self) -> str:
        return (
            f"{self.condition}__seed{self.seed:04d}__n{self.noise_pa:g}__draw{self.noise_draw:02d}"
        )


@dataclass(frozen=True)
class ShiftJob:
    condition: str
    seed: int
    shift_kind: str
    delta_deg: float
    checkpoint_ms: float

    @property
    def job_id(self) -> str:
        sign = "p" if self.delta_deg >= 0 else "m"
        mag = f"{abs(self.delta_deg):g}".replace(".", "p")
        return f"{self.condition}__seed{self.seed:04d}__{self.shift_kind}__{sign}{mag}"


def iter_module_a_jobs() -> list[DiffusionJob]:
    jobs: list[DiffusionJob] = []
    for seed in PILOT_SEEDS:
        t0 = checkpoint_ms_for_seed(seed)
        end_ms = t0 + DIFFUSION_HORIZON_MS
        for condition in MODULE_A_CONDITIONS:
            jobs.append(DiffusionJob(condition, seed, 0.0, 0, t0, end_ms))
            for noise_pa in NOISE_PA:
                if noise_pa == 0.0:
                    continue
                for draw in NOISE_DRAWS:
                    jobs.append(DiffusionJob(condition, seed, float(noise_pa), draw, t0, end_ms))
    return jobs


def iter_module_b_jobs() -> list[ShiftJob]:
    jobs: list[ShiftJob] = []
    for seed in PILOT_SEEDS:
        t0 = checkpoint_ms_for_seed(seed)
        for condition in MODULE_B_CONDITIONS:
            jobs.append(ShiftJob(condition, seed, "no_shift", 0.0, t0))
            for kind in ("activity_shift", "co_shift"):
                for delta in (SMALL_DELTA_DEG, -SMALL_DELTA_DEG):
                    jobs.append(ShiftJob(condition, seed, kind, float(delta), t0))
    return jobs


def diffusion_path(root: Path, job: DiffusionJob) -> Path:
    return root / "module_a" / job.condition / f"{job.job_id}.npz"


def shift_path(root: Path, job: ShiftJob) -> Path:
    return root / "module_b" / job.condition / f"{job.job_id}.npz"


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


def _diffusion_task(payload: dict) -> str:
    from config import load_config

    out = Path(payload["out_path"])
    if payload["resume"] and out.exists():
        return str(out)
    cfg = load_config(payload["config_path"])
    state = load_checkpoint(payload["ckpt_path"])
    kw = mode_kwargs(payload["condition"])
    result, _ = run_segment(
        cfg,
        condition=payload["condition"],
        seed=int(payload["seed"]),
        noise_pa=float(payload["noise_pa"]),
        start_state=state,
        end_ms=float(payload["end_ms"]),
        noise_draw=int(payload["noise_draw"]),
        **kw,
    )
    save_result(result, out)
    return str(out)


def _shift_task(payload: dict) -> str:
    from config import load_config

    out = Path(payload["out_path"])
    if payload["resume"] and out.exists():
        return str(out)
    cfg = load_config(payload["config_path"])
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
        end_ms=MODULE_B_END_MS,
        **kw,
    )
    save_result(result, out)
    return str(out)


def _run_pool(tasks: list[dict], workers: int, label: str, fn) -> None:
    total = len(tasks)
    if total == 0:
        return
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


def _checkpoint_payloads(config_path: str, ckpt_root: Path, conditions: tuple[str, ...], resume: bool) -> list[dict]:
    payloads = []
    seen = set()
    for seed in PILOT_SEEDS:
        t0 = checkpoint_ms_for_seed(seed)
        for condition in conditions:
            key = (condition, seed)
            if key in seen:
                continue
            seen.add(key)
            payloads.append({
                "config_path": config_path,
                "condition": condition,
                "seed": seed,
                "checkpoint_ms": t0,
                "ckpt_path": str(checkpoint_path(ckpt_root, condition, seed)),
                "resume": resume,
            })
    return payloads


def run_phase12(
    *,
    output_dir: str | Path,
    config_path: str,
    checkpoint_dir: str | Path | None = None,
    workers: int = 1,
    resume: bool = True,
    run_module_a: bool = True,
    run_module_b: bool = True,
) -> dict:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    ckpt_root = Path(checkpoint_dir) if checkpoint_dir else root
    conditions = []
    if run_module_a:
        conditions.extend(MODULE_A_CONDITIONS)
    if run_module_b:
        conditions.extend(MODULE_B_CONDITIONS)
    conditions_u = tuple(dict.fromkeys(conditions))

    pre = _checkpoint_payloads(config_path, ckpt_root, conditions_u, resume)
    print(f"Phase-12 checkpoints: {len(pre)} (dir={ckpt_root})")
    _run_pool(pre, workers, "checkpoint", _ensure_checkpoint)

    a_jobs = iter_module_a_jobs() if run_module_a else []
    b_jobs = iter_module_b_jobs() if run_module_b else []
    a_payloads = [
        {
            "config_path": config_path,
            "condition": j.condition,
            "seed": j.seed,
            "noise_pa": j.noise_pa,
            "noise_draw": j.noise_draw,
            "end_ms": j.end_ms,
            "ckpt_path": str(checkpoint_path(ckpt_root, j.condition, j.seed)),
            "out_path": str(diffusion_path(root, j)),
            "resume": resume,
        }
        for j in a_jobs
    ]
    b_payloads = [
        {
            "config_path": config_path,
            "condition": j.condition,
            "seed": j.seed,
            "shift_kind": j.shift_kind,
            "delta_deg": j.delta_deg,
            "ckpt_path": str(checkpoint_path(ckpt_root, j.condition, j.seed)),
            "out_path": str(shift_path(root, j)),
            "resume": resume,
        }
        for j in b_jobs
    ]
    print(f"Phase-12 module A (diffusion): {len(a_payloads)}")
    _run_pool(a_payloads, workers, "module_a", _diffusion_task)
    print(f"Phase-12 module B (11.25 deg): {len(b_payloads)}")
    _run_pool(b_payloads, workers, "module_b", _shift_task)

    manifest = {
        "experiment": "phase12_attractor",
        "checkpoint_dir": str(ckpt_root),
        "module_a_jobs": len(a_jobs),
        "module_b_jobs": len(b_jobs),
        "noise_pa": list(NOISE_PA),
        "small_delta_deg": SMALL_DELTA_DEG,
        "diffusion_horizon_ms": DIFFUSION_HORIZON_MS,
        "module_a": [asdict(j) | {"job_id": j.job_id} for j in a_jobs],
        "module_b": [asdict(j) | {"job_id": j.job_id} for j in b_jobs],
    }
    write_json(manifest, root / "experiment_manifest.json")
    return manifest
