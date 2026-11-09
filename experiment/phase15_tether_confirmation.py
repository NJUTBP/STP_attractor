from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from config.schema import ProjectConfig
from experiment.phase11_pilot import checkpoint_ms_for_seed, checkpoint_path, mode_kwargs
from io_utils import save_result, write_json
from model.simulator import run_segment
from model.state import aligned_roll_bins, apply_named_shift, load_checkpoint, save_checkpoint

# Phase-15 is a focused independent-seed confirmation of the Phase-14 discovery.
# No parameter grid, no noise sweep, and no refitting of Phase-10 anchor scales.
PHASE15_SEEDS = tuple(range(200, 216))
PHASE15_CONDITIONS = ("dynamic", "static_gain", "u_output_only", "u_release_only")
PHASE15_DELTAS_DEG = (11.25, 22.5)
PHASE15_SURGERIES = ("activity_u_shift", "activity_r_shift", "co_shift")
PHASE15_CONTINUATION_MS = 2500.0


@dataclass(frozen=True)
class ConfirmBranch:
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


def iter_phase15_branches() -> list[ConfirmBranch]:
    out: list[ConfirmBranch] = []
    for seed in PHASE15_SEEDS:
        t0 = checkpoint_ms_for_seed(seed)
        for condition in PHASE15_CONDITIONS:
            out.append(ConfirmBranch(condition, seed, "no_shift", 0.0, t0))
            for mag in PHASE15_DELTAS_DEG:
                for delta in (mag, -mag):
                    out.append(ConfirmBranch(condition, seed, "activity_shift", float(delta), t0))
        for kind in PHASE15_SURGERIES:
            for mag in PHASE15_DELTAS_DEG:
                for delta in (mag, -mag):
                    out.append(ConfirmBranch("dynamic", seed, kind, float(delta), t0))
    ids = [b.branch_id for b in out]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Phase-15 branch construction produced duplicate branch IDs")
    return out


def branch_path(root: Path, branch: ConfirmBranch) -> Path:
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


def run_phase15_tether_confirmation(
    cfg: ProjectConfig,
    *,
    output_dir: str | Path,
    config_path: str,
    workers: int = 1,
    resume: bool = True,
) -> dict:
    if abs(cfg.stf.tau_facilitation_ms - 1000.0) > 1e-9 or abs(cfg.stf.tau_recovery_ms - 20.0) > 1e-9:
        raise ValueError("Phase-15 confirmation is frozen to tf=1000 ms, tr=20 ms")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    branches = iter_phase15_branches()

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
            "ckpt_path": str(checkpoint_path(root, b.condition, b.seed)),
            "resume": resume,
        })
    print(f"Phase-15 independent checkpoints: {len(pre)}")
    _run_pool(pre, workers, "checkpoint", _ensure_checkpoint)

    jobs = []
    for b in branches:
        jobs.append({
            "config_path": config_path,
            "condition": b.condition,
            "seed": b.seed,
            "shift_kind": b.shift_kind,
            "delta_deg": b.delta_deg,
            "end_ms": b.checkpoint_ms + PHASE15_CONTINUATION_MS,
            "ckpt_path": str(checkpoint_path(root, b.condition, b.seed)),
            "out_path": str(branch_path(root, b)),
            "resume": resume,
        })
    print(f"Phase-15 branches: {len(jobs)}")
    _run_pool(jobs, workers, "branch", _branch_task)

    manifest = {
        "experiment": "phase15_tether_confirmation",
        "scientific_goal": (
            "Independently confirm that the spatial restoring trace is carried by u/direct-u-output rather than the u-to-R release route."
        ),
        "evidence_role": "confirmatory follow-up to Phase-14 discovery",
        "anchor": {"tau_facilitation_ms": 1000.0, "tau_recovery_ms": 20.0},
        "seeds": list(PHASE15_SEEDS),
        "conditions": list(PHASE15_CONDITIONS),
        "deltas_deg": list(PHASE15_DELTAS_DEG),
        "surgeries": list(PHASE15_SURGERIES),
        "primary_delta_deg": 11.25,
        "primary_readout": "R_2000ms",
        "primary_contrasts": [
            "dynamic activity_shift > dynamic activity_u_shift",
            "u_output_only activity_shift > u_release_only activity_shift",
        ],
        "supporting_contrasts": [
            "dynamic activity_shift > co_shift",
            "u_output_only activity_shift > static_gain activity_shift",
            "u trace retention > R trace retention at 250 ms after activity shift",
        ],
        "checkpoint_count": len(pre),
        "branch_count": len(branches),
        "continuation_ms": PHASE15_CONTINUATION_MS,
        "recorded_stp_trace": True,
        "branches": [asdict(b) | {"branch_id": b.branch_id} for b in branches],
        "completed": True,
    }
    write_json(manifest, root / "experiment_manifest.json")
    return manifest
