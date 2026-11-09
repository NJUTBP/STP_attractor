from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from config.phase8 import Phase8SweepConfig
from config.schema import ProjectConfig
from experiment.mechanism_map import combo_id, config_for_combo
from io_utils import load_result, save_result, read_json, write_json
from model import simulate

HYBRID_MODES = ("u_output_only", "u_release_only", "mean_u_release")


def _path(root: Path, cid: str, stage: str, mode: str, seed: int) -> Path:
    return root / "combos" / cid / stage / mode / f"seed_{seed:04d}.npz"


def _simulate_task(
    cfg: ProjectConfig,
    condition: str,
    seed: int,
    noise_pa: float,
    temporal_gain_scale: float,
    release_u_override: float | None,
    path: str,
    resume: bool,
) -> tuple[str, float, float, float, int]:
    out = Path(path)
    if resume and out.exists():
        r = load_result(out)
        return (
            path,
            r.gain_match_numerator,
            r.gain_match_denominator,
            r.resource_release_u_sum,
            r.resource_release_u_count,
        )
    r = simulate(
        cfg,
        condition=condition,
        seed=int(seed),
        noise_pa=float(noise_pa),
        temporal_gain_scale=float(temporal_gain_scale),
        resource_release_u_override=release_u_override,
    )
    save_result(r, out)
    return (
        path,
        r.gain_match_numerator,
        r.gain_match_denominator,
        r.resource_release_u_sum,
        r.resource_release_u_count,
    )


def _run_parallel(tasks: list[tuple], workers: int, label: str) -> dict[str, tuple[float, float, float, int]]:
    out: dict[str, tuple[float, float, float, int]] = {}
    total = len(tasks)
    if workers <= 1:
        for i, task in enumerate(tasks, 1):
            p, n, d, rs, rc = _simulate_task(*task)
            out[p] = (n, d, rs, rc)
            if i % 25 == 0 or i == total:
                print(f"[{label}] {i}/{total}")
        return out
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_simulate_task, *task) for task in tasks]
        for i, f in enumerate(as_completed(futures), 1):
            p, n, d, rs, rc = f.result()
            out[p] = (n, d, rs, rc)
            if i % 50 == 0 or i == total:
                print(f"[{label}] {i}/{total}")
    return out


def _pooled_gain(paths: Iterable[Path], contributions: dict[str, tuple[float, float, float, int]] | None = None) -> float:
    num = 0.0
    den = 0.0
    contributions = contributions or {}
    for p in paths:
        key = str(p)
        if key in contributions:
            n, d, _, _ = contributions[key]
        else:
            r = load_result(p)
            n, d = r.gain_match_numerator, r.gain_match_denominator
        num += float(n)
        den += float(d)
    if den <= 0:
        raise RuntimeError("Cannot compute pooled gain: denominator <= 0")
    return num / den


def _pooled_release_u(paths: Iterable[Path], contributions: dict[str, tuple[float, float, float, int]] | None = None) -> float:
    total = 0.0
    count = 0
    contributions = contributions or {}
    for p in paths:
        key = str(p)
        if key in contributions:
            _, _, s, c = contributions[key]
        else:
            r = load_result(p)
            s, c = r.resource_release_u_sum, r.resource_release_u_count
        total += float(s)
        count += int(c)
    if count <= 0:
        raise RuntimeError("No resource-release spike events were recorded.")
    return total / count


def run_phase10_facilitation_coupling_map(
    base_cfg: ProjectConfig,
    sweep: Phase8SweepConfig,
    *,
    source_phase8_dir: str | Path,
    source_phase9_dir: str | Path,
    output_dir: str | Path,
    workers: int = 1,
    resume: bool = True,
) -> dict:
    """Large factorial decomposition of facilitation's two mechanistic routes.

    Existing data already provide:
      both routes ON  = Full STP (Phase 8)
      both routes OFF = gain-rescued depletion (Phase 9)

    Phase 10 adds three controls across the full 42-point time-scale grid:
      u_output_only:  dynamic u multiplies output, resource release uses fixed U.
      u_release_only: dynamic u controls resource release, output uses U*R.
      mean_u_release: resource release uses Full-STP's pooled mean spike-event u,
                      output uses U*R, with no dynamic u fluctuations.

    Every new mode is independently mean-gain matched to Full STP using two
    pooled 0-pA calibration waves, then the same final scale is used at 0 and
    60 pA.  This separates direct output modulation from u->R coupling while
    preserving the paired-noise design.
    """
    base_cfg.validate()
    sweep.validate()
    source8 = Path(source_phase8_dir)
    source9 = Path(source_phase9_dir)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(x) for x in sweep.seeds)

    combos = []
    for tau_f, tau_r in sweep.combos:
        cid = combo_id(tau_f, tau_r)
        src_manifest = read_json(source8 / "combos" / cid / "manifest.json")
        target = float(src_manifest["p_match"]["theta"]["dynamic"])
        cfg = config_for_combo(base_cfg, tau_f, tau_r)
        combos.append((cid, float(tau_f), float(tau_r), cfg, target))

    manifest = {
        "experiment": "phase10_facilitation_route_factorial",
        "scientific_question": (
            "Does dynamic facilitation create Full-STP theta mainly through its direct u(t) output multiplier, "
            "through facilitation-dependent resource depletion u(t)->R(t), or through the interaction of both?"
        ),
        "source_phase8_dir": str(source8.resolve()),
        "source_phase9_dir": str(source9.resolve()),
        "hybrid_modes": list(HYBRID_MODES),
        "seeds": list(seeds),
        "combo_count": len(combos),
        "theta_noise_pa": float(sweep.theta_noise_pa),
        "attractor_noise_pa": float(sweep.attractor_noise_pa),
        "full_probe_runs": len(combos) * len(seeds),
        "calibration_wave_runs": len(combos) * len(seeds) * len(HYBRID_MODES),
        "calibration_waves": 2,
        "final_runs": len(combos) * len(seeds) * len(HYBRID_MODES) * 2,
        "expected_new_run_count": (
            len(combos) * len(seeds)
            + 2 * len(combos) * len(seeds) * len(HYBRID_MODES)
            + 2 * len(combos) * len(seeds) * len(HYBRID_MODES)
        ),
        "mean_u_release_definition": (
            "pooled spike-event-weighted release probability actually used by Full STP during 1-7 s, "
            "after the spike-triggered facilitation update"
        ),
        "gain_matching": (
            "each new mode is independently matched to Phase-8 Full-STP 0-pA pooled E->I effective gain; "
            "two pooled calibration waves are used before final 0/60-pA runs"
        ),
    }
    write_json(manifest, root / "experiment_manifest.json")

    # Wave 0: rerun Full STP at 0 pA only to record the actual spike-event release u.
    probe_tasks: list[tuple] = []
    for cid, tau_f, tau_r, cfg, target in combos:
        for seed in seeds:
            probe_tasks.append((
                cfg, "dynamic", seed, float(sweep.theta_noise_pa), 1.0, None,
                str(_path(root, cid, "probe_full", "dynamic", seed)), resume,
            ))
    probe_contrib = _run_parallel(probe_tasks, workers, "phase10-full-probe")

    combo_meta: dict[str, dict] = {}
    for cid, tau_f, tau_r, cfg, target in combos:
        paths = [_path(root, cid, "probe_full", "dynamic", s) for s in seeds]
        mean_release_u = _pooled_release_u(paths, probe_contrib)
        probe_gain = _pooled_gain(paths, probe_contrib)
        combo_meta[cid] = {
            "combo_id": cid,
            "tau_facilitation_ms": tau_f,
            "tau_recovery_ms": tau_r,
            "target_full_p_match_theta": target,
            "probe_full_p_match_theta": probe_gain,
            "full_mean_spike_release_u": mean_release_u,
            "modes": {},
        }

    # Calibration wave 1: all new modes at unit postsynaptic scale.
    cal1_tasks: list[tuple] = []
    for cid, tau_f, tau_r, cfg, target in combos:
        release_mean = combo_meta[cid]["full_mean_spike_release_u"]
        for mode in HYBRID_MODES:
            override = release_mean if mode == "mean_u_release" else None
            for seed in seeds:
                cal1_tasks.append((
                    cfg, mode, seed, float(sweep.theta_noise_pa), 1.0, override,
                    str(_path(root, cid, "calibration1", mode, seed)), resume,
                ))
    cal1_contrib = _run_parallel(cal1_tasks, workers, "phase10-calibration1")

    # Initial correction per combo/mode.
    scale1: dict[tuple[str, str], float] = {}
    for cid, tau_f, tau_r, cfg, target in combos:
        for mode in HYBRID_MODES:
            paths = [_path(root, cid, "calibration1", mode, s) for s in seeds]
            obs = _pooled_gain(paths, cal1_contrib)
            alpha = target / obs
            scale1[(cid, mode)] = alpha
            combo_meta[cid]["modes"][mode] = {
                "calibration1_p_match": obs,
                "scale_after_calibration1": alpha,
            }

    # Calibration wave 2: evaluate the first correction under network feedback.
    cal2_tasks: list[tuple] = []
    for cid, tau_f, tau_r, cfg, target in combos:
        release_mean = combo_meta[cid]["full_mean_spike_release_u"]
        for mode in HYBRID_MODES:
            override = release_mean if mode == "mean_u_release" else None
            alpha = scale1[(cid, mode)]
            for seed in seeds:
                cal2_tasks.append((
                    cfg, mode, seed, float(sweep.theta_noise_pa), alpha, override,
                    str(_path(root, cid, "calibration2", mode, seed)), resume,
                ))
    cal2_contrib = _run_parallel(cal2_tasks, workers, "phase10-calibration2")

    final_scales: dict[tuple[str, str], float] = {}
    for cid, tau_f, tau_r, cfg, target in combos:
        for mode in HYBRID_MODES:
            paths = [_path(root, cid, "calibration2", mode, s) for s in seeds]
            obs2 = _pooled_gain(paths, cal2_contrib)
            alpha2 = scale1[(cid, mode)] * target / obs2
            final_scales[(cid, mode)] = alpha2
            m = combo_meta[cid]["modes"][mode]
            m["calibration2_p_match"] = obs2
            m["final_scale"] = alpha2
        write_json(combo_meta[cid], root / "combos" / cid / "manifest.json")

    # Final paired runs: one final scale per combo/mode is reused at 0 and 60 pA.
    final_tasks: list[tuple] = []
    for cid, tau_f, tau_r, cfg, target in combos:
        release_mean = combo_meta[cid]["full_mean_spike_release_u"]
        for regime, noise in (("theta", sweep.theta_noise_pa), ("attractor", sweep.attractor_noise_pa)):
            for mode in HYBRID_MODES:
                override = release_mean if mode == "mean_u_release" else None
                alpha = final_scales[(cid, mode)]
                for seed in seeds:
                    final_tasks.append((
                        cfg, mode, seed, float(noise), alpha, override,
                        str(_path(root, cid, regime, mode, seed)), resume,
                    ))
    _run_parallel(final_tasks, workers, "phase10-final")

    manifest["completed"] = True
    manifest["combo_manifests"] = [f"combos/{cid}/manifest.json" for cid, *_ in combos]
    write_json(manifest, root / "experiment_manifest.json")
    return manifest
