from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import csv

from config.phase13 import Phase13ConfirmatoryConfig
from config.schema import ProjectConfig
from experiment.mechanism_map import combo_id, config_for_combo
from io_utils import load_result, read_json, save_result, write_json
from model import simulate

# Frozen Phase-10 zero-noise Full p_match at the current working point.
# Used when Phase-8 discovery files are not supplied and the run is anchor-only.
ANCHOR_P_MATCH = {"tf_1000_tr_20": {"dynamic": 0.7042313040351903}}


def noise_id(noise_pa: float) -> str:
    if float(noise_pa).is_integer():
        return f"noise_{int(noise_pa):03d}"
    return f"noise_{noise_pa:g}".replace(".", "p")


def result_path(
    root: Path,
    cid: str,
    mode: str,
    condition_kind: str,
    noise_pa: float,
    seed: int,
) -> Path:
    label = mode if condition_kind == "temporal" else f"{mode}_static"
    return root / "combos" / cid / noise_id(noise_pa) / label / f"seed_{seed:04d}.npz"


def load_frozen_controls(
    confirm: Phase13ConfirmatoryConfig,
    base_cfg: ProjectConfig,
    phase8_results: str | Path | None,
) -> tuple[dict[str, dict[str, float]], dict[str, dict]]:
    """Freeze zero-noise mean-gain controls. Never refit on Phase-13 data."""
    if phase8_results:
        root = Path(phase8_results)
        exp_path = root / "experiment_manifest.json"
        csv_path = root / "analysis" / "combo_summary.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Phase-13 full/partial grid needs {csv_path}")
        exp = read_json(exp_path) if exp_path.exists() else {}
        if exp and abs(float(exp.get("u_baseline_fixed", base_cfg.stf.u_baseline)) - base_cfg.stf.u_baseline) > 1e-12:
            raise ValueError("Phase-8 U baseline differs from the current base config.")
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            rows = {row["combo_id"]: row for row in csv.DictReader(f)}
        controls: dict[str, dict[str, float]] = {}
        context: dict[str, dict] = {}
        for tau_f, tau_r in confirm.combos:
            cid = combo_id(tau_f, tau_r)
            if cid not in rows:
                raise KeyError(f"Phase-8 discovery results lack {cid}")
            row = rows[cid]
            controls[cid] = {}
            for mode in confirm.temporal_modes:
                key = f"{mode}_p_match_theta"
                if key not in row or row[key] == "":
                    raise KeyError(f"{cid} lacks {key}")
                p = float(row[key])
                if not 0.0 <= p <= 1.0:
                    raise ValueError(f"Invalid p_match {p} for {cid}/{mode}")
                controls[cid][mode] = p
            context[cid] = {
                "source": "phase8_zero_noise_p_match_theta",
                "tau_facilitation_ms": float(row["tau_facilitation_ms"]),
                "tau_recovery_ms": float(row["tau_recovery_ms"]),
                "dynamic_frequency_hz": float(row.get("dynamic_frequency_hz", "nan")),
                "dynamic_theta_fraction": float(row.get("dynamic_theta_fraction", "nan")),
                "facilitation_only_frequency_hz": float(row.get("facilitation_only_frequency_hz", "nan")),
                "facilitation_only_theta_fraction": float(row.get("facilitation_only_theta_fraction", "nan")),
                "depletion_only_frequency_hz": float(row.get("depletion_only_frequency_hz", "nan")),
                "depletion_only_theta_fraction": float(row.get("depletion_only_theta_fraction", "nan")),
            }
        return controls, context

    # Anchor-only fallback from the frozen Phase-10 operating point.
    controls = {}
    context = {}
    for tau_f, tau_r in confirm.combos:
        cid = combo_id(tau_f, tau_r)
        if cid not in ANCHOR_P_MATCH:
            raise FileNotFoundError(
                f"No Phase-8 results and no hardcoded p_match for {cid}. "
                "Pass --phase8-results or restrict the run to the 1000/20 dynamic anchor."
            )
        for mode in confirm.temporal_modes:
            if mode not in ANCHOR_P_MATCH[cid]:
                raise FileNotFoundError(
                    f"Hardcoded fallback has no p_match for {cid}/{mode}. Provide --phase8-results."
                )
        controls[cid] = dict(ANCHOR_P_MATCH[cid])
        context[cid] = {
            "source": "phase10_anchor_full_p_match_theta",
            "tau_facilitation_ms": float(tau_f),
            "tau_recovery_ms": float(tau_r),
        }
    return controls, context


def _simulate_task(cfg, condition, seed, noise_pa, static_gain, path, resume) -> str:
    out = Path(path)
    if resume and out.exists():
        load_result(out)
        return path
    result = simulate(
        cfg,
        condition=condition,
        seed=int(seed),
        noise_pa=float(noise_pa),
        static_gain=static_gain,
    )
    save_result(result, out)
    return path


def _run_parallel(tasks: list[tuple], workers: int, label: str) -> None:
    total = len(tasks)
    if workers <= 1:
        for i, task in enumerate(tasks, 1):
            _simulate_task(*task)
            if i % 10 == 0 or i == total:
                print(f"[{label}] {i}/{total}")
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_simulate_task, *task) for task in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            fut.result()
            if i % 25 == 0 or i == total:
                print(f"[{label}] {i}/{total}")


def run_phase13_attractor_confirmation(
    base_cfg: ProjectConfig,
    confirm: Phase13ConfirmatoryConfig,
    *,
    output_dir: str | Path,
    phase8_results: str | Path | None = None,
    workers: int = 1,
    resume: bool = True,
) -> dict:
    base_cfg.validate()
    confirm.validate()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    p_match, discovery_context = load_frozen_controls(confirm, base_cfg, phase8_results)

    manifest = {
        "experiment": "phase13_attractor_noise_confirmation",
        "scientific_question": (
            "After Phase-12 showed weak trace restoring and no short-time D_short advantage, "
            "does full E->I STP still reduce 7 s noise-induced displacement relative to a "
            "zero-noise mean-gain control that is frozen across 0/30/60/90 pA?"
        ),
        "confirmatory_seeds": list(confirm.seeds),
        "tau_facilitation_ms": list(confirm.tau_facilitation_ms),
        "tau_recovery_ms": list(confirm.tau_recovery_ms),
        "noise_levels_pa": list(confirm.noise_levels_pa),
        "temporal_modes": list(confirm.temporal_modes),
        "primary_confirmatory_anchor": {
            "tau_facilitation_ms": confirm.primary_tau_facilitation_ms,
            "tau_recovery_ms": confirm.primary_tau_recovery_ms,
            "mode": confirm.primary_mode,
        },
        "control_rule": (
            "Use a frozen zero-noise p_match at every noise level. "
            "Do not refit on Phase-13 data. Prefer Phase-8 *_p_match_theta; "
            "the 1000/20 dynamic anchor may use the Phase-10 Full p_match."
        ),
        "fixed_zero_noise_p_match": p_match,
        "discovery_context": discovery_context,
        "expected_run_count": confirm.expected_run_count(),
        "resume": bool(resume),
    }
    write_json(manifest, root / "experiment_manifest.json")

    tasks: list[tuple] = []
    for tau_f, tau_r in confirm.combos:
        cid = combo_id(tau_f, tau_r)
        cfg = config_for_combo(base_cfg, tau_f, tau_r)
        write_json(
            {
                "combo_id": cid,
                "tau_facilitation_ms": float(tau_f),
                "tau_recovery_ms": float(tau_r),
                "frozen_p_match": p_match[cid],
            },
            root / "combos" / cid / "manifest.json",
        )
        for mode in confirm.temporal_modes:
            fixed_gain = float(p_match[cid][mode])
            for noise in confirm.noise_levels_pa:
                for seed in confirm.seeds:
                    tasks.append((
                        cfg, mode, int(seed), float(noise), None,
                        str(result_path(root, cid, mode, "temporal", noise, seed)), resume,
                    ))
                    tasks.append((
                        cfg, "static_gain", int(seed), float(noise), fixed_gain,
                        str(result_path(root, cid, mode, "static", noise, seed)), resume,
                    ))
    if len(tasks) != confirm.expected_run_count():
        raise RuntimeError(f"task count {len(tasks)} != {confirm.expected_run_count()}")
    _run_parallel(tasks, workers, "phase13")
    manifest["completed"] = True
    manifest["completed_run_count"] = len(tasks)
    write_json(manifest, root / "experiment_manifest.json")
    return manifest
