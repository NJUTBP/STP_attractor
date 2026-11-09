"""Phase 17 — physiological-regime STP sweep (authorised by the author on 2026-10-02).

Pre-registration: phase17/PREREGISTRATION.md (written before any physiological result).
No existing module is modified: this file only composes public functions of
model.simulate / model.simulator.run_segment / phase16.snapshots / io_utils.

Variants
  raw      temporal_gain_scale = 1 (frozen conductances; functional screen)
  matched  temporal_gain_scale calibrated so the pooled 0-pA E->I effective gain of Full STP equals the
           reference point's p_match (0.7042313); only the STP time course differs from the reference.
Stages
  gate         reference point must reproduce frozen Phase-8 / Phase-16D per-seed values
  full         raw theta (= calibration wave 0) -> matched calibration waves -> final theta + attractor
  routes       hybrid modes (Phase-10 logic), each gain-matched to the variant's Full STP
  restoration  Phase-16 module-D protocol (checkpoint 3 s + jitter, +-delta activity shifts, 2.5 s)
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # Python <= 3.10
    import tomli as tomllib

from config import load_config
from config.schema import ProjectConfig
from io_utils import load_result, read_json, save_result, write_json
from model import simulate

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = (0.4, 1000.0, 20.0)
ROUTE_CONDITION = {"u_output_only": "u_output_only", "u_release_only": "u_release_only",
                   "mean_u_release": "mean_u_release", "neither": "depletion_only"}


# ----------------------------------------------------------------------------- configuration
@dataclass(frozen=True)
class Phase17Config:
    base_config: Path
    anchor_reference: Path
    u_baseline: tuple
    tau_facilitation_ms: tuple
    tau_recovery_ms: tuple
    seeds: tuple
    theta_noise_pa: float
    attractor_noise_pa: float
    variants: tuple
    hybrid_modes: tuple
    include_neither: bool
    max_calibration_waves: int
    calibration_tolerance: float
    calibration_validity: float
    restoration: dict
    functional: dict
    decision: dict

    @property
    def combos(self) -> tuple:
        return tuple((float(u), float(tf), float(tr)) for u in self.u_baseline
                     for tf in self.tau_facilitation_ms for tr in self.tau_recovery_ms)

    @property
    def route_modes(self) -> tuple:
        return tuple(self.hybrid_modes) + (("neither",) if self.include_neither else ())


def load_phase17_config(path: str | Path) -> Phase17Config:
    path = Path(path)
    raw = tomllib.load(path.open("rb"))
    resolve = lambda p: (Path(p) if Path(p).is_absolute() else ROOT / p)
    g, r = raw["grid"], raw["run"]
    cfg = Phase17Config(
        base_config=resolve(raw["paths"]["base_config"]),
        anchor_reference=resolve(raw["paths"]["anchor_reference"]),
        u_baseline=tuple(float(x) for x in g["u_baseline"]),
        tau_facilitation_ms=tuple(float(x) for x in g["tau_facilitation_ms"]),
        tau_recovery_ms=tuple(float(x) for x in g["tau_recovery_ms"]),
        seeds=tuple(int(x) for x in r["seeds"]),
        theta_noise_pa=float(r["theta_noise_pa"]), attractor_noise_pa=float(r["attractor_noise_pa"]),
        variants=tuple(r["variants"]), hybrid_modes=tuple(r["hybrid_modes"]),
        include_neither=bool(r.get("include_neither", False)),
        max_calibration_waves=int(r["max_calibration_waves"]),
        calibration_tolerance=float(r["calibration_tolerance"]),
        calibration_validity=float(r["calibration_validity"]),
        restoration=dict(raw["restoration"]), functional=dict(raw["functional"]), decision=dict(raw["decision"]))
    bad = set(cfg.variants) - {"matched", "raw"}
    if bad:
        raise ValueError(f"unknown variants {bad}")
    for m in cfg.hybrid_modes:
        if m not in ROUTE_CONDITION:
            raise ValueError(f"unknown hybrid mode {m}")
    if any(not 0 < u <= 1 for u in cfg.u_baseline):
        raise ValueError("u_baseline must lie in (0, 1]")
    return cfg


def combo_id(u: float, tf: float, tr: float) -> str:
    return f"u_{u:g}_tf_{tf:g}_tr_{tr:g}".replace(".", "p")


def config_for(base: ProjectConfig, u: float, tf: float, tr: float) -> ProjectConfig:
    return replace(base, stf=replace(base.stf, u_baseline=float(u), tau_facilitation_ms=float(tf),
                                     tau_recovery_ms=float(tr)))


def target_p_match(cfg17: Phase17Config) -> float:
    return float(read_json(cfg17.anchor_reference)["full_p_match_theta"])


# ----------------------------------------------------------------------------- simulation tasks
def _sim_task(cfg, condition, seed, noise, scale, override, path, resume):
    out = Path(path)
    if resume and out.exists():
        r = load_result(out)
    else:
        r = simulate(cfg, condition=condition, seed=int(seed), noise_pa=float(noise),
                     temporal_gain_scale=float(scale), resource_release_u_override=override)
        save_result(r, out)
    return (path, float(r.gain_match_numerator), float(r.gain_match_denominator),
            float(r.resource_release_u_sum), int(r.resource_release_u_count))


def _run_parallel(tasks, workers, label):
    out = {}
    if not tasks:
        return out
    if workers <= 1:
        for i, t in enumerate(tasks, 1):
            p, *v = _sim_task(*t); out[p] = tuple(v)
            print(f"[{label}] {i}/{len(tasks)}", flush=True)
        return out
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_sim_task, *t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            p, *v = f.result(); out[p] = tuple(v)
            if i % 10 == 0 or i == len(tasks):
                print(f"[{label}] {i}/{len(tasks)}", flush=True)
    return out


def _seed_paths(d: Path, seeds) -> list[Path]:
    return [Path(d) / f"seed_{s:04d}.npz" for s in seeds]


def pooled_gain(paths, contrib=None) -> float:
    num = den = 0.0
    for p in paths:
        if contrib and str(p) in contrib:
            n, d, _, _ = contrib[str(p)]
        else:
            r = load_result(p); n, d = r.gain_match_numerator, r.gain_match_denominator
        num += n; den += d
    if den <= 0:
        raise RuntimeError("zero denominator in pooled gain")
    return num / den


def pooled_release_u(paths, contrib=None) -> float:
    s = 0.0; c = 0
    for p in paths:
        if contrib and str(p) in contrib:
            _, _, rs, rc = contrib[str(p)]
        else:
            r = load_result(p); rs, rc = r.resource_release_u_sum, r.resource_release_u_count
        s += rs; c += rc
    return s / c if c > 0 else float("nan")


# ----------------------------------------------------------------------------- calibration (lock-step)
def calibrate(items, cfg17: Phase17Config, workers: int, resume: bool, label: str) -> dict:
    """items: dicts with key, cfg, condition, target, init_scale, override, dir.
    Wave k runs at scale s_k; s_{k+1} = s_k * target / p_k. Converged runs double as final theta runs."""
    seeds = cfg17.seeds
    st = {it["key"]: dict(scale=float(it["init_scale"]), history=[], converged=False) for it in items}
    for wave in range(1, cfg17.max_calibration_waves + 1):
        tasks = []
        for it in items:
            s = st[it["key"]]
            if s["converged"]:
                continue
            for p in _seed_paths(it["dir"] / f"cal{wave}", seeds):
                tasks.append((it["cfg"], it["condition"], int(p.stem.split("_")[1]), cfg17.theta_noise_pa,
                              s["scale"], it["override"], str(p), resume))
        if not tasks:
            break
        contrib = _run_parallel(tasks, workers, f"{label}-cal{wave}")
        for it in items:
            s = st[it["key"]]
            if s["converged"]:
                continue
            d = it["dir"] / f"cal{wave}"
            p = pooled_gain(_seed_paths(d, seeds), contrib)
            rel = abs(p - it["target"]) / it["target"]
            s["history"].append(dict(wave=wave, scale=s["scale"], p_match=p, relative_error=rel))
            if rel < cfg17.calibration_tolerance:
                s.update(converged=True, final_scale=s["scale"], theta_dir=str(d))
            else:
                s["scale"] = s["scale"] * it["target"] / p
    for it in items:
        s = st[it["key"]]
        if not s["converged"]:
            s.update(final_scale=s["scale"], theta_dir=str(it["dir"] / "theta"))
    return st


def final_runs(items, states, cfg17, workers, resume, label):
    """Final theta (if not converged) and attractor runs at the final scale."""
    tasks = []
    for it in items:
        s = states[it["key"]]
        if not s["converged"]:
            for p in _seed_paths(Path(s["theta_dir"]), cfg17.seeds):
                tasks.append((it["cfg"], it["condition"], int(p.stem.split("_")[1]), cfg17.theta_noise_pa,
                              s["final_scale"], it["override"], str(p), resume))
        s["attractor_dir"] = str(it["dir"] / "attractor")
        for p in _seed_paths(Path(s["attractor_dir"]), cfg17.seeds):
            tasks.append((it["cfg"], it["condition"], int(p.stem.split("_")[1]), cfg17.attractor_noise_pa,
                          s["final_scale"], it["override"], str(p), resume))
    _run_parallel(tasks, workers, f"{label}-final")
    for it in items:
        s = states[it["key"]]
        s["final_p_match"] = pooled_gain(_seed_paths(Path(s["theta_dir"]), cfg17.seeds))
        s["final_relative_error"] = abs(s["final_p_match"] - it["target"]) / it["target"]
        s["target"] = it["target"]
        s["mean_release_u"] = pooled_release_u(_seed_paths(Path(s["theta_dir"]), cfg17.seeds))
    return states


# ----------------------------------------------------------------------------- manifests
def manifest_path(out: Path, cid: str) -> Path:
    return Path(out) / "combos" / cid / "manifest.json"


def read_manifest(out, cid):
    p = manifest_path(out, cid)
    return read_json(p) if p.exists() else {}


def write_manifest(out, cid, data):
    p = manifest_path(out, cid); p.parent.mkdir(parents=True, exist_ok=True)
    write_json(data, p)


def _functional_quick(cfg, theta_dir, seeds) -> dict:
    """Functional screen from 0-pA runs (theta band + localization), same functions as the analysis."""
    from analysis.pav import summarize_attractor
    from analysis.theta import analyze_theta
    th, loc = [], []
    for p in _seed_paths(Path(theta_dir), seeds):
        r = load_result(p)
        try:
            t, _ = analyze_theta(r, cfg.simulation, cfg.analysis); th.append(float(t.dominant_in_theta_band))
        except Exception:
            th.append(np.nan)
        try:
            a, _ = summarize_attractor(r, cfg.simulation, cfg.analysis); loc.append(float(a.localization_failure))
        except Exception:
            loc.append(np.nan)
    return dict(theta_fraction=float(np.nanmean(th)) if np.isfinite(th).any() else float("nan"),
                localization_failure_fraction_0pa=float(np.nanmean(loc)) if np.isfinite(loc).any() else float("nan"))


def is_functional(f: dict, cfg17: Phase17Config) -> bool:
    return bool(f.get("theta_fraction", 0) >= cfg17.functional["min_theta_fraction"]
                and f.get("localization_failure_fraction_0pa", 1) <= cfg17.functional["max_localization_failure_0pa"])


# ----------------------------------------------------------------------------- stages
def select_combos(cfg17, only: list[str] | None):
    combos = [(combo_id(*c), *c) for c in cfg17.combos]
    if only:
        combos = [c for c in combos if c[0] in set(only)]
        if not combos:
            raise ValueError(f"none of {only} is in the grid")
    return combos


def run_gate(cfg17, out, workers=1, resume=True, phase10_root=None):
    base = load_config(cfg17.base_config)
    if (base.stf.u_baseline, base.stf.tau_facilitation_ms, base.stf.tau_recovery_ms) != ANCHOR:
        raise ValueError("base config is not the frozen reference point")
    g = Path(out) / "gate"
    tasks = []
    for regime, noise in (("theta", cfg17.theta_noise_pa), ("attractor", cfg17.attractor_noise_pa)):
        for p in _seed_paths(g / "full" / regime, cfg17.seeds):
            tasks.append((base, "dynamic", int(p.stem.split("_")[1]), noise, 1.0, None, str(p), resume))
    _run_parallel(tasks, workers, "gate-full")
    scales = {"dynamic": 1.0}
    if phase10_root:
        man = Path(phase10_root) / "combos" / "tf_1000_tr_20" / "manifest.json"
        if man.exists():
            scales["u_output_only"] = float(read_json(man)["modes"]["u_output_only"]["final_scale"])
    run_restoration_items(cfg17, out, [("ANCHOR", "gate", c, base, s) for c, s in scales.items()],
                          workers, root=g / "restoration")
    write_json(dict(stage="gate", scales=scales, seeds=list(cfg17.seeds)), g / "gate_manifest.json")


def run_full(cfg17, out, workers=1, resume=True, only=None):
    base = load_config(cfg17.base_config)
    target = target_p_match(cfg17)
    combos = select_combos(cfg17, only)
    # wave 0 = raw theta (scale 1), shared by both variants
    tasks = []
    for cid, u, tf, tr in combos:
        cfg = config_for(base, u, tf, tr)
        for p in _seed_paths(Path(out) / "combos" / cid / "raw" / "full" / "theta", cfg17.seeds):
            tasks.append((cfg, "dynamic", int(p.stem.split("_")[1]), cfg17.theta_noise_pa, 1.0, None, str(p), resume))
    contrib = _run_parallel(tasks, workers, "full-raw-theta")
    raw_p = {}
    for cid, u, tf, tr in combos:
        cfg = config_for(base, u, tf, tr)
        d = Path(out) / "combos" / cid / "raw" / "full" / "theta"
        raw_p[cid] = pooled_gain(_seed_paths(d, cfg17.seeds), contrib)
        man = read_manifest(out, cid)
        man.update(combo_id=cid, u_baseline=u, tau_facilitation_ms=tf, tau_recovery_ms=tr)
        man["raw"] = dict(full=dict(final_scale=1.0, theta_dir=str(d), final_p_match=raw_p[cid],
                                    mean_release_u=pooled_release_u(_seed_paths(d, cfg17.seeds), contrib),
                                    functional=_functional_quick(cfg, d, cfg17.seeds)))
        write_manifest(out, cid, man)
    if "matched" in cfg17.variants:
        items = [dict(key=cid, cfg=config_for(base, u, tf, tr), condition="dynamic", target=target,
                      init_scale=target / raw_p[cid], override=None,
                      dir=Path(out) / "combos" / cid / "matched" / "full")
                 for cid, u, tf, tr in combos]
        st = calibrate(items, cfg17, workers, resume, "full-matched")
        st = final_runs(items, st, cfg17, workers, resume, "full-matched")
        for it in items:
            s = st[it["key"]]
            s["valid"] = bool(s["final_relative_error"] <= cfg17.calibration_validity)
            s["functional"] = _functional_quick(it["cfg"], s["theta_dir"], cfg17.seeds)
            man = read_manifest(out, it["key"]); man.setdefault("matched", {})["full"] = s
            write_manifest(out, it["key"], man)


def run_routes(cfg17, out, workers=1, resume=True, only=None, variants=None, force_raw=False):
    base = load_config(cfg17.base_config)
    variants = tuple(variants or cfg17.variants)
    combos = select_combos(cfg17, only)
    items, raw_att = [], []
    for cid, u, tf, tr in combos:
        man = read_manifest(out, cid)
        cfg = config_for(base, u, tf, tr)
        for v in variants:
            if v not in man or "full" not in man[v]:
                raise RuntimeError(f"{cid}: run stage 'full' first")
            full = man[v]["full"]
            if v == "raw":
                if not (force_raw or is_functional(full["functional"], cfg17)):
                    print(f"[routes] {cid} raw not functional -> skipped (pre-registered)"); continue
                d = Path(out) / "combos" / cid / "raw" / "full" / "attractor"
                raw_att += [(cfg, "dynamic", int(p.stem.split("_")[1]), cfg17.attractor_noise_pa, 1.0, None, str(p), resume)
                            for p in _seed_paths(d, cfg17.seeds)]
            for mode in cfg17.route_modes:
                override = float(full["mean_release_u"]) if mode == "mean_u_release" else None
                items.append(dict(key=(cid, v, mode), cfg=cfg, condition=ROUTE_CONDITION[mode],
                                  target=float(full["final_p_match"]), init_scale=float(full["final_scale"]),
                                  override=override, dir=Path(out) / "combos" / cid / v / "routes" / mode))
    _run_parallel(raw_att, workers, "routes-raw-full-attractor")
    if not items:
        return
    st = calibrate(items, cfg17, workers, resume, "routes")
    st = final_runs(items, st, cfg17, workers, resume, "routes")
    for (cid, v, mode), s in st.items():
        s["valid"] = bool(s["final_relative_error"] <= cfg17.calibration_validity)
        man = read_manifest(out, cid)
        if v == "raw":
            man["raw"]["full"]["attractor_dir"] = str(Path(out) / "combos" / cid / "raw" / "full" / "attractor")
        man[v].setdefault("routes", {})[mode] = {k: (str(x) if isinstance(x, Path) else x) for k, x in s.items()}
        write_manifest(out, cid, man)


# ----------------------------------------------------------------------------- restoration (module-D protocol)
def _t0(cfg17, seed):
    from experiment.phase11_pilot import checkpoint_ms_for_seed
    o = cfg17.restoration.get("checkpoint_ms_override")
    return float(o) if o is not None else float(checkpoint_ms_for_seed(int(seed)))


def _kw(scale):
    return dict(temporal_gain_scale=float(scale), resource_release_u_override=None, static_gain=None)


def _ckpt_task(payload):
    from model.simulator import run_segment
    from model.state import save_checkpoint
    cfg, cond, seed, scale, t0, path = payload
    path = Path(path)
    if path.exists():
        return str(path)
    _r, state = run_segment(cfg, condition=cond, seed=int(seed), noise_pa=0.0, end_ms=float(t0), **_kw(scale))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(state, path)
    return str(path)


def _branch_task(payload):
    from model.state import aligned_roll_bins, apply_named_shift, load_checkpoint
    from phase16.snapshots import run_with_snapshots
    cfg, cond, seed, scale, t0, ckpt, kind, delta, T, out, meta = payload
    out = Path(out)
    if out.exists():
        return str(out)
    state = load_checkpoint(ckpt)
    t_now = state.step * cfg.simulation.dt_ms
    if abs(t_now - t0) > 1e-6:
        raise RuntimeError(f"checkpoint time {t_now} != protocol time {t0}")
    if kind == "no_shift":
        br = state.copy()
    else:
        e, i = aligned_roll_bins(float(delta), cfg.network.ne, cfg.network.ni)
        br = apply_named_shift(state, kind, e, i)
    res, _s, _snaps = run_with_snapshots(cfg, condition=cond, seed=int(seed), noise_pa=0.0, start_state=br,
                                         end_ms=float(t0) + float(T), every_ms=None, **_kw(scale))
    m = res.spike_times_e_ms > t0
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, spike_t=(res.spike_times_e_ms[m] - t0).astype(np.float32),
                        spike_a=res.spike_angles_e_deg[m].astype(np.float32), meta=json.dumps(meta))
    return str(out)


def _pool(fn, items, workers, label):
    if workers <= 1:
        for k, it in enumerate(items, 1):
            fn(it); print(f"[{label}] {k}/{len(items)}", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, it) for it in items]
        for k, f in enumerate(as_completed(futs), 1):
            f.result()
            if k % 10 == 0 or k == len(items):
                print(f"[{label}] {k}/{len(items)}", flush=True)


def run_restoration_items(cfg17, out, items, workers, root=None):
    """items: (cid, variant, condition, cfg, scale)."""
    R = cfg17.restoration
    root = Path(root) if root else Path(out) / "restoration"
    kinds_dyn = ("activity_shift",) + (("activity_u_shift", "activity_r_shift") if R.get("include_trace_shifts") else ())
    ck, br = [], []
    for cid, v, cond, cfg, scale in items:
        for seed in R["seeds"]:
            t0 = _t0(cfg17, seed)
            ckpt = root / cid / v / "checkpoints" / cond / f"seed_{int(seed):04d}.pkl"
            ck.append((cfg, cond, seed, scale, t0, str(ckpt)))
            specs = [("no_shift", 0.0)]
            for kind in (kinds_dyn if cond == "dynamic" else ("activity_shift",)):
                for d in R["deltas_deg"]:
                    specs += [(kind, float(d)), (kind, -float(d))]
            for kind, d in specs:
                tag = "no_shift" if kind == "no_shift" else f"{kind}__{'p' if d >= 0 else 'm'}{abs(d):g}".replace(".", "p")
                meta = dict(combo_id=cid, variant=v, condition=cond, seed=int(seed), kind=kind, delta=d,
                            scale=float(scale), t0_ms=t0, T_ms=float(R["continuation_ms"]))
                br.append((cfg, cond, seed, scale, t0, str(ckpt), kind, d, float(R["continuation_ms"]),
                           str(root / cid / v / "branches" / f"{cond}__seed{int(seed):04d}__{tag}.npz"), meta))
    _pool(_ckpt_task, ck, workers, "restoration-checkpoints")
    _pool(_branch_task, br, workers, "restoration-branches")


def run_restoration(cfg17, out, workers=1, only=None, variants=None, conditions=None):
    base = load_config(cfg17.base_config)
    variants = tuple(variants or cfg17.variants)
    conditions = tuple(conditions or cfg17.restoration["conditions"])
    items = []
    for cid, u, tf, tr in select_combos(cfg17, only):
        man = read_manifest(out, cid); cfg = config_for(base, u, tf, tr)
        for v in variants:
            if v not in man:
                raise RuntimeError(f"{cid}: run stage 'full' first")
            if v == "raw" and not is_functional(man["raw"]["full"]["functional"], cfg17):
                print(f"[restoration] {cid} raw not functional -> skipped (pre-registered)"); continue
            for cond in conditions:
                if cond == "dynamic":
                    scale = man[v]["full"]["final_scale"]
                else:
                    mode = {"u_output_only": "u_output_only", "u_release_only": "u_release_only"}[cond]
                    if "routes" not in man[v] or mode not in man[v]["routes"]:
                        print(f"[restoration] {cid}/{v}: {cond} needs stage 'routes' -> skipped"); continue
                    scale = man[v]["routes"][mode]["final_scale"]
                items.append((cid, v, cond, cfg, scale))
    run_restoration_items(cfg17, out, items, workers)


# ----------------------------------------------------------------------------- cost estimate
def estimate_runs(cfg17: Phase17Config, n_combos: int) -> dict:
    """Worst-case run counts (all calibration waves used) and CPU hours at ~3 min per 8-s simulation.
    Restoration and route counts assume the matched variant only (raw runs only where functional)."""
    s = len(cfg17.seeds); w = cfg17.max_calibration_waves; m = len(cfg17.route_modes)
    R = cfg17.restoration; ns = len(R["seeds"]); nd = len(R["deltas_deg"])
    k_dyn = 3 if R.get("include_trace_shifts") else 1
    gate_8s = 2 * s
    full_8s = n_combos * (s + w * s + s + s)                     # raw theta + waves + final theta + attractor
    routes_8s = n_combos * m * (w * s + 2 * s)
    n_cond = len(R["conditions"])
    ckpts = n_combos * n_cond * ns
    branches = n_combos * ns * ((1 + 2 * nd * k_dyn) + (n_cond - 1) * (1 + 2 * nd))
    gate_ckpts = n_cond * ns; gate_branches = ns * ((1 + 2 * nd * k_dyn) + (n_cond - 1) * (1 + 2 * nd))
    f_ck, f_br = 3.15 / 8.0, float(R["continuation_ms"]) / 8000.0  # cost relative to one 8-s run
    eq = gate_8s + full_8s + routes_8s + (ckpts + gate_ckpts) * f_ck + (branches + gate_branches) * f_br
    return dict(n_combos=n_combos, gate_8s_runs=gate_8s, gate_restoration_runs=gate_ckpts + gate_branches,
                full_8s_runs_max=full_8s, routes_8s_runs_max=routes_8s,
                restoration_checkpoints=ckpts, restoration_branches=branches,
                cpu_hours_max=round(eq * 3.0 / 60.0, 1))
