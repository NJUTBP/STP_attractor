from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from model.result import SimulationResult


def save_result(result: SimulationResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        condition=np.array(result.condition),
        seed=np.array(result.seed),
        noise_pa=np.array(result.noise_pa),
        static_gain=np.array(np.nan if result.static_gain is None else result.static_gain),
        temporal_gain_scale=np.array(result.temporal_gain_scale),
        resource_release_u_override=np.array(
            np.nan if result.resource_release_u_override is None else result.resource_release_u_override
        ),
        theta_e_deg=result.theta_e_deg,
        spike_times_e_ms=result.spike_times_e_ms,
        spike_angles_e_deg=result.spike_angles_e_deg,
        spike_times_i_ms=result.spike_times_i_ms,
        spike_angles_i_deg=result.spike_angles_i_deg,
        sample_times_ms=result.sample_times_ms,
        plfp_gaba_e=result.plfp_gaba_e,
        mean_efficacy_by_e=result.mean_efficacy_by_e,
        gain_match_numerator=np.array(result.gain_match_numerator),
        gain_match_denominator=np.array(result.gain_match_denominator),
        resource_release_u_sum=np.array(result.resource_release_u_sum),
        resource_release_u_count=np.array(result.resource_release_u_count),
        stp_trace_times_ms=result.stp_trace_times_ms,
        stp_u_axis_deg=result.stp_u_axis_deg,
        stp_u_resultant=result.stp_u_resultant,
        stp_u_mass=result.stp_u_mass,
        stp_r_axis_deg=result.stp_r_axis_deg,
        stp_r_resultant=result.stp_r_resultant,
        stp_r_mass=result.stp_r_mass,
    )
    return path


def load_result(path: str | Path) -> SimulationResult:
    with np.load(path, allow_pickle=False) as z:
        static_gain = float(z["static_gain"])
        temporal_gain_scale = float(z["temporal_gain_scale"]) if "temporal_gain_scale" in z.files else 1.0
        release_override = (
            float(z["resource_release_u_override"])
            if "resource_release_u_override" in z.files
            else float("nan")
        )
        return SimulationResult(
            condition=str(z["condition"].item()),
            seed=int(z["seed"]),
            noise_pa=float(z["noise_pa"]),
            static_gain=None if np.isnan(static_gain) else static_gain,
            temporal_gain_scale=temporal_gain_scale,
            resource_release_u_override=None if np.isnan(release_override) else release_override,
            theta_e_deg=z["theta_e_deg"].copy(),
            spike_times_e_ms=z["spike_times_e_ms"].copy(),
            spike_angles_e_deg=z["spike_angles_e_deg"].copy(),
            spike_times_i_ms=z["spike_times_i_ms"].copy(),
            spike_angles_i_deg=z["spike_angles_i_deg"].copy(),
            sample_times_ms=z["sample_times_ms"].copy(),
            plfp_gaba_e=z["plfp_gaba_e"].copy(),
            mean_efficacy_by_e=z["mean_efficacy_by_e"].copy(),
            gain_match_numerator=float(z["gain_match_numerator"]),
            gain_match_denominator=float(z["gain_match_denominator"]),
            resource_release_u_sum=float(z["resource_release_u_sum"]) if "resource_release_u_sum" in z.files else 0.0,
            resource_release_u_count=int(z["resource_release_u_count"]) if "resource_release_u_count" in z.files else 0,
            stp_trace_times_ms=z["stp_trace_times_ms"].copy() if "stp_trace_times_ms" in z.files else np.empty(0),
            stp_u_axis_deg=z["stp_u_axis_deg"].copy() if "stp_u_axis_deg" in z.files else np.empty(0),
            stp_u_resultant=z["stp_u_resultant"].copy() if "stp_u_resultant" in z.files else np.empty(0),
            stp_u_mass=z["stp_u_mass"].copy() if "stp_u_mass" in z.files else np.empty(0),
            stp_r_axis_deg=z["stp_r_axis_deg"].copy() if "stp_r_axis_deg" in z.files else np.empty(0),
            stp_r_resultant=z["stp_r_resultant"].copy() if "stp_r_resultant" in z.files else np.empty(0),
            stp_r_mass=z["stp_r_mass"].copy() if "stp_r_mass" in z.files else np.empty(0),
        )


def write_json(data: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
