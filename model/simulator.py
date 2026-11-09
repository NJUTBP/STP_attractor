from __future__ import annotations

import numpy as np

from config.schema import ProjectConfig
from .connectivity import Connectivity, build_connectivity, circular_distance_deg
from .result import SimulationResult
from .state import RingNetworkState
from .stp import DynamicSTFState, TEMPORAL_CONDITIONS, efficacy_for_condition, temporal_efficacy_from_components


def axial_profile_summary(weights: np.ndarray, theta_deg: np.ndarray) -> tuple[float, float, float]:
    """Axial centroid, normalized resultant and mean mass of a nonnegative ring profile."""
    w = np.asarray(weights, dtype=float)
    theta = np.asarray(theta_deg, dtype=float)
    if w.shape != theta.shape:
        raise ValueError("weights and theta_deg must have the same shape")
    if np.any(~np.isfinite(w)) or np.any(w < -1e-12):
        raise ValueError("profile weights must be finite and nonnegative")
    w = np.maximum(w, 0.0)
    total = float(np.sum(w))
    mass = float(np.mean(w)) if w.size else 0.0
    if total <= 1e-15:
        return float("nan"), 0.0, mass
    phi = np.deg2rad(2.0 * theta)
    z = np.sum(w * np.exp(1j * phi))
    axis = float(np.mod(0.5 * np.rad2deg(np.angle(z)), 180.0))
    resultant = float(np.abs(z) / total)
    return axis, resultant, mass


def _rng_streams(seed: int) -> tuple[np.random.Generator, np.random.Generator, np.random.Generator]:
    """Independent streams make paired conditions share initial state and exogenous noise exactly."""
    children = np.random.SeedSequence(seed).spawn(3)
    return tuple(np.random.default_rng(child) for child in children)  # type: ignore[return-value]


def paired_post_checkpoint_rng(prehistory_seed: int, noise_draw: int) -> tuple[np.random.Generator, np.random.Generator]:
    """Condition-independent OU streams for one (prehistory_seed, noise_draw) pair."""
    children = np.random.SeedSequence((int(prehistory_seed), int(noise_draw), 0xA11E51DE)).spawn(2)
    return np.random.default_rng(children[0]), np.random.default_rng(children[1])


def _cue_vector(theta_e_deg: np.ndarray, cfg: ProjectConfig) -> np.ndarray:
    sim = cfg.simulation
    cue = np.zeros_like(theta_e_deg)
    for center in sim.cue_centers_deg:
        cue[circular_distance_deg(theta_e_deg, center) < sim.cue_half_width_deg] += sim.cue_amplitude_pa
    return cue


def _ie_sum_gaba(sgaba: np.ndarray, conn: Connectivity, ne: int) -> np.ndarray:
    fsgaba = np.fft.fft(sgaba)
    # Batched inverse FFT keeps the four residue-class convolutions in one NumPy call.
    conv = np.real(np.fft.ifft(conn.fwie_matrix * fsgaba[None, :], axis=1))
    out = np.empty(ne, dtype=float)
    out[0::4] = conv[0]
    out[1::4] = conv[1]
    out[2::4] = conv[2]
    out[3::4] = conv[3]
    return out


def _ei_sum_nmda(effective_snmda: np.ndarray, conn: Connectivity) -> np.ndarray:
    conv = np.real(np.fft.ifft(conn.fwei * np.fft.fft(effective_snmda)))
    return conv[0::4]


def simulate(
    cfg: ProjectConfig,
    *,
    condition: str,
    seed: int,
    noise_pa: float,
    static_gain: float | None = None,
    frozen_profile: np.ndarray | None = None,
    temporal_gain_scale: float = 1.0,
    resource_release_u_override: float | None = None,
    connectivity: Connectivity | None = None,
    start_state: RingNetworkState | None = None,
    end_ms: float | None = None,
    noise_draw: int | None = None,
    record_stp_trace: bool = False,
) -> SimulationResult:
    result, _state = run_segment(
        cfg,
        condition=condition,
        seed=seed,
        noise_pa=noise_pa,
        static_gain=static_gain,
        frozen_profile=frozen_profile,
        temporal_gain_scale=temporal_gain_scale,
        resource_release_u_override=resource_release_u_override,
        connectivity=connectivity,
        start_state=start_state,
        end_ms=end_ms,
        noise_draw=noise_draw,
        record_stp_trace=record_stp_trace,
    )
    return result


def run_segment(
    cfg: ProjectConfig,
    *,
    condition: str,
    seed: int,
    noise_pa: float,
    static_gain: float | None = None,
    frozen_profile: np.ndarray | None = None,
    temporal_gain_scale: float = 1.0,
    resource_release_u_override: float | None = None,
    connectivity: Connectivity | None = None,
    start_state: RingNetworkState | None = None,
    end_ms: float | None = None,
    noise_draw: int | None = None,
    record_stp_trace: bool = False,
) -> tuple[SimulationResult, RingNetworkState]:
    """Run one ring-network trial.

    The only plastic pathway is E->I NMDA.  Phase-10 hybrid modes independently
    manipulate whether dynamic u(t) acts directly on postsynaptic efficacy and
    whether it controls resource depletion:
      dynamic: uR output, dynamic-u release
      u_output_only: uR output, fixed-U release
      u_release_only: UR output, dynamic-u release
      mean_u_release: UR output, fixed externally supplied release probability
    All inhibitory synapses use the same non-plastic dynamics in every condition.
    """
    cfg.validate()
    if condition not in {*TEMPORAL_CONDITIONS, "baseline", "static_gain", "frozen_profile"}:
        raise ValueError(f"Unknown condition: {condition}")
    if condition == "static_gain" and static_gain is None:
        raise ValueError("static_gain condition requires static_gain=p_match")
    if not np.isfinite(temporal_gain_scale) or temporal_gain_scale <= 0:
        raise ValueError("temporal_gain_scale must be finite and > 0")
    if condition not in TEMPORAL_CONDITIONS and abs(temporal_gain_scale - 1.0) > 1e-12:
        raise ValueError("temporal_gain_scale is only defined for temporal STP conditions")
    if condition == "mean_u_release":
        if resource_release_u_override is None or not np.isfinite(resource_release_u_override):
            raise ValueError("mean_u_release requires finite resource_release_u_override")
        if not (0.0 < float(resource_release_u_override) <= 1.0):
            raise ValueError("resource_release_u_override must lie in (0, 1]")
    elif resource_release_u_override is not None:
        raise ValueError("resource_release_u_override is only valid for mean_u_release")

    net, stf, sim, ana = cfg.network, cfg.stf, cfg.simulation, cfg.analysis
    conn = connectivity or build_connectivity(net)
    ne, ni = net.ne, net.ni
    if end_ms is None:
        end_ms = sim.duration_ms
    if end_ms < -1e-12 or end_ms - sim.duration_ms > 1e-9:
        raise ValueError("end_ms must lie in [0, duration_ms]")
    end_step = int(round(end_ms / sim.dt_ms))
    if abs(end_step * sim.dt_ms - end_ms) > 1e-9:
        raise ValueError("end_ms must land on an integer timestep")
    if end_step > int(round(sim.duration_ms / sim.dt_ms)):
        raise ValueError("end_ms exceeds configured duration")
    if condition == "frozen_profile":
        if frozen_profile is None:
            raise ValueError("frozen_profile condition requires frozen_profile")
        frozen_profile = np.asarray(frozen_profile, dtype=float)
        if frozen_profile.shape != (ne,):
            raise ValueError(f"frozen_profile must have shape ({ne},), got {frozen_profile.shape}")
        if not np.all(np.isfinite(frozen_profile)):
            raise ValueError("frozen_profile must contain only finite values")
        if np.any(frozen_profile < 0.0) or np.any(frozen_profile > 1.0):
            raise ValueError("frozen_profile efficacy values must lie in [0, 1]")
    dt = sim.dt_ms
    steps = int(round(sim.duration_ms / dt))
    sample_every = int(round((1000.0 / sim.sampling_rate_hz) / dt))
    n_samples = steps // sample_every

    rng_init, rng_noise_e, rng_noise_i = _rng_streams(seed)
    max_ref_e = int(round(net.tau_ref_e_ms / dt))
    max_ref_i = int(round(net.tau_ref_i_ms / dt))
    temporal_condition = condition in TEMPORAL_CONDITIONS
    cue = _cue_vector(conn.theta_e_deg, cfg)

    if start_state is None:
        start_step = 0
        ve = rng_init.uniform(-60.0, -50.0, ne)
        vi = rng_init.uniform(-60.0, -50.0, ni)
        ref_e = np.full(ne, max_ref_e, dtype=np.int32)
        ref_i = np.full(ni, max_ref_i, dtype=np.int32)
        x_nmda = np.zeros(ne)
        s_nmda = np.zeros(ne)
        s_gaba = np.zeros(ni)
        noise_e = np.zeros(ne)
        noise_i = np.zeros(ni)
        stf_state = DynamicSTFState.initialize(ne, stf) if temporal_condition else None
        spike_t_e = []
        spike_a_e = []
        spike_t_i = []
        spike_a_i = []
        sample_times = np.empty(n_samples, dtype=float)
        plfp = np.empty(n_samples, dtype=float)
        sample_idx = 0
        efficacy_sum = np.zeros(ne)
        efficacy_count = 0
        match_num = 0.0
        match_den = 0.0
        release_u_sum = 0.0
        release_u_count = 0
        if noise_draw is not None:
            rng_noise_e, rng_noise_i = paired_post_checkpoint_rng(seed, noise_draw)
    else:
        if start_state.step < 0 or start_state.step > steps:
            raise ValueError("start_state.step is outside the configured trial")
        if start_state.step > end_step:
            raise ValueError("start_state.step is after end_ms")
        start_step = int(start_state.step)
        ve = start_state.ve.copy()
        vi = start_state.vi.copy()
        ref_e = start_state.ref_e.copy()
        ref_i = start_state.ref_i.copy()
        x_nmda = start_state.x_nmda.copy()
        s_nmda = start_state.s_nmda.copy()
        s_gaba = start_state.s_gaba.copy()
        noise_e = start_state.noise_e.copy()
        noise_i = start_state.noise_i.copy()
        if temporal_condition:
            if start_state.u is None or start_state.resources is None:
                raise ValueError("temporal conditions require STP fields in start_state")
            stf_state = DynamicSTFState(u=start_state.u.copy(), resources=start_state.resources.copy())
        else:
            stf_state = None
        if noise_draw is None:
            rng_noise_e.bit_generator.state = start_state.rng_noise_e_state
            rng_noise_i.bit_generator.state = start_state.rng_noise_i_state
        else:
            rng_noise_e, rng_noise_i = paired_post_checkpoint_rng(seed, noise_draw)
            noise_e = np.zeros_like(noise_e)
            noise_i = np.zeros_like(noise_i)
        spike_t_e = list(start_state.spike_times_e_ms)
        spike_a_e = list(start_state.spike_angles_e_deg)
        spike_t_i = list(start_state.spike_times_i_ms)
        spike_a_i = list(start_state.spike_angles_i_deg)
        sample_times = start_state.sample_times_ms.copy()
        plfp = start_state.plfp_gaba_e.copy()
        sample_idx = int(start_state.sample_idx)
        efficacy_sum = start_state.efficacy_sum.copy()
        efficacy_count = int(start_state.efficacy_count)
        match_num = float(start_state.match_num)
        match_den = float(start_state.match_den)
        release_u_sum = float(start_state.release_u_sum)
        release_u_count = int(start_state.release_u_count)

    trace_times: list[float] = []
    trace_u_axis: list[float] = []
    trace_u_resultant: list[float] = []
    trace_u_mass: list[float] = []
    trace_r_axis: list[float] = []
    trace_r_resultant: list[float] = []
    trace_r_mass: list[float] = []

    dec_x_half = np.exp(-dt / (2.0 * net.tau_x_ms))
    dec_x = np.exp(-dt / net.tau_x_ms)
    dec_g_half = np.exp(-dt / (2.0 * net.tau_gaba_ms))
    dec_g = np.exp(-dt / net.tau_gaba_ms)

    noise_scale = float(noise_pa) * np.sqrt(dt) / np.sqrt(net.tau_noise_ms)
    mg = net.magnesium_mm

    for step in range(start_step, end_step):
        t_ms = step * dt
        stim_e: np.ndarray | float = cue if (sim.cue_start_ms < t_ms < sim.cue_end_ms) else 0.0

        out_ref_e = ref_e == max_ref_e
        out_ref_i = ref_i == max_ref_i

        # ----- K1: pyramidal cells -----
        sum_gie = _ie_sum_gaba(s_gaba, conn, ne)
        i_gaba_e = (ve - net.e_inh_mv) * net.g_gaba_ie_ns * sum_gie
        i_leak_e = net.g_leak_e_ns * (ve - net.e_leak_mv)
        k1_e = (stim_e - i_gaba_e - i_leak_e + net.background_e_pa - noise_e) / net.cm_e_pf
        ve_half = ve + (dt / 2.0) * k1_e

        # ----- K1: interneurons -----
        eff_now = efficacy_for_condition(condition, stf_state, stf, static_gain, frozen_profile)
        if temporal_condition:
            eff_now = eff_now * temporal_gain_scale
        effective_nmda = s_nmda * eff_now
        sum_nei = _ei_sum_nmda(effective_nmda, conn)
        i_nmda_i = (
            (vi - net.e_exc_mv)
            * net.g_nmda_ei_ns
            * sum_nei
            / (1.0 + mg * np.exp(-0.062 * vi) / 3.57)
        )
        i_gaba_i = (vi - net.e_inh_mv) * net.g_gaba_ii_ns * np.sum(s_gaba)
        i_leak_i = net.g_leak_i_ns * (vi - net.e_leak_mv)
        k1_i = (-i_nmda_i - i_gaba_i - i_leak_i + net.background_i_pa - noise_i + net.ms_current_i_pa) / net.cm_i_pf
        vi_half = vi + (dt / 2.0) * k1_i

        # ----- midpoint synaptic/noise/STF states -----
        k1_s = -s_nmda / net.tau_nmda_ms + net.alpha_nmda_khz * x_nmda * (1.0 - s_nmda)
        s_nmda_half = s_nmda + (dt / 2.0) * k1_s
        x_nmda_half = x_nmda * dec_x_half
        s_gaba_half = s_gaba * dec_g_half
        noise_e_half = noise_e - (dt / 2.0) * noise_e / net.tau_noise_ms
        noise_i_half = noise_i - (dt / 2.0) * noise_i / net.tau_noise_ms

        if temporal_condition:
            assert stf_state is not None
            u_half, r_half = stf_state.half_step_recovery(dt, stf)
            eff_half: np.ndarray | float = temporal_efficacy_from_components(condition, u_half, r_half, stf) * temporal_gain_scale
        else:
            eff_half = eff_now

        # ----- K2: pyramidal cells -----
        sum_gie_half = _ie_sum_gaba(s_gaba_half, conn, ne)
        i_gaba_e_half = (ve_half - net.e_inh_mv) * net.g_gaba_ie_ns * sum_gie_half
        i_leak_e_half = net.g_leak_e_ns * (ve_half - net.e_leak_mv)
        k2_e = (stim_e - i_gaba_e_half - i_leak_e_half + net.background_e_pa - noise_e_half) / net.cm_e_pf
        ve = ve + dt * k2_e * out_ref_e

        # ----- K2: interneurons -----
        sum_nei_half = _ei_sum_nmda(s_nmda_half * eff_half, conn)
        i_nmda_i_half = (
            (vi_half - net.e_exc_mv)
            * net.g_nmda_ei_ns
            * sum_nei_half
            / (1.0 + mg * np.exp(-0.062 * vi_half) / 3.57)
        )
        i_gaba_i_half = (vi_half - net.e_inh_mv) * net.g_gaba_ii_ns * np.sum(s_gaba_half)
        i_leak_i_half = net.g_leak_i_ns * (vi_half - net.e_leak_mv)
        k2_i = (
            -i_nmda_i_half
            - i_gaba_i_half
            - i_leak_i_half
            + net.background_i_pa
            - noise_i_half
            + net.ms_current_i_pa
        ) / net.cm_i_pf
        vi = vi + dt * k2_i * out_ref_i

        # ----- advance synapses and exogenous OU noise -----
        k2_s = -s_nmda_half / net.tau_nmda_ms + net.alpha_nmda_khz * x_nmda_half * (1.0 - s_nmda_half)
        x_nmda *= dec_x
        s_nmda += dt * k2_s
        s_gaba *= dec_g

        noise_e = noise_e - dt * noise_e_half / net.tau_noise_ms + rng_noise_e.normal(size=ne) * noise_scale
        noise_i = noise_i - dt * noise_i_half / net.tau_noise_ms + rng_noise_i.normal(size=ni) * noise_scale

        if temporal_condition:
            assert stf_state is not None
            stf_state.recover_full_step(dt, stf)

        # ----- threshold, refractory state, synaptic spike updates -----
        ve = np.where(out_ref_e, ve, net.reset_mv)
        spikes_e = ve >= net.threshold_mv
        ve = np.where(spikes_e, net.reset_mv, ve)
        ref_e = (~spikes_e) * out_ref_e * ref_e + (~out_ref_e) * (ref_e + 1)
        ref_e = ref_e.astype(np.int32, copy=False)
        x_nmda += spikes_e

        if temporal_condition:
            assert stf_state is not None
            step_release_sum, step_release_count = stf_state.on_spikes(
                spikes_e,
                stf,
                condition,
                release_u_override=resource_release_u_override,
            )
            if ana.gain_match_start_ms <= t_ms <= ana.gain_match_end_ms:
                release_u_sum += float(step_release_sum)
                release_u_count += int(step_release_count)

        vi = np.where(out_ref_i, vi, net.reset_mv)
        spikes_i = vi >= net.threshold_mv
        vi = np.where(spikes_i, net.reset_mv, vi)
        ref_i = (~spikes_i) * out_ref_i * ref_i + (~out_ref_i) * (ref_i + 1)
        ref_i = ref_i.astype(np.int32, copy=False)
        s_gaba += spikes_i

        if np.any(spikes_e):
            idx = np.flatnonzero(spikes_e)
            spike_t_e.extend([t_ms] * idx.size)
            spike_a_e.extend(conn.theta_e_deg[idx].tolist())
        if np.any(spikes_i):
            idx = np.flatnonzero(spikes_i)
            spike_t_i.extend([t_ms] * idx.size)
            spike_a_i.extend(conn.theta_i_deg[idx].tolist())

        # Gain matching is defined only by the dynamic condition and pooled later across seeds.
        if temporal_condition and ana.gain_match_start_ms <= t_ms <= ana.gain_match_end_ms:
            assert stf_state is not None
            weighted_gate = conn.ei_presynaptic_weight_sum * s_nmda
            match_num += float(np.dot(weighted_gate, stf_state.efficacy_for_mode(condition, stf) * temporal_gain_scale))
            match_den += float(np.sum(weighted_gate))

        # Record at the same 200-Hz cadence used by the phase-4 analysis.
        if (step + 1) % sample_every == 0 and sample_idx < n_samples:
            sample_times[sample_idx] = (step + 1) * dt
            plfp[sample_idx] = float(np.mean(i_gaba_e_half))
            if record_stp_trace and temporal_condition:
                assert stf_state is not None
                ua, ur, um = axial_profile_summary(
                    np.maximum(stf_state.u - stf.u_baseline, 0.0), conn.theta_e_deg
                )
                ra, rr, rm = axial_profile_summary(
                    np.maximum(1.0 - stf_state.resources, 0.0), conn.theta_e_deg
                )
                trace_times.append(float((step + 1) * dt))
                trace_u_axis.append(ua)
                trace_u_resultant.append(ur)
                trace_u_mass.append(um)
                trace_r_axis.append(ra)
                trace_r_resultant.append(rr)
                trace_r_mass.append(rm)
            if ana.gain_match_start_ms <= t_ms <= ana.gain_match_end_ms:
                if temporal_condition:
                    assert stf_state is not None
                    efficacy_sum += stf_state.efficacy_for_mode(condition, stf) * temporal_gain_scale
                elif condition == "baseline":
                    efficacy_sum += stf.u_baseline
                elif condition == "static_gain":
                    efficacy_sum += float(static_gain)
                else:
                    assert frozen_profile is not None
                    efficacy_sum += frozen_profile
                efficacy_count += 1
            sample_idx += 1

    if efficacy_count:
        mean_efficacy = efficacy_sum / efficacy_count
    else:
        if condition == "frozen_profile":
            assert frozen_profile is not None
            mean_efficacy = frozen_profile.copy()
        else:
            const = stf.u_baseline if condition == "baseline" else (static_gain if condition == "static_gain" else stf.u_baseline)
            mean_efficacy = np.full(ne, float(const))

    result = SimulationResult(
        condition=condition,
        seed=int(seed),
        noise_pa=float(noise_pa),
        static_gain=None if static_gain is None else float(static_gain),
        temporal_gain_scale=float(temporal_gain_scale),
        resource_release_u_override=None if resource_release_u_override is None else float(resource_release_u_override),
        theta_e_deg=conn.theta_e_deg.copy(),
        spike_times_e_ms=np.asarray(spike_t_e, dtype=float),
        spike_angles_e_deg=np.asarray(spike_a_e, dtype=float),
        spike_times_i_ms=np.asarray(spike_t_i, dtype=float),
        spike_angles_i_deg=np.asarray(spike_a_i, dtype=float),
        sample_times_ms=sample_times[:sample_idx].copy(),
        plfp_gaba_e=plfp[:sample_idx].copy(),
        mean_efficacy_by_e=mean_efficacy,
        gain_match_numerator=match_num,
        gain_match_denominator=match_den,
        resource_release_u_sum=float(release_u_sum),
        resource_release_u_count=int(release_u_count),
        stp_trace_times_ms=np.asarray(trace_times, dtype=float),
        stp_u_axis_deg=np.asarray(trace_u_axis, dtype=float),
        stp_u_resultant=np.asarray(trace_u_resultant, dtype=float),
        stp_u_mass=np.asarray(trace_u_mass, dtype=float),
        stp_r_axis_deg=np.asarray(trace_r_axis, dtype=float),
        stp_r_resultant=np.asarray(trace_r_resultant, dtype=float),
        stp_r_mass=np.asarray(trace_r_mass, dtype=float),
    )
    state = RingNetworkState(
        step=end_step,
        ve=ve,
        vi=vi,
        ref_e=ref_e.astype(np.int32, copy=True),
        ref_i=ref_i.astype(np.int32, copy=True),
        x_nmda=x_nmda,
        s_nmda=s_nmda,
        s_gaba=s_gaba,
        noise_e=noise_e,
        noise_i=noise_i,
        u=None if stf_state is None else stf_state.u.copy(),
        resources=None if stf_state is None else stf_state.resources.copy(),
        rng_noise_e_state=rng_noise_e.bit_generator.state,
        rng_noise_i_state=rng_noise_i.bit_generator.state,
        spike_times_e_ms=list(spike_t_e),
        spike_angles_e_deg=list(spike_a_e),
        spike_times_i_ms=list(spike_t_i),
        spike_angles_i_deg=list(spike_a_i),
        sample_times_ms=sample_times.copy(),
        plfp_gaba_e=plfp.copy(),
        sample_idx=int(sample_idx),
        efficacy_sum=efficacy_sum,
        efficacy_count=int(efficacy_count),
        match_num=float(match_num),
        match_den=float(match_den),
        release_u_sum=float(release_u_sum),
        release_u_count=int(release_u_count),
    )
    return result, state
