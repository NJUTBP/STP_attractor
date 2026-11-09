from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np


FAST_E_FIELDS = ("ve", "ref_e", "x_nmda", "s_nmda", "noise_e")
FAST_I_FIELDS = ("vi", "ref_i", "s_gaba", "noise_i")
STP_FIELDS = ("u", "resources")


def _copy_rng_state(state: dict) -> dict:
    copied: dict = {}
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        elif isinstance(value, dict):
            copied[key] = _copy_rng_state(value)
        else:
            copied[key] = value
    return copied


@dataclass
class RingNetworkState:
    """Complete spatially indexed simulator state at the *next* step to execute."""

    step: int
    ve: np.ndarray
    vi: np.ndarray
    ref_e: np.ndarray
    ref_i: np.ndarray
    x_nmda: np.ndarray
    s_nmda: np.ndarray
    s_gaba: np.ndarray
    noise_e: np.ndarray
    noise_i: np.ndarray
    u: np.ndarray | None
    resources: np.ndarray | None
    rng_noise_e_state: dict
    rng_noise_i_state: dict
    spike_times_e_ms: list[float]
    spike_angles_e_deg: list[float]
    spike_times_i_ms: list[float]
    spike_angles_i_deg: list[float]
    sample_times_ms: np.ndarray
    plfp_gaba_e: np.ndarray
    sample_idx: int
    efficacy_sum: np.ndarray
    efficacy_count: int
    match_num: float
    match_den: float
    release_u_sum: float
    release_u_count: int

    def copy(self) -> "RingNetworkState":
        return RingNetworkState(
            step=int(self.step),
            ve=self.ve.copy(),
            vi=self.vi.copy(),
            ref_e=self.ref_e.copy(),
            ref_i=self.ref_i.copy(),
            x_nmda=self.x_nmda.copy(),
            s_nmda=self.s_nmda.copy(),
            s_gaba=self.s_gaba.copy(),
            noise_e=self.noise_e.copy(),
            noise_i=self.noise_i.copy(),
            u=None if self.u is None else self.u.copy(),
            resources=None if self.resources is None else self.resources.copy(),
            rng_noise_e_state=_copy_rng_state(self.rng_noise_e_state),
            rng_noise_i_state=_copy_rng_state(self.rng_noise_i_state),
            spike_times_e_ms=list(self.spike_times_e_ms),
            spike_angles_e_deg=list(self.spike_angles_e_deg),
            spike_times_i_ms=list(self.spike_times_i_ms),
            spike_angles_i_deg=list(self.spike_angles_i_deg),
            sample_times_ms=self.sample_times_ms.copy(),
            plfp_gaba_e=self.plfp_gaba_e.copy(),
            sample_idx=int(self.sample_idx),
            efficacy_sum=self.efficacy_sum.copy(),
            efficacy_count=int(self.efficacy_count),
            match_num=float(self.match_num),
            match_den=float(self.match_den),
            release_u_sum=float(self.release_u_sum),
            release_u_count=int(self.release_u_count),
        )

    @property
    def has_stp(self) -> bool:
        return self.u is not None and self.resources is not None


def aligned_roll_bins(delta_deg: float, ne: int, ni: int) -> tuple[int, int]:
    e_bins = delta_deg * ne / 360.0
    i_bins = delta_deg * ni / 360.0
    if abs(e_bins - round(e_bins)) > 1e-9 or abs(i_bins - round(i_bins)) > 1e-9:
        raise ValueError(
            f"delta_deg={delta_deg} is not an integer number of E and I bins "
            f"(E bins={e_bins}, I bins={i_bins})"
        )
    return int(round(e_bins)), int(round(i_bins))


def roll_fields(state: RingNetworkState, names: tuple[str, ...], e_bins: int, i_bins: int) -> None:
    e_set = set(FAST_E_FIELDS + STP_FIELDS)
    i_set = set(FAST_I_FIELDS)
    for name in names:
        arr = getattr(state, name)
        if arr is None:
            if name in STP_FIELDS:
                continue
            raise ValueError(f"cannot roll missing field {name}")
        if name in e_set:
            setattr(state, name, np.roll(arr, e_bins))
        elif name in i_set:
            setattr(state, name, np.roll(arr, i_bins))
        else:
            raise ValueError(f"{name} is not a spatially indexed ring field")


def apply_activity_shift(state: RingNetworkState, e_bins: int, i_bins: int) -> RingNetworkState:
    out = state.copy()
    roll_fields(out, FAST_E_FIELDS + FAST_I_FIELDS, e_bins, i_bins)
    return out


def apply_trace_shift(state: RingNetworkState, e_bins: int, i_bins: int) -> RingNetworkState:
    out = state.copy()
    roll_fields(out, STP_FIELDS, e_bins, i_bins)
    return out




def apply_activity_u_shift(state: RingNetworkState, e_bins: int, i_bins: int) -> RingNetworkState:
    """Shift fast neural activity and facilitation u, leaving resource R behind."""
    out = state.copy()
    roll_fields(out, FAST_E_FIELDS + FAST_I_FIELDS + ("u",), e_bins, i_bins)
    return out


def apply_activity_resource_shift(state: RingNetworkState, e_bins: int, i_bins: int) -> RingNetworkState:
    """Shift fast neural activity and resource R, leaving facilitation u behind."""
    out = state.copy()
    roll_fields(out, FAST_E_FIELDS + FAST_I_FIELDS + ("resources",), e_bins, i_bins)
    return out

def apply_co_shift(state: RingNetworkState, e_bins: int, i_bins: int) -> RingNetworkState:
    out = state.copy()
    roll_fields(out, FAST_E_FIELDS + FAST_I_FIELDS + STP_FIELDS, e_bins, i_bins)
    return out


def apply_named_shift(state: RingNetworkState, kind: str, e_bins: int, i_bins: int) -> RingNetworkState:
    if kind in {"none", "no_shift"}:
        return state.copy()
    if kind == "activity_shift":
        return apply_activity_shift(state, e_bins, i_bins)
    if kind == "trace_shift":
        return apply_trace_shift(state, e_bins, i_bins)
    if kind == "activity_u_shift":
        return apply_activity_u_shift(state, e_bins, i_bins)
    if kind in {"activity_r_shift", "activity_resource_shift"}:
        return apply_activity_resource_shift(state, e_bins, i_bins)
    if kind == "co_shift":
        return apply_co_shift(state, e_bins, i_bins)
    raise ValueError(f"Unknown shift kind: {kind}")


def save_checkpoint(state: RingNetworkState, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(state.copy(), f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_checkpoint(path: str | Path) -> RingNetworkState:
    with Path(path).open("rb") as f:
        state = pickle.load(f)
    if not isinstance(state, RingNetworkState):
        raise TypeError(f"checkpoint {path} is not a RingNetworkState")
    return state
