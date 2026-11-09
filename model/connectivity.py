from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from config.schema import NetworkConfig


@dataclass(frozen=True)
class Connectivity:
    theta_e_deg: np.ndarray
    theta_i_deg: np.ndarray
    wei_kernel: np.ndarray
    fwei: np.ndarray
    wie_kernels: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    fwie: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    fwie_matrix: np.ndarray
    ei_presynaptic_weight_sum: np.ndarray


def circular_distance_deg(a: np.ndarray, b: float) -> np.ndarray:
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def build_connectivity(cfg: NetworkConfig) -> Connectivity:
    cfg.validate()
    ne, ni = cfg.ne, cfg.ni
    theta_e = np.arange(ne, dtype=float) * 360.0 / ne
    theta_i = np.arange(ni, dtype=float) * 360.0 / ni

    integrate = np.linspace(0.0, 180.0, 10_000)
    exp_ei = np.exp(-(integrate**2) / (2.0 * cfg.sigma_ei_deg**2))
    sum_exp_ei = exp_ei.sum()
    j_neg = (10_000.0 - sum_exp_ei * cfg.j_ei_pos) / (10_000.0 - sum_exp_ei)

    delta_ei = circular_distance_deg(theta_e, theta_i[0])
    wei = j_neg + (cfg.j_ei_pos - j_neg) * np.exp(
        -(delta_ei**2) / (2.0 * cfg.sigma_ei_deg**2)
    )
    fwei = np.fft.fft(wei)

    # Reference code exploits NE == 4*NI. Each kernel describes one E-index residue class.
    exp_ie = np.exp(-((integrate - cfg.ie_offset_deg) ** 2) / (2.0 * cfg.sigma_ie_deg**2))
    g_ie = 10_000.0 / exp_ie.sum()
    wie = []
    fwie = []
    for residue in range(4):
        # Match the public reference implementation exactly.  For residue r the
        # kernel is defined by the distance from I cell 0 to E[r::4], not by
        # the distance from every I cell to E[r].  Those expressions are only
        # equivalent for residue 0; reversing them for residues 1..3 introduces
        # a small but systematic spatial chirality that can drive a traveling bump.
        delta = circular_distance_deg(theta_e[residue::4], theta_i[0])
        kernel = g_ie * np.exp(-((delta - cfg.ie_offset_deg) ** 2) / (2.0 * cfg.sigma_ie_deg**2))
        wie.append(kernel)
        fwie.append(np.fft.fft(kernel))

    # For the gain-matched control, the phase-4 specification weights each presynaptic E neuron
    # by the total E->I structural weight it projects onto all I cells.
    post_e_indices = 4 * np.arange(ni)
    pre_indices = np.arange(ne)
    offsets = (post_e_indices[:, None] - pre_indices[None, :]) % ne
    pre_weight_sum = wei[offsets].sum(axis=0)

    return Connectivity(
        theta_e_deg=theta_e,
        theta_i_deg=theta_i,
        wei_kernel=wei,
        fwei=fwei,
        wie_kernels=tuple(wie),
        fwie=tuple(fwie),
        fwie_matrix=np.stack(fwie, axis=0),
        ei_presynaptic_weight_sum=pre_weight_sum,
    )
