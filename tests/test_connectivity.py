import numpy as np

from config.schema import NetworkConfig
from model.connectivity import build_connectivity


def _reference_wie_kernels(cfg: NetworkConfig) -> list[np.ndarray]:
    ne, ni = cfg.ne, cfg.ni
    theta_e = np.linspace(0.0, 360.0 * (ne - 1) / ne, ne)
    theta_i = np.linspace(0.0, 360.0 * (ni - 1) / ni, ni)
    integrate = np.linspace(0.0, 180.0, 10_000)
    sum_exp = np.sum(
        np.exp(-((integrate - cfg.ie_offset_deg) ** 2) / (2.0 * cfg.sigma_ie_deg**2))
        * 180.0
        / 10_000.0
    )
    g_ie = 180.0 / sum_exp

    out = []
    for residue in range(4):
        delta = np.abs(theta_i[0] - theta_e[residue::4])
        delta = np.where(delta <= 180.0, delta, 360.0 - delta)
        out.append(g_ie * np.exp(-((delta - cfg.ie_offset_deg) ** 2) / (2.0 * cfg.sigma_ie_deg**2)))
    return out


def test_connectivity_shapes():
    cfg = NetworkConfig(ne=64, ni=16)
    conn = build_connectivity(cfg)
    assert conn.wei_kernel.shape == (64,)
    assert conn.fwei.shape == (64,)
    assert all(k.shape == (16,) for k in conn.wie_kernels)
    assert conn.ei_presynaptic_weight_sum.shape == (64,)
    assert np.all(np.isfinite(conn.ei_presynaptic_weight_sum))


def test_wie_kernels_match_public_reference_construction():
    cfg = NetworkConfig(ne=1024, ni=256)
    conn = build_connectivity(cfg)
    expected = _reference_wie_kernels(cfg)
    for got, want in zip(conn.wie_kernels, expected):
        np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-13)
