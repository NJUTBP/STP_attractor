from pathlib import Path
import numpy as np

from config import load_config
from io_utils import load_result, save_result
from model import simulate


def test_result_roundtrip(tmp_path: Path):
    cfg = load_config("config/smoke.toml")
    result = simulate(cfg, condition="baseline", seed=1, noise_pa=0.0)
    path = save_result(result, tmp_path / "run.npz")
    loaded = load_result(path)
    assert loaded.condition == result.condition
    assert loaded.seed == result.seed
    assert np.array_equal(loaded.plfp_gaba_e, result.plfp_gaba_e)
