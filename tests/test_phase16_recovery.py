from dataclasses import replace
from pathlib import Path

from config import load_config
from model.simulator import run_segment
from phase16.snapshots import run_with_snapshots


def test_snapshot_continuation_can_extend_past_checkpoint_config_duration():
    root = Path(__file__).resolve().parent
    # This file is intended to be copied into tests/; parents[1] is repo root there.
    cfg = load_config(root.parents[0] / "config" / "smoke.toml")
    _r, st = run_segment(cfg, condition="dynamic", seed=0, noise_pa=0.0, end_ms=100.0)
    assert len(st.sample_times_ms) == int(cfg.simulation.duration_ms * cfg.simulation.sampling_rate_hz / 1000.0)
    end_ms = cfg.simulation.duration_ms + 40.0
    _r, out, snaps = run_with_snapshots(cfg, condition="dynamic", seed=0, noise_pa=0.0,
                                        start_state=st, end_ms=end_ms, every_ms=20.0)
    assert out.step * cfg.simulation.dt_ms == end_ms
    assert snaps["t"][-1] == end_ms
    assert out.sample_idx > len(st.sample_times_ms)
