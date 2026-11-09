"""End-to-end smoke test of T2 on three 1.4-s simulations written in the Phase-10 layout (code check only)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def main() -> int:
    from config import load_config
    from io_utils import save_result
    from model import build_connectivity, simulate
    from revision_r1.t2_efficacy_stats import analyze_t2, run_t2
    cfg = load_config(CODE_ROOT / "config" / "phase4_causal.toml")
    conn = build_connectivity(cfg.network)
    runs = {("probe_full", "dynamic"): (1.0, None), ("theta", "u_output_only"): (0.8721143, None),
            ("theta", "u_release_only"): (2.1329120, None)}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "phase10_like"
        for (stage, mode), (k, ov) in runs.items():
            res = simulate(cfg, condition=mode, seed=0, noise_pa=0.0, connectivity=conn,
                           temporal_gain_scale=k, resource_release_u_override=ov, end_ms=1400.0)
            save_result(res, root / "combos" / "tf_1000_tr_20" / stage / mode / "seed_0000.npz")
        out = Path(tmp) / "t2_out"
        rc = run_t2.main(["--roots", str(root), "--out", str(out), "--window", "1000", "1400", "--execute"])
        rc |= analyze_t2.main(["--out", str(out)])
        print((out / "per_trial_metrics.csv").read_text()[:2000])
    print("SMOKE_OK" if rc == 0 else "SMOKE_FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
