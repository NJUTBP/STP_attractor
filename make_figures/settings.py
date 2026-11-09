"""Default relative paths and protocol constants for main-text figures."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

RESULTS_ROOT = PROJECT_ROOT / "results"
RAW_ROOT = PROJECT_ROOT / "results"
CHECKPOINT_ROOT = RESULTS_ROOT / "phase11_anchor_pilot"

ALLOW_RERUN_EXAMPLES = False

DEFAULT_EXAMPLES = BASE_DIR / "examples"
DEFAULT_OUTPUT = BASE_DIR / "output"

EXAMPLE_SEED = 0
ANCHOR_U = 0.4
ANCHOR_TAU_F_MS = 1000.0
ANCHOR_TAU_R_MS = 20.0
PAV_WINDOW_MS = 400.0
DEFAULT_ZOOM_S = (5.0, 6.2)
THETA_ANALYSIS_WINDOW_MS = (1000.0, 8000.0)

# None = use the archived Phase-16 tables in make_suppfigures/bundled_data/phase16
PHASE16_ROOT = None
PHASE16_VERIFY = False
