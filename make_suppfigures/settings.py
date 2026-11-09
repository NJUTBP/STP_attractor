"""Default relative paths and protocol constants for supplementary figures."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

RESULTS_ROOT = PROJECT_ROOT / "results"
RAW_ROOT = PROJECT_ROOT / "results"
CHECKPOINT_ROOT = RESULTS_ROOT / "phase11_anchor_pilot"

OUTPUT_DIR = HERE / "output"
ALLOW_RERUN_EXAMPLES = False

EXAMPLES_ROOT = HERE / "examples"
SOURCE_DATA_DIR = HERE / "source_data"
FORMATS = ("pdf", "svg", "png")

# None = use the archived Phase-16 tables in bundled_data/phase16
PHASE16_ROOT = None
PHASE16_VERIFY = False
