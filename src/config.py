"""Shared constants. Everyone imports from here; do not hardcode paths/thresholds elsewhere."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
INDEX_DIR = ROOT / "index"
RESULTS_DIR = ROOT / "results"

KB_PATH = DATA_DIR / "kb_train.parquet"
TEST_PATH = DATA_DIR / "test.parquet"

HF_DATASET = "Tobi-Bueck/customer-support-tickets"
SEED = 42
KB_FRACTION = 0.8

TARGETS = ["type", "queue", "priority"]
NO_SUBJECT = "[no subject]"
VALID_PRIORITIES = ["high", "medium", "low"]   # dataset also has critical/very_low on unlabeled rows

# Gate thresholds (S4 tunes with S1's validation plots; S3 supplies the score scale)
CONF_THRESHOLD = 0.50        # overall classifier confidence below this -> escalate
SIM_THRESHOLD = 0.55         # top retrieval score below this -> escalate
TOP_K = 5
