"""Owner: The team. Shared constants. Everyone imports from here; do not hardcode paths/thresholds elsewhere."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
INDEX_DIR = ROOT / "index"
RESULTS_DIR = ROOT / "results"

KB_PATH = DATA_DIR / "kb_train.parquet"            # has `answer`
TEST_INPUT_PATH = DATA_DIR / "test_inputs.parquet"  # ticket_id, subject, body ONLY
TEST_LABEL_PATH = DATA_DIR / "test_labels.parquet"  # labels for scoring; NO answer column, ever

HF_DATASET = "Tobi-Bueck/customer-support-tickets"
SEED = 42
KB_FRACTION = 0.8
VAL_PCT = 10   # % of KB ticket_ids in the shared validation slice (see data.val_mask)

# Dataset pin: make_split.py aborts if the downloaded data differs, so every teammate
# gets identical TKT- ids. (Filled in after the first verified run.)
EXPECTED_ROWS = 61765
EXPECTED_FINGERPRINT = "c161e37ec07f3cb1"

TARGETS = ["type", "queue", "priority"]
NO_SUBJECT = "[no subject]"
VALID_PRIORITIES = ["high", "medium", "low"]   # dataset also has critical/very_low on unlabeled rows

# Gate thresholds (S4 tunes on the shared VAL slice; S3 supplies the score scale)
CONF_THRESHOLD = 0.50        # overall_conf below this -> escalate
SIM_THRESHOLD = 0.85         # top retrieval score below this -> escalate. From S3's scripts/tune_threshold.py:
                             # e5 scores sit in ~0.75-0.95; 0.85 flags none of 19 hand-typed in-scope tickets
                             # and 0% of val, and catches 12/17 off-topic ones
TOP_K = 5

# Latency keys returned by run_ticket()["latency"], all in milliseconds
LATENCY_KEYS = ["triage_ms", "retrieve_ms", "gate_ms", "generate_ms", "total_ms"]
