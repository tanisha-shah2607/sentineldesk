"""THE ONLY place data is read. Owner: S1.

Leakage rule: the holdout's answer/queue/type/priority must never reach inference code.
- Inference/model code uses load_kb() and load_test_inputs() only.
- load_test_labels() is for evaluate.py scoring ONLY.
"""
from __future__ import annotations

import pandas as pd

from .config import (DATA_DIR, HF_DATASET, KB_FRACTION, KB_PATH, NO_SUBJECT,
                     SEED, TEST_PATH, VALID_PRIORITIES)


def build_text(subject, body) -> str:
    """Canonical model input. Handles null/empty/NaN subject and body."""
    s = subject if isinstance(subject, str) and subject.strip() else NO_SUBJECT
    b = body if isinstance(body, str) else ""
    return f"{s}\n{b}".strip()


def make_split(csv_path: str | None = None) -> None:
    """Create the locked 80/20 split exactly as the challenge brief specifies."""
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        from datasets import load_dataset
        df = load_dataset(HF_DATASET)["train"].to_pandas()

    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(df))]
    split_idx = int(len(df) * KB_FRACTION)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.iloc[:split_idx].to_parquet(KB_PATH, index=False)
    df.iloc[split_idx:].to_parquet(TEST_PATH, index=False)
    print(f"KB: {split_idx} rows -> {KB_PATH}\nTest: {len(df) - split_idx} rows -> {TEST_PATH}")


def load_kb() -> pd.DataFrame:
    """Historical KB / training split. Answers allowed here."""
    return pd.read_parquet(KB_PATH)


def clean_mask(df: pd.DataFrame, need_answer: bool = False) -> pd.Series:
    """Rows in the task's label scheme (4 types, 3 priorities, 10 queues).
    ~21% of the raw dataset (all German) has no type/answer, 5-level priority and 42 extra
    queues; those rows are excluded from training, RAG and scoring."""
    m = df["type"].notna() & df["priority"].isin(VALID_PRIORITIES)
    if need_answer:
        m &= df["answer"].notna()
    return m


def load_kb_clean(need_answer: bool = False) -> pd.DataFrame:
    """KB rows to TRAIN classifiers on (need_answer=False) or INDEX for RAG (need_answer=True)."""
    kb = load_kb()
    return kb[clean_mask(kb, need_answer)].reset_index(drop=True)


def load_test_inputs() -> pd.DataFrame:
    """Holdout with ONLY ticket_id, subject, body. Safe for inference."""
    return pd.read_parquet(TEST_PATH)[["ticket_id", "subject", "body"]]


def load_test_labels(clean_only: bool = True) -> pd.DataFrame:
    """Holdout ground truth. evaluate.py scoring ONLY. Never feed to the pipeline.
    'answer' is deliberately excluded. clean_only drops rows outside the task's label scheme."""
    t = pd.read_parquet(TEST_PATH)
    if clean_only:
        t = t[clean_mask(t)]
    return t[["ticket_id", "type", "queue", "priority", "language"]]
