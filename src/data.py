"""THE ONLY place data is read. Owner: S1.

Leakage rule: the holdout's answer/queue/type/priority must never reach inference code.
- The holdout `answer` column is NEVER written to disk (nothing needs it).
- Holdout inputs and labels live in separate files; only evaluate.py calls load_test_labels().
- KB split has two parts, defined once here: part="train" and part="val" (shared by S2/S3/S4).
  Tune calibration/thresholds on "val" with models/indexes built from "train" only.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from . import config as C
from .text import build_text  # re-exported: `from src.data import build_text` still works

__all__ = ["build_text", "make_split", "load_kb", "load_kb_clean", "clean_mask", "val_mask",
           "load_test_inputs", "load_test_labels"]


def _fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for r in pd.concat([df.head(100), df.tail(100)]).itertuples():
        h.update(f"{r.ticket_id}|{r.subject}|{r.body}".encode("utf-8", "replace"))
    return h.hexdigest()[:16]


def make_split(csv_path: str | None = None, allow_mismatch: bool = False) -> None:
    """Create the locked 80/20 split exactly as the challenge brief specifies."""
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        from datasets import load_dataset
        df = load_dataset(C.HF_DATASET)["train"].to_pandas()

    n_raw = len(df)
    df = df.sample(frac=1.0, random_state=C.SEED).reset_index(drop=True)
    df["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(df))]
    fp = _fingerprint(df)
    langs = df["language"].value_counts(dropna=False).to_dict()

    print(f"rows: {n_raw}\nfirst ids: {list(df.ticket_id[:3])}\nfingerprint: {fp}\nlanguages: {langs}")
    if set(langs) - {"en", "de"}:
        print("WARNING: languages other than en/de present; filter at load time, do NOT reshuffle.")
    if C.EXPECTED_ROWS is not None and (n_raw != C.EXPECTED_ROWS or fp != C.EXPECTED_FINGERPRINT):
        msg = (f"DATASET MISMATCH: expected rows={C.EXPECTED_ROWS} fp={C.EXPECTED_FINGERPRINT}, "
               f"got rows={n_raw} fp={fp}. Your ticket ids would differ from the team's.")
        if not allow_mismatch:
            raise SystemExit(msg)
        print("WARNING:", msg)

    split_idx = int(len(df) * C.KB_FRACTION)
    kb, test = df.iloc[:split_idx], df.iloc[split_idx:]
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    kb.to_parquet(C.KB_PATH, index=False)
    test[["ticket_id", "subject", "body"]].to_parquet(C.TEST_INPUT_PATH, index=False)
    test[["ticket_id", "type", "queue", "priority", "language"]].to_parquet(C.TEST_LABEL_PATH, index=False)
    print(f"KB: {len(kb)} | Test: {len(test)} (holdout answer NOT saved)")


def val_mask(df: pd.DataFrame) -> pd.Series:
    """Deterministic ~VAL_PCT% validation slice keyed on ticket_id (independent of row order/filters)."""
    ids = df["ticket_id"].str[4:].astype("int64")
    return (ids * 2654435761 % 100) < C.VAL_PCT


def load_kb(part: str = "all") -> pd.DataFrame:
    """Historical KB. part: 'all' | 'train' | 'val'. Answers allowed here."""
    assert part in ("all", "train", "val"), part
    kb = pd.read_parquet(C.KB_PATH)
    if part == "all":
        return kb
    m = val_mask(kb)
    return kb[m] if part == "val" else kb[~m]


def clean_mask(df: pd.DataFrame, need_answer: bool = False) -> pd.Series:
    """Rows in the task's label scheme (4 types, 3 priorities, 10 queues).
    ~21% of the raw dataset (all German) has no type/answer, 5-level priority and 42 extra
    queues; those rows are excluded from training, RAG and scoring."""
    m = df["type"].notna() & df["priority"].isin(C.VALID_PRIORITIES)
    if need_answer:
        m &= df["answer"].notna()
    return m


def load_kb_clean(need_answer: bool = False, part: str = "all") -> pd.DataFrame:
    """KB rows to TRAIN on (need_answer=False) or INDEX for RAG (need_answer=True)."""
    kb = load_kb(part)
    return kb[clean_mask(kb, need_answer)].reset_index(drop=True)


def load_test_inputs() -> pd.DataFrame:
    """Holdout: ticket_id, subject, body ONLY. Safe for inference."""
    return pd.read_parquet(C.TEST_INPUT_PATH)


def load_test_labels(clean_only: bool = True) -> pd.DataFrame:
    """Holdout ground truth (no answer exists on disk). evaluate.py scoring ONLY."""
    t = pd.read_parquet(C.TEST_LABEL_PATH)
    return t[clean_mask(t)] if clean_only else t
