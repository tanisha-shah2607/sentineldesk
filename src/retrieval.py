"""Owner: S3. Replace the STUB bodies; keep signatures/shape.

retrieve(text, k=5) -> [{
  "ticket_id": "TKT-00012", "score": float (cosine, 0..1),
  "subject": str, "answer": str, "snippet": str (answer[:300]),
  "queue": str, "type": str, "language": str
}]  sorted by score desc
retrieve_batch(texts, k) -> [list_of_hits, ...]   (evaluate.py runs this on the full holdout)

RULES:
  - Index load_kb_clean(need_answer=True) only (never anything from the holdout).
  - Support two index builds: part="train" (for tuning SIM_THRESHOLD on part="val" queries, so a val
    ticket never retrieves itself) and part="all" (the final index used by the app).
  - e5-style query/passage prefixes live here; the input text comes from src.text.build_text.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from .config import INDEX_DIR, TOP_K
from .text import build_text

MODEL_NAME = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
QUERY_PREFIX, PASSAGE_PREFIX = "query: ", "passage: "   # required by e5 models
BATCH_SIZE = 64

EMB_PATH = INDEX_DIR / "kb_emb.npy"          # float32 (n, d), L2-normalised, row-aligned with META_PATH
META_PATH = INDEX_DIR / "kb_meta.parquet"    # ticket_id, subject, answer, queue, type, language
MODEL_TXT = INDEX_DIR / "model.txt"          # model that produced EMB_PATH
META_COLS = ["ticket_id", "subject", "answer", "queue", "type", "language"]


def _device() -> str:
    """EMBED_DEVICE overrides; otherwise cuda > mps (Apple GPU) > cpu (e.g. Cloud Run)."""
    if os.getenv("EMBED_DEVICE"):
        return os.environ["EMBED_DEVICE"]
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME, device=_device())


def embed_texts(texts: list[str], is_query: bool = True, show_progress: bool = False) -> np.ndarray:
    """L2-normalised float32 embeddings. Shared with S2 (classifier) and S5 (burst.py).
    Pass texts built with build_text(); the e5 prefix is added here."""
    prefix = QUERY_PREFIX if is_query else PASSAGE_PREFIX
    return _model().encode([prefix + t for t in texts], batch_size=BATCH_SIZE,
                           normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=show_progress).astype(np.float32)


def embed_kb(force: bool = False) -> None:
    """Step 4: embed every clean KB row with an answer ONCE and cache to INDEX_DIR.
    Keyed by ticket_id, so the train/all indexes are just subsets (no re-embedding)."""
    if not force and EMB_PATH.exists() and META_PATH.exists() and MODEL_TXT.exists() \
            and MODEL_TXT.read_text().strip() == MODEL_NAME:
        print(f"Embedding cache already exists ({EMB_PATH}); use --force to rebuild.")
        return
    from .data import load_kb_clean   # data access stays in src/data.py
    kb = load_kb_clean(need_answer=True, part="all")
    texts = [build_text(s, b) for s, b in zip(kb["subject"], kb["body"])]
    print(f"Embedding {len(texts)} KB tickets with {MODEL_NAME} on {_device()} ...")
    emb = embed_texts(texts, is_query=False, show_progress=True)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_PATH, emb)
    kb[META_COLS].reset_index(drop=True).to_parquet(META_PATH, index=False)
    MODEL_TXT.write_text(MODEL_NAME + "\n")
    print(f"Saved {emb.shape} -> {EMB_PATH}\nSaved metadata -> {META_PATH}")


def load_kb_embeddings() -> tuple[np.ndarray, "pd.DataFrame"]:  # noqa: F821
    """(embeddings, metadata) from the cache, row-aligned. For S3's index build and S2's classifier."""
    import pandas as pd
    return np.load(EMB_PATH), pd.read_parquet(META_PATH)


def build_index(part: str = "all") -> None:
    """S3: embed load_kb_clean(need_answer=True, part=part) and persist to INDEX_DIR."""
    raise NotImplementedError


def retrieve(text: str, k: int = TOP_K) -> list[dict]:
    # STUB: S3 replaces with FAISS search
    return [{
        "ticket_id": "TKT-00000", "score": 0.0, "subject": "(stub)",
        "answer": "(stub answer)", "snippet": "(stub answer)", "queue": "Technical Support",
        "type": "Incident", "language": "en",
    }][:k]


def retrieve_batch(texts: list[str], k: int = TOP_K) -> list[list[dict]]:
    return [retrieve(t, k) for t in texts]
