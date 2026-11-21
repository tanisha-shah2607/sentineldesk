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
from .config import TOP_K


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
