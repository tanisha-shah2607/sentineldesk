"""Owner: S3. Index ONLY load_kb(). Replace the STUB body; keep signature/shape.

retrieve(text, k=5) -> [{
  "ticket_id": "TKT-00012", "score": float (cosine, 0..1),
  "subject": str, "answer": str, "queue": str, "type": str, "language": str
}]  sorted by score desc
"""
from .config import TOP_K


def build_index() -> None:
    """S3: embed load_kb() and persist to INDEX_DIR."""
    raise NotImplementedError


def retrieve(text: str, k: int = TOP_K) -> list[dict]:
    # STUB: S3 replaces with FAISS search
    return [{
        "ticket_id": "TKT-00000", "score": 0.0, "subject": "(stub)",
        "answer": "(stub answer)", "queue": "Technical Support",
        "type": "Incident", "language": "en",
    }][:k]
