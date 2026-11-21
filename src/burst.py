"""Owner: Niti (bonus). Major-incident (outage burst) detection over a rolling window of recent tickets.

Idea: one outage produces many near-duplicate tickets in a short time. Each new ticket is compared with
the last `window` tickets; if at least `min_cluster` of them (counting the new one) are similar, we flag a
probable major incident so the on-call engineer handles it once instead of N times.

Similarity backends:
  - BurstDetector.auto(): S3's multilingual e5 encoder (src.retrieval.embed_texts) when the retrieval index
    is built, so the app loads one model and English + German reports of one outage are grouped.
  - Fallback: character 3-5-gram hashing vectors (no model, no fitting; same-language matches only).
  - Or pass any embed_fn returning vectors, plus a matching sim_threshold.
Thresholds were checked on src/examples.py: same-outage English reports score 0.35-0.59, unrelated
tickets <= 0.19 (so the default is 0.30).
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

DEFAULT_HASH_THRESHOLD = 0.30
DEFAULT_EMBED_THRESHOLD = 0.80
# For S3's e5 vectors (scores sit in a narrow 0.85-0.95 band). Tuned on the demo outage + 300 random KB val
# tickets streamed through the detector: fires on the 3rd outage report (also mixed into real traffic, and
# for the German report); ~4% false alarms on real tickets, which are highly repetitive (generated dataset).
E5_THRESHOLD = 0.91
E5_WINDOW = 10


class BurstDetector:
    def __init__(self, window: int = 20, sim_threshold: float | None = None, min_cluster: int = 3,
                 embed_fn=None, max_age_s: float | None = None):
        self.window, self.min_cluster, self.max_age_s = window, min_cluster, max_age_s
        self.embed_fn = embed_fn
        if sim_threshold is None:
            sim_threshold = DEFAULT_EMBED_THRESHOLD if embed_fn else DEFAULT_HASH_THRESHOLD
        self.sim_threshold = sim_threshold
        self._recent: deque = deque(maxlen=window)  # (ticket_id, text, vector, ts)
        self._n = 0
        self._vec = None
        self.backend = "custom embed_fn" if embed_fn else "char n-gram hashing"

    @classmethod
    def auto(cls, **kw) -> "BurstDetector":
        """Use S3's shared e5 encoder if the retrieval index is built here, else the hashing fallback."""
        try:
            from .retrieval import embed_texts, index_ready
            if index_ready():
                kw.setdefault("sim_threshold", E5_THRESHOLD)
                kw.setdefault("window", E5_WINDOW)
                det = cls(embed_fn=lambda t: embed_texts([t])[0], **kw)
                det.backend = "e5 (shared with retrieval)"
                return det
        except Exception:  # noqa: BLE001  (torch/sentence-transformers missing etc.)
            pass
        det = cls(**kw)
        det.backend = "char n-gram hashing"
        return det

    def _embed(self, text: str) -> np.ndarray:
        if self.embed_fn is not None:
            v = np.asarray(self.embed_fn(text), dtype=float).ravel()
            n = np.linalg.norm(v)
            return v / n if n else v
        if self._vec is None:
            from sklearn.feature_extraction.text import HashingVectorizer
            self._vec = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2 ** 18,
                                          alternate_sign=False, norm="l2", lowercase=True)
        return self._vec.transform([text or ""])

    @staticmethod
    def _sim(a, b) -> float:
        if hasattr(a, "multiply"):  # sparse
            return float(a.multiply(b).sum())
        return float(np.dot(a, b))

    def reset(self) -> None:
        self._recent.clear()

    def add(self, ticket_text: str, ticket_id: str | None = None, ts: float | None = None) -> dict:
        """Register a ticket. Returns banner info:
        {burst, count, similar_ids, threshold, label, ticket_id}
        count = similar tickets in the window including this one."""
        self._n += 1
        ticket_id = ticket_id or f"LIVE-{self._n:04d}"
        ts = time.time() if ts is None else ts
        if self.max_age_s is not None:
            while self._recent and ts - self._recent[0][3] > self.max_age_s:
                self._recent.popleft()
        v = self._embed(ticket_text)
        similar = [(tid, txt, s) for tid, txt, vec, _ in self._recent
                   if (s := self._sim(v, vec)) >= self.sim_threshold]
        self._recent.append((ticket_id, ticket_text, v, ts))
        count = len(similar) + 1
        burst = count >= self.min_cluster
        label = ""
        if burst:
            first = similar[0][1] if similar else ticket_text
            label = " ".join(first.split())[:80]
        return {"burst": burst, "count": count, "similar_ids": [tid for tid, _, _ in similar],
                "threshold": self.sim_threshold, "label": label, "ticket_id": ticket_id}
