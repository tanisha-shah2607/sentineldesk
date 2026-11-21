"""Owner: Niti (bonus). Major-incident (outage burst) detection over a rolling window of recent tickets.

Idea: one outage produces many near-duplicate tickets in a short time. Each new ticket is compared with
the last `window` tickets; if at least `min_cluster` of them (counting the new one) are similar, we flag a
probable major incident so the on-call engineer handles it once instead of N times.

Default similarity: character 3-5-gram hashing vectors (no fitting, no model download, works on German
compounds). Cross-lingual matching needs a multilingual embedder: pass embed_fn (e.g. S3's encoder,
returning L2-normalised vectors) and a matching sim_threshold.
Thresholds were checked on src/examples.py: same-outage English reports score 0.35-0.59, unrelated
tickets <= 0.19 (so the default is 0.30).
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

DEFAULT_HASH_THRESHOLD = 0.30
DEFAULT_EMBED_THRESHOLD = 0.80


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
