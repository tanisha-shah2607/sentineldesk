"""Retrieval tests (S3). Real-index tests are skipped until `python scripts/build_index.py` has run."""
import os

import numpy as np
import pytest

from src.retrieval import _load, index_ready, retrieve, retrieve_batch

needs_index = pytest.mark.skipif(not index_ready() or os.getenv("RETRIEVAL_STUB") == "1",
                                 reason="run python scripts/build_index.py first")

KEYS = {"ticket_id", "score", "subject", "answer", "snippet", "queue", "type", "language"}
QUERIES = ["My VPN connection keeps dropping every few minutes since this morning.",
           "Die Rechnung für diesen Monat ist doppelt abgebucht worden."]


def test_stub_mode_keeps_contract(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_STUB", "1")
    assert KEYS <= set(retrieve("vpn down")[0])
    assert retrieve("") == []


@needs_index
def test_contract_keys_sorted_and_snippet():
    hits = retrieve(QUERIES[0], k=5)
    assert len(hits) == 5
    for h in hits:
        assert KEYS <= set(h)
        assert h["snippet"] == h["answer"][:300]
        assert all(isinstance(h[c], str) for c in KEYS - {"score"})
        assert -1.0 <= h["score"] <= 1.0001
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert len({h["ticket_id"] for h in hits}) == 5


@needs_index
def test_k_respected():
    assert len(retrieve(QUERIES[0], k=1)) == 1
    assert len(retrieve(QUERIES[0], k=3)) == 3


@needs_index
def test_empty_text_returns_no_hits():
    assert retrieve("") == [] and retrieve("   ") == []
    out = retrieve_batch(["", QUERIES[1]], k=2)
    assert out[0] == [] and len(out[1]) == 2


@needs_index
def test_batch_matches_single():
    batch = retrieve_batch(QUERIES, k=3)
    for q, b in zip(QUERIES, batch):
        single = retrieve(q, k=3)
        assert single[0]["ticket_id"] == b[0]["ticket_id"]
        assert np.allclose([h["score"] for h in single], [h["score"] for h in b], atol=1e-3)


@needs_index
def test_null_subjects_are_empty_strings():
    recs = _load("all")[1]
    assert all(isinstance(r["subject"], str) and r["subject"] != "nan" for r in recs)
    assert any(r["subject"] == "" for r in recs)   # the KB does have null subjects


@needs_index
def test_index_holds_only_kb_ids():
    from src.data import load_kb_clean, load_test_inputs
    ids = {r["ticket_id"] for r in _load("all")[1]}
    assert ids.isdisjoint(set(load_test_inputs().ticket_id))
    assert ids <= set(load_kb_clean(need_answer=True).ticket_id)


@needs_index
def test_train_index_excludes_val():
    from src.data import load_kb
    ids = {r["ticket_id"] for r in _load("train")[1]}
    assert ids.isdisjoint(set(load_kb(part="val").ticket_id))


@needs_index
def test_index_has_no_duplicate_bodies():
    from src.data import load_kb_clean
    kb = load_kb_clean(need_answer=True)
    ids = [r["ticket_id"] for r in _load("all")[1]]
    body = kb.set_index("ticket_id").loc[ids, "body"]
    norm = body.fillna("").str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    assert len(ids) == len(set(ids)) and not norm.duplicated().any()
    assert len(ids) == kb["body"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip().str.lower().nunique()
