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
def test_cross_lingual_mix():
    de = "Unser Netzwerkdrucker wird seit heute als offline angezeigt und niemand kann drucken."
    mixed, plain = retrieve(de, k=5), retrieve(de, k=5, xl_slots=0)
    assert {h["language"] for h in mixed} == {"en", "de"}          # both languages present
    assert mixed[0]["ticket_id"] == plain[0]["ticket_id"]           # top-1 (gate score) unchanged
    assert mixed[0]["score"] == plain[0]["score"]
    scores = [h["score"] for h in mixed]
    assert scores == sorted(scores, reverse=True) and len({h["ticket_id"] for h in mixed}) == 5
    assert retrieve(de, k=1)[0]["ticket_id"] == plain[0]["ticket_id"]   # k <= slots: no mixing


@needs_index
def test_hit_language_is_detected_not_label():
    """The dataset label is wrong on ~11% of KB tickets; the index stores the detected language."""
    from src.data import load_kb_clean
    from src.triage import detect_language, language_text
    recs = _load("all")[1][:300]
    kb = load_kb_clean(need_answer=True).set_index("ticket_id")
    for r in recs:
        row = kb.loc[r["ticket_id"]]
        assert r["language"] == detect_language(language_text(row["subject"], row["body"]))


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
    from src.retrieval import _norm_body
    kb = load_kb_clean(need_answer=True)
    ids = [r["ticket_id"] for r in _load("all")[1]]
    norm = _norm_body(kb.set_index("ticket_id").loc[ids, "body"])
    assert len(ids) == len(set(ids)) and not norm.duplicated().any()
    assert len(ids) == _norm_body(kb["body"]).nunique()


@needs_index
def test_answers_have_no_literal_newline_escapes():
    recs = _load("all")[1]
    assert not any("\\n" in r["answer"] or "\\n" in r["snippet"] for r in recs)


def test_cache_key_tracks_text_builder():
    from src.retrieval import MODEL_NAME, _cache_key
    assert _cache_key().startswith(MODEL_NAME + "|text:")
