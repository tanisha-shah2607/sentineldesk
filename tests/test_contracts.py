"""Contract tests: run these after every change. They need no data or models."""
from src.config import LATENCY_KEYS
from src.data import build_text
from src.gate import decide
from src.pii import mask_pii
from src.pipeline import run_ticket
from src.retrieval import retrieve, retrieve_batch
from src.triage import triage, triage_batch


def test_null_subject_handled():
    assert "[no subject]" in build_text(None, "hello")
    assert "[no subject]" in build_text("", "hello")
    assert build_text(float("nan"), None)  # no crash


def test_pipeline_shape():
    r = run_ticket(None, "My VPN is down")
    assert set(r) == {"triage", "hits", "gate", "draft", "latency"}
    assert {"language", "type", "queue", "priority", "conf", "overall_conf"} <= set(r["triage"])
    assert {"escalate", "reasons"} <= set(r["gate"])
    assert {"text", "cited_ids", "warnings", "used_fallback", "for_review"} <= set(r["draft"])
    assert set(LATENCY_KEYS) <= set(r["latency"])


def test_retrieve_shape():
    h = retrieve("vpn down")[0]
    assert {"ticket_id", "score", "subject", "answer", "snippet", "queue", "type", "language"} <= set(h)
    assert len(retrieve_batch(["a", "b"])) == 2


def test_triage_batch_matches_single():
    items = [(None, "vpn down"), ("Billing", "invoice wrong")]
    assert len(triage_batch(items)) == 2
    assert triage_batch(items[:1])[0].keys() == triage(*items[0]).keys()


def test_gate_high_incident_escalates():
    tri = {"priority": "high", "type": "Incident", "overall_conf": 0.99}
    out = decide(tri, [{"score": 0.99}], "x")
    assert out["escalate"] and out["reasons"]


def test_pii_mask():
    m = mask_pii("mail a@b.com from 10.0.0.1")
    assert "a@b.com" not in m and "10.0.0.1" not in m
