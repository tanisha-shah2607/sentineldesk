"""Contract tests: run these after every change. They need no data or models."""
from src.data import build_text
from src.gate import decide
from src.pii import mask_pii
from src.pipeline import run_ticket


def test_null_subject_handled():
    assert "[no subject]" in build_text(None, "hello")
    assert "[no subject]" in build_text("", "hello")
    assert build_text(float("nan"), None)  # no crash


def test_pipeline_shape():
    r = run_ticket(None, "My VPN is down")
    assert set(r) == {"triage", "hits", "gate", "draft", "latency"}
    assert {"language", "type", "queue", "priority", "conf", "overall_conf"} <= set(r["triage"])
    assert "escalate" in r["gate"] and "reasons" in r["gate"]
    assert "total_ms" in r["latency"]


def test_gate_high_incident_escalates():
    tri = {"priority": "high", "type": "Incident", "overall_conf": 0.99}
    out = decide(tri, [{"score": 0.99}], "x")
    assert out["escalate"] and out["reasons"]


def test_pii_mask():
    m = mask_pii("mail a@b.com from 10.0.0.1")
    assert "a@b.com" not in m and "10.0.0.1" not in m
