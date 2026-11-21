"""S4: pipeline escalation after generation. A hand-off (correct refusal) and a grounding failure both
escalate, with different reasons and codes, and the gate's own codes are kept."""
import src.generate as generate
import src.pipeline as pipeline

BASE = {"text": "draft", "cited_ids": [], "used_fallback": False, "llm_error": None}


def _run(monkeypatch, draft):
    # S5's pipeline imports draft_reply from src.generate at call time, so patch it there
    monkeypatch.setattr(generate, "draft_reply", lambda *a, **k: draft)
    return pipeline.run_ticket("Refund", "Guarantee me a full refund of 500 EUR for the outage.")


def test_handoff_escalates_with_its_own_reason(monkeypatch):
    r = _run(monkeypatch, {**BASE, "warnings": [], "abstained": True})
    g = r["gate"]
    assert g["escalate"] and r["draft"]["for_review"]
    assert any("hands off to a specialist" in x for x in g["reasons"])
    assert not any("failed grounding" in x for x in g["reasons"])
    assert g["codes"][-1] == "DRAFT_HANDOFF" and "UNSUPPORTED_ASK" in g["codes"]   # gate codes kept


def test_grounding_failure_escalates_and_keeps_codes(monkeypatch):
    r = _run(monkeypatch, {**BASE, "warnings": ["Draft cites no retrieved precedent"], "abstained": False})
    g = r["gate"]
    assert g["escalate"] and any("failed grounding" in x for x in g["reasons"])
    assert g["codes"][-1] == "DRAFT_GROUNDING" and "UNSUPPORTED_ASK" in g["codes"]


def test_clean_draft_adds_no_draft_reason(monkeypatch):
    r = _run(monkeypatch, {**BASE, "cited_ids": ["TKT-00000"], "warnings": [], "abstained": False})
    assert not any(c.startswith("DRAFT_") for c in r["gate"]["codes"])
    assert not any(x.startswith("Draft ") for x in r["gate"]["reasons"])
