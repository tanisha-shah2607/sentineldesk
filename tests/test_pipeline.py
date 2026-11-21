"""Owner: Niti. Tests: pipeline resilience, PII wiring, two-stage escalation, contracts, burst detection.
Need no data, models or LLM."""
import pytest

import src.pipeline as P
from src import mocks
from src.burst import BurstDetector
from src.contracts import validate_draft, validate_hits, validate_result, validate_triage
from src.examples import EXAMPLES, OUTAGE_BURST
from src.text import build_text


def _patch_stage(monkeypatch, stage, fn):
    real = P._real
    monkeypatch.setattr(P, "_real", lambda s: fn if s == stage else real(s))


def _raise(exc):
    def f(*a, **k):
        raise exc
    return f


@pytest.mark.parametrize("ex", EXAMPLES, ids=lambda e: e["name"])
def test_examples_satisfy_contract(ex):
    assert validate_result(P.run_ticket(ex["subject"], ex["body"])) == []


@pytest.mark.parametrize("subject,body", [(None, None), (float("nan"), ""), ("", "   "), ("x" * 5000, "ü" * 20000)])
def test_edge_inputs_never_crash(subject, body):
    assert validate_result(P.run_ticket(subject, body)) == []


@pytest.mark.parametrize("stage", P.STAGES)
def test_not_implemented_falls_back_to_mock(monkeypatch, stage):
    _patch_stage(monkeypatch, stage, _raise(NotImplementedError))
    r, status = P.run_ticket_verbose(None, "My VPN is down")
    assert status[stage] == "mock (not implemented)"
    assert validate_result(r) == []


def test_other_errors_fall_back_unless_strict(monkeypatch):
    _patch_stage(monkeypatch, "retrieve", _raise(ValueError("index missing")))
    r, status = P.run_ticket_verbose(None, "My VPN is down", strict=False)
    assert status["retrieve"].startswith("mock (error: ValueError")
    assert validate_result(r) == []
    with pytest.raises(ValueError):
        P.run_ticket_verbose(None, "My VPN is down", strict=True)


def test_force_mocks_env(monkeypatch):
    monkeypatch.setenv("SENTINELDESK_FORCE_MOCKS", "all")
    _, status = P.run_ticket_verbose("s", "b")
    assert all(v == "mock (forced)" for v in status.values())


def test_pii_masked_before_generation(monkeypatch):
    seen = {}

    def spy(subject, body, lang, hits):
        seen["subject"], seen["body"] = subject, body
        return mocks.mock_draft(subject, body, lang, hits)

    _patch_stage(monkeypatch, "generate", spy)
    P.run_ticket("Contact bob@corp.com", "Server 10.0.0.7 down, call +1 617 555 0199")
    joined = f"{seen['subject']} {seen['body']}"
    assert "bob@corp.com" not in joined and "10.0.0.7" not in joined and "555 0199" not in joined


def test_draft_warnings_escalate(monkeypatch):
    def bad_draft(subject, body, lang, hits):
        return {"text": "We guarantee a full refund [TKT-99999].", "cited_ids": [],
                "warnings": ["stripped unsupported claim"], "used_fallback": False}

    _patch_stage(monkeypatch, "generate", bad_draft)
    _patch_stage(monkeypatch, "gate", lambda t, h, x: {"escalate": False, "reasons": []})
    r = P.run_ticket("Billing", "please refund")
    assert r["gate"]["escalate"] and r["draft"]["for_review"]
    assert any("grounding" in x for x in r["gate"]["reasons"])


def test_check_cli_passes():
    assert P.check() == 0


def test_validators_catch_bad_shapes():
    assert validate_triage({"language": "fr"})  # missing keys
    t = mocks.mock_triage(None, "x")
    assert validate_triage(t) == []
    assert validate_triage({**t, "priority": "critical"})
    assert validate_hits([{"score": 0.1}])
    assert validate_hits(mocks.mock_retrieve("vpn")) == []
    assert validate_draft({"text": "x", "cited_ids": ["TKT-1"], "warnings": [], "used_fallback": False},
                          hits=[{"ticket_id": "TKT-2"}])


def test_mocks_are_contract_shaped_and_bilingual():
    assert mocks.mock_triage("Rechnung", "Die Rechnung ist nicht korrekt, bitte prüfen")["language"] == "de"
    assert mocks.mock_triage("Outage", "Production is down for everyone")["priority"] == "high"
    assert all(h["ticket_id"].startswith("MOCK-") for h in mocks.mock_retrieve("vpn"))


def test_burst_detects_outage_and_ignores_noise():
    d = BurstDetector()
    for e in EXAMPLES:
        assert not d.add(build_text(e["subject"], e["body"]))["burst"]
    english = [e for e in OUTAGE_BURST if "Anmeldung" not in (e["subject"] or "")]
    results = [d.add(build_text(e["subject"], e["body"])) for e in english]
    assert not results[0]["burst"]
    assert results[2]["burst"] and results[-1]["count"] >= 3
    assert results[-1]["label"]


def test_burst_window_and_age():
    d = BurstDetector(window=2, min_cluster=3)
    for _ in range(5):
        r = d.add("login error 500 for everyone")
    assert r["count"] == 3  # only 2 kept in window + this one
    d = BurstDetector(max_age_s=10)
    d.add("login error 500", ts=0)
    d.add("login error 500", ts=1)
    assert not d.add("login error 500", ts=100)["burst"]


def test_burst_custom_embedder():
    import numpy as np
    d = BurstDetector(embed_fn=lambda t: np.array([1.0, 0.0]) if "a" in t else np.array([0.0, 1.0]))
    assert d.sim_threshold == 0.80
    d.add("a"); d.add("a")
    assert d.add("a")["burst"] and not d.add("b")["burst"]


def test_burst_auto_falls_back_without_index(monkeypatch):
    import src.retrieval as R
    monkeypatch.setattr(R, "index_ready", lambda: False)
    d = BurstDetector.auto()
    assert d.backend == "char n-gram hashing" and d.sim_threshold == 0.30


def test_mock_scores_follow_sim_threshold():
    from src.config import SIM_THRESHOLD
    good = mocks.mock_retrieve("VPN connection drops, reinstall the VPN client and reconnect")[0]
    assert good["score"] >= SIM_THRESHOLD


def test_placeholder_outputs_are_not_live(monkeypatch, tmp_path):
    import src.triage as T
    monkeypatch.setattr(T, "MODEL_PATH", tmp_path / "missing.joblib")
    monkeypatch.setenv("RETRIEVAL_STUB", "1")
    _, status = P.run_ticket_verbose(None, "My VPN is down")
    assert status["triage"].startswith("placeholder")
    assert status["retrieve"].startswith("placeholder")


def test_grounding_escalation_keeps_gate_codes(monkeypatch):
    def bad_draft(subject, body, lang, hits):
        return {"text": "x [TKT-1]", "cited_ids": [], "warnings": ["w"], "used_fallback": False}
    _patch_stage(monkeypatch, "generate", bad_draft)
    _patch_stage(monkeypatch, "gate", lambda t, h, x: {"escalate": False, "reasons": [], "codes": []})
    g = P.run_ticket("s", "b")["gate"]
    assert g["escalate"] and g["codes"] == ["DRAFT_GROUNDING"] and len(g["reasons"]) == 1
