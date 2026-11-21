"""triage() / triage_batch() tests (S2). The model tests skip until models/triage.joblib exists."""
import pytest

import src.triage as T

needs_model = pytest.mark.skipif(not T.MODEL_PATH.exists(),
                                  reason="run python scripts/train_triage.py first")
TICKETS = [
    ("VPN down at the hospital", "Since 8am nobody can reach the patient records system over VPN."),
    (None, "Ich wurde für meine Rechnung im Oktober doppelt belastet. Bitte erstatten."),
    ("Pricing question", "What does the enterprise plan cost for 200 seats?"),
    ("", ""),
]


@needs_model
def test_labels_come_from_the_training_classes():
    model = T._load_model()
    for subject, body in TICKETS:
        r = T.triage(subject, body)
        for name in ("type", "queue", "priority"):
            assert r[name] in model["heads"][name]["clf"].classes_
            assert 0.0 <= r["conf"][name] <= 1.0
        assert r["overall_conf"] == min(r["conf"]["type"], r["conf"]["queue"])


@needs_model
def test_batch_matches_one_at_a_time():
    batch = T.triage_batch(TICKETS)
    single = [T.triage(s, b) for s, b in TICKETS]
    assert [r["type"] for r in batch] == [r["type"] for r in single]
    assert [r["priority"] for r in batch] == [r["priority"] for r in single]
    for a, b in zip(batch, single):
        assert a["conf"] == pytest.approx(b["conf"])


@needs_model
@pytest.mark.parametrize("subject, body", [
    ("VPN down - telemedicine and EMR cut off",
     "Our site-to-site VPN has been down since 06:00. The hospital cannot reach the EMR."),
    ("VPN ausgefallen - Patientenakte nicht erreichbar",
     "Seit 06:00 Uhr ist unser VPN ausgefallen. Das Krankenhaus kommt nicht an die Patientenakten."),
])
def test_hospital_vpn_outage_is_a_high_priority_incident(subject, body):
    r = T.triage(subject, body)
    assert (r["type"], r["priority"]) == ("Incident", "high")
    assert r["outage_terms"]


@needs_model
def test_model_info_reports_training():
    info = T.model_info()
    assert info["n_train"] > 0 and 0.0 < info["priority_threshold"] < 1.0


def test_placeholder_without_a_model(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "MODEL_PATH", tmp_path / "missing.joblib")
    T._load_model.cache_clear()
    try:
        with pytest.warns(UserWarning, match="train_triage"):
            r = T.triage("VPN down", "nobody can connect")
        assert r["type"] == "Incident" and r["overall_conf"] == 0.5
        assert r["outage_terms"] == ["vpn down"]   # wording is reported even without a model
        assert T.triage_batch([]) == []
    finally:
        T._load_model.cache_clear()
