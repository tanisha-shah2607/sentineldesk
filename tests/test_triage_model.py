"""Unit tests for src/triage_model.py (S2). Need no data or trained models."""
import numpy as np
import pytest

from src.triage_model import (expected_calibration_error, final_labels, fit_temperature, make_head,
                              make_vectorizer, outage_terms, predict, priority_decision, softmax_t)


@pytest.mark.parametrize("text, terms", [
    ("VPN down - telemedicine cut off. It has been down since 06:00.", ["vpn down", "been down"]),
    ("Major outage: the login is down for everyone", ["outage", "login is down"]),
    ("Seit 6 Uhr ist das VPN ausgefallen, Patientenakte nicht mehr erreichbar", ["ausgefallen", "nicht mehr erreichbar"]),
    ("Wir vermuten einen Cyberangriff auf den Server", ["cyberangriff"]),
    ("The shop has been offline all morning", ["offline"]),
])
def test_outage_wording_found(text, terms):
    assert outage_terms(text) == terms


@pytest.mark.parametrize("text", [
    "Please slow down the sync job", "The drop-down menu is empty", "Engine breakdown report",
    "Download the invoice", "We shut down the old server last week", "Printer not working",
    "Possible data breach in billing", "", None,
])
def test_outage_lookalikes_ignored(text):
    assert outage_terms(text) == []


def test_outage_rule_forces_incident_high_with_honest_confidence():
    classes = {"type": np.array(["Change", "Incident", "Problem"]),
               "queue": np.array(["A", "B"]),
               "priority": np.array(["high", "low", "medium"])}
    probas = {"type": np.array([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]]),
              "queue": np.array([[0.6, 0.4], [0.6, 0.4]]),
              "priority": np.array([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])}
    labels, confs = final_labels(probas, classes, 0.3, outage=np.array([True, False]))
    assert list(labels["type"]) == ["Incident", "Problem"]
    assert list(labels["priority"]) == ["high", "medium"]
    assert np.allclose(confs["type"], [0.2, 0.7]) and np.allclose(confs["priority"], [0.1, 0.7])


def test_temperature_never_changes_the_prediction():
    scores = np.random.default_rng(0).normal(size=(50, 4))
    for t in (0.5, 1.0, 3.0):
        p = softmax_t(scores, t)
        assert np.allclose(p.sum(axis=1), 1.0)
        assert (p.argmax(axis=1) == scores.argmax(axis=1)).all()


def test_temperature_undoes_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=3000)
    scores = rng.normal(size=(3000, 3))
    scores[np.arange(3000), y] += 1.5
    t_fair = fit_temperature(scores, y)
    assert abs(fit_temperature(scores * 4, y) / t_fair - 4) < 0.2   # 4x too sharp -> 4x temperature


def test_priority_bar():
    classes = np.array(["high", "low", "medium"])
    p = np.array([[0.30, 0.20, 0.50], [0.10, 0.60, 0.30]])
    idx, conf = priority_decision(p, classes, 0.25)
    assert list(classes[idx]) == ["high", "low"]
    assert np.allclose(conf, [0.30, 0.60])   # probability of the label actually output
    idx, _ = priority_decision(p, classes, 0.35)
    assert list(classes[idx]) == ["medium", "low"]


def test_calibration_error():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, 20000)
    y = np.where(rng.uniform(size=20000) < conf, 0, 1)   # right exactly as often as stated
    honest = np.column_stack([conf, 1 - conf])
    sure = np.column_stack([np.full(20000, 0.99), np.full(20000, 0.01)])
    assert expected_calibration_error(honest, y) < 0.02
    assert expected_calibration_error(sure, y) > 0.15


def test_predict_returns_contract_fields():
    texts = ["vpn down at the hospital", "invoice charged twice", "password reset please",
             "server outage right now", "refund my order", "new laptop for a new hire"] * 3
    labels = {
        "type": ["Incident", "Request", "Request", "Incident", "Request", "Change"] * 3,
        "queue": ["IT Support", "Billing and Payments", "IT Support", "Technical Support",
                  "Billing and Payments", "IT Support"] * 3,
        "priority": ["high", "medium", "low", "high", "medium", "low"] * 3,
    }
    vec = make_vectorizer()
    X = vec.fit_transform(texts)
    heads = {n: {"clf": make_head(1.0).fit(X, y), "temperature": 1.0} for n, y in labels.items()}
    artifact = {"vectorizer": vec, "heads": heads, "priority_threshold": 0.3}
    out = predict(artifact, ["vpn down", "invoice charged twice"])
    assert len(out) == 2
    for r in out:
        assert set(r) == {"type", "queue", "priority", "conf", "overall_conf", "outage_terms"}
        assert r["priority"] in {"high", "medium", "low"}
        assert r["overall_conf"] == min(r["conf"]["type"], r["conf"]["queue"])
        assert all(0.0 <= v <= 1.0 for v in r["conf"].values())
