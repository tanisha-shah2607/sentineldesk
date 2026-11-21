"""Owner: S2. The classifier behind triage(): features, calibration helpers, prediction.

Design (agreed with the team):
- One TF-IDF vectorizer (word 1-2 grams) shared by three logistic-regression heads:
  type, queue, priority. class_weight="balanced" because the score is Macro-F1, where
  small classes count as much as big ones.
- Each head is calibrated by temperature scaling on the shared validation slice:
  one number per head that makes probabilities honest without changing predictions.
- Priority says "high" when P(high) >= a threshold tuned on validation, to catch more
  high-priority incidents.
- Outage rule: tickets with clear outage wording (OUTAGE_PATTERN) are labelled
  Incident + high. Only 3 training tickets describe a VPN outage, so the model alone
  misreads hand-written outage tickets; the rule makes those escalate for the right reason.
- Every confidence is the calibrated probability of the label actually output, so a
  ticket labelled "high" by the bar or the rule can show a low conf.priority.
- overall_conf = min(conf.type, conf.queue).

Training is in scripts/train_triage.py. This module never reads data.
"""
from __future__ import annotations

import re

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

HEADS = ("type", "queue", "priority")
HIGH = "high"
INCIDENT = "Incident"

# Wording that, in the training tickets, marks a high-priority Incident about half the time or
# more (base rate 17%): "outage" 53%, "<system> down" 60%, "ausgefallen/Ausfall" 53%, "offline"
# 45%, "Angriff" 56%. Chosen on part="train". Weak signals such as "breach" (24%), "not working"
# (15%) and "nicht verfügbar" (21%) are left out, and so are "crash" and "interruption"
# (45%, but common in ordinary app-crash tickets: 2.6x the wrong flags for the extra recall).
OUTAGE_PATTERN = re.compile(
    r"\boutages?\b"
    r"|\b(?:is|are|was|were|been|went|goes|still)\s+down\b"
    r"|\b(?:vpn|servers?|network|systems?|site|website|portal|database|email|login|app|application"
    r"|services?|platform|internet|wifi)\s+(?:is\s+|are\s+)?down\b"
    r"|\boffline\b|\bunreachable\b|\bnot\s+reachable\b|\bnicht\s+(?:mehr\s+)?erreichbar\b"
    r"|\b(?:ausgefallen|ausfall|ausfälle|systemausfall)\b|\b(?:cyberangriff|angriff)\b",
    re.IGNORECASE,
)


def outage_terms(text: str) -> list[str]:
    """Outage wording found in a ticket: lowercased, in order of appearance, no repeats."""
    found: list[str] = []
    for match in OUTAGE_PATTERN.finditer(text or ""):
        term = " ".join(match.group(0).lower().split())
        if term not in found:
            found.append(term)
    return found


def make_vectorizer(char_ngrams: bool = False) -> FeatureUnion:
    """Word 1-2 grams, optionally plus character 3-5 grams.

    Characters are off by default: at C=16 on the validation slice, words alone scored
    type 0.895 / queue 0.682 / priority 0.714 Macro-F1 against 0.889 / 0.675 / 0.720
    with characters added, at about an eighth of the feature volume."""
    parts = [("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9,
                                      sublinear_tf=True, dtype=np.float32))]
    if char_ngrams:
        parts.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                              max_features=300_000, sublinear_tf=True,
                                              dtype=np.float32)))
    return FeatureUnion(parts)


def make_head(C: float) -> LogisticRegression:
    return LogisticRegression(C=C, class_weight="balanced", max_iter=3000)


def logits(clf, X) -> np.ndarray:
    """Per-class scores whose softmax is the model's probability (binary heads too)."""
    z = clf.decision_function(X)
    return np.column_stack([np.zeros_like(z), z]) if z.ndim == 1 else z


def softmax_t(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = scores / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(scores: np.ndarray, y_idx: np.ndarray) -> float:
    """Temperature that minimises log-loss. T > 1 softens an overconfident head."""
    rows = np.arange(len(y_idx))

    def nll(log_t: float) -> float:
        p = softmax_t(scores, float(np.exp(log_t)))
        return float(-np.mean(np.log(p[rows, y_idx] + 1e-12)))

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))


def expected_calibration_error(proba: np.ndarray, y_idx: np.ndarray, bins: int = 15) -> float:
    """Average gap between stated confidence and actual accuracy, weighted by ticket count."""
    conf = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_idx
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def priority_decision(proba: np.ndarray, classes, high_threshold: float):
    """'high' when P(high) >= high_threshold, else the likelier of the other labels.
    Returns (label index per row, probability of that label)."""
    hi = list(classes).index(HIGH)
    others = proba.copy()
    others[:, hi] = -1.0
    idx = np.where(proba[:, hi] >= high_threshold, hi, others.argmax(axis=1))
    return idx, proba[np.arange(len(idx)), idx]


def final_labels(probas: dict, classes: dict, priority_threshold: float, outage=None):
    """Labels and confidences from calibrated head probabilities, as triage() outputs them:
    most likely type and queue, priority with the lowered "high" bar, then the outage rule
    (Incident + high wherever `outage` is True). Shared by training and triage()."""
    rows = np.arange(len(probas["type"]))
    labels, confs = {}, {}
    for name in ("type", "queue"):
        idx = probas[name].argmax(axis=1)
        labels[name] = np.asarray(classes[name][idx], dtype=object)
        confs[name] = probas[name][rows, idx]
    idx, conf = priority_decision(probas["priority"], classes["priority"], priority_threshold)
    labels["priority"], confs["priority"] = np.asarray(classes["priority"][idx], dtype=object), conf
    if outage is not None and np.any(outage):
        inc = list(classes["type"]).index(INCIDENT)
        hi = list(classes["priority"]).index(HIGH)
        labels["type"] = np.where(outage, INCIDENT, labels["type"]).astype(object)
        confs["type"] = np.where(outage, probas["type"][:, inc], confs["type"])
        labels["priority"] = np.where(outage, HIGH, labels["priority"]).astype(object)
        confs["priority"] = np.where(outage, probas["priority"][:, hi], confs["priority"])
    return labels, confs


def head_proba(artifact: dict, name: str, X) -> np.ndarray:
    head = artifact["heads"][name]
    return softmax_t(logits(head["clf"], X), head["temperature"])


def predict(artifact: dict, texts: list[str]) -> list[dict]:
    """Labels, calibrated confidences and outage wording for texts made by src.text.build_text."""
    X = artifact["vectorizer"].transform(texts)
    probas = {name: head_proba(artifact, name, X) for name in HEADS}
    classes = {name: artifact["heads"][name]["clf"].classes_ for name in HEADS}
    terms = [outage_terms(t) for t in texts]
    outage = np.array([bool(t) for t in terms], dtype=bool) if artifact.get("outage_rule", True) else None
    labels, confs = final_labels(probas, classes, artifact["priority_threshold"], outage)
    out = []
    for i in range(len(texts)):
        conf = {name: float(confs[name][i]) for name in HEADS}
        out.append({
            "type": str(labels["type"][i]),
            "queue": str(labels["queue"][i]),
            "priority": str(labels["priority"][i]),
            "conf": conf,
            "overall_conf": min(conf["type"], conf["queue"]),
            "outage_terms": terms[i],
        })
    return out
