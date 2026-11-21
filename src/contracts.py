"""Owner: The team. Executable versions of the stage contracts documented in each module and README.

Every validator returns a list of human-readable problems ([] means OK). Used by
`python -m src.pipeline --check`, tests/test_pipeline.py and the console's status panel.
Keep these in sync with the docstrings at the top of triage.py / retrieval.py / gate.py / generate.py.
"""
from __future__ import annotations

from .config import LATENCY_KEYS, VALID_PRIORITIES

TRIAGE_KEYS = {"language", "type", "queue", "priority", "conf", "overall_conf"}
CONF_KEYS = {"type", "queue", "priority"}
HIT_KEYS = {"ticket_id", "score", "subject", "answer", "snippet", "queue", "type", "language"}
GATE_KEYS = {"escalate", "reasons"}
DRAFT_KEYS = {"text", "cited_ids", "warnings", "used_fallback"}
RESULT_KEYS = {"triage", "hits", "gate", "draft", "latency"}
LANGUAGES = {"en", "de"}


def _is_prob(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and 0.0 <= float(x) <= 1.0


def validate_triage(t) -> list[str]:
    if not isinstance(t, dict):
        return [f"triage: expected dict, got {type(t).__name__}"]
    p = [f"triage: missing {k}" for k in sorted(TRIAGE_KEYS - set(t))]
    if p:
        return p
    if t["language"] not in LANGUAGES:
        p.append(f"triage.language {t['language']!r} not in {sorted(LANGUAGES)}")
    if t["priority"] not in VALID_PRIORITIES:
        p.append(f"triage.priority {t['priority']!r} not in {VALID_PRIORITIES}")
    for k in ("type", "queue"):
        if not isinstance(t[k], str) or not t[k]:
            p.append(f"triage.{k} must be a non-empty str")
    conf = t["conf"]
    if not isinstance(conf, dict) or not CONF_KEYS <= set(conf):
        p.append(f"triage.conf must have keys {sorted(CONF_KEYS)}")
    else:
        p += [f"triage.conf.{k} not a probability" for k in sorted(CONF_KEYS) if not _is_prob(conf[k])]
        # S2 contract: priority is excluded (high+Incident still escalates via the gate)
        if not p and abs(t["overall_conf"] - min(conf["type"], conf["queue"])) > 1e-6:
            p.append("triage.overall_conf != min(conf.type, conf.queue)")
    if not _is_prob(t["overall_conf"]):
        p.append("triage.overall_conf not a probability")
    return p


def validate_hits(hits) -> list[str]:
    if not isinstance(hits, list):
        return [f"hits: expected list, got {type(hits).__name__}"]
    p = []
    for i, h in enumerate(hits):
        missing = HIT_KEYS - set(h) if isinstance(h, dict) else HIT_KEYS
        if missing:
            p.append(f"hits[{i}]: missing {sorted(missing)}")
            continue
        if not _is_prob(h["score"]):
            p.append(f"hits[{i}].score {h['score']!r} not in 0..1")
    scores = [h.get("score", 0) for h in hits if isinstance(h, dict)]
    if scores != sorted(scores, reverse=True):
        p.append("hits: not sorted by score desc")
    return p


def validate_gate(g) -> list[str]:
    if not isinstance(g, dict) or not GATE_KEYS <= set(g):
        return [f"gate: must be dict with {sorted(GATE_KEYS)}"]
    p = []
    if not isinstance(g["escalate"], bool):
        p.append("gate.escalate must be bool")
    if not isinstance(g["reasons"], list):
        p.append("gate.reasons must be a list")
    elif g["escalate"] and not g["reasons"]:
        p.append("gate: escalate=True but no reasons")
    return p


def validate_draft(d, hits=None, require_for_review: bool = False) -> list[str]:
    keys = DRAFT_KEYS | ({"for_review"} if require_for_review else set())
    if not isinstance(d, dict):
        return [f"draft: expected dict, got {type(d).__name__}"]
    p = [f"draft: missing {k}" for k in sorted(keys - set(d))]
    if p:
        return p
    if not isinstance(d["text"], str) or not d["text"].strip():
        p.append("draft.text is empty")
    if hits is not None:
        ids = {h.get("ticket_id") for h in hits if isinstance(h, dict)}
        bad = [c for c in d["cited_ids"] if c not in ids]
        if bad:
            p.append(f"draft cites ids not in hits: {bad}")
    return p


def validate_result(r) -> list[str]:
    """Full run_ticket() output."""
    if not isinstance(r, dict):
        return ["result: not a dict"]
    if set(r) != RESULT_KEYS:
        return [f"result keys {sorted(r)} != {sorted(RESULT_KEYS)}"]
    p = validate_triage(r["triage"]) + validate_hits(r["hits"]) + validate_gate(r["gate"])
    p += validate_draft(r["draft"], r["hits"], require_for_review=True)
    missing = [k for k in LATENCY_KEYS if k not in r["latency"]]
    if missing:
        p.append(f"latency: missing {missing}")
    if r["draft"].get("warnings") and not r["gate"].get("escalate"):
        p.append("draft has warnings but ticket was not escalated")
    if r["draft"].get("abstained") and not r["gate"].get("escalate"):
        p.append("draft abstained (hand-off) but ticket was not escalated")
    return p
