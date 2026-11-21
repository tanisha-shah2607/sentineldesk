"""Owner: S4. Deterministic, unit-testable escalation rules.

decide(triage_out, hits, text) -> {
  "escalate": bool,
  "reasons": [str],   # human-readable, shown in the UI badge
  "codes":   [str],   # machine-readable rule ids (additive key; evaluate.py can break escalations down by rule)
}

Rules (any one fires -> escalate). Every rule is a pure function of its inputs.
  HIGH_INCIDENT     priority == high and type == Incident
  LOW_CONFIDENCE    overall_conf < CONF_THRESHOLD
  NO_PRECEDENT      no hits, or top retrieval score < SIM_THRESHOLD
  UNSUPPORTED_ASK   ticket asks for a refund / compensation / price / guarantee / SLA credit and
                    no retrieved resolution covers that topic
  LEGAL             legal threat or legal commitment request (always a human)
  PROMPT_INJECTION  text tries to steer the assistant ("ignore previous instructions", ...)
  EMPTY_TICKET      body too short to triage
"""
from __future__ import annotations

import re

from .config import CONF_THRESHOLD, NO_SUBJECT, SIM_THRESHOLD

MIN_TEXT_CHARS = 15

# --- Unsupported-action detection -------------------------------------------------------------
# Each topic: (label, request pattern in the ticket, evidence pattern in retrieved answers).
# If the ticket matches the request pattern and NO retrieved answer matches the evidence pattern,
# the request is unsupported by precedent -> escalate. English + German.
_F = re.IGNORECASE
UNSUPPORTED_TOPICS: list[tuple[str, re.Pattern, re.Pattern]] = [
    ("refund or compensation",
     re.compile(r"\b(refund\w*|money back|reimburs\w*|compensat\w*|chargeback|credit note|"
                r"r[üu]ckerstatt\w*|erstatt\w*|geld zur[üu]ck|entsch[äa]dig\w*|gutschrift\w*)", _F),
     re.compile(r"\b(refund\w*|reimburs\w*|compensat\w*|credit note|r[üu]ckerstatt\w*|erstatt\w*|"
                r"entsch[äa]dig\w*|gutschrift\w*)", _F)),
    ("specific pricing or discount",
     re.compile(r"\b(how much (does|will|would|is)|price\w*|pricing|quote|quotation|discount\w*|"
                r"cost (of|for)|preis\w*|kosten f[üu]r|rabatt\w*|angebot\w*|(was|wie ?viel) kostet)", _F),
     re.compile(r"(\b(price\w*|pricing|discount\w*|preis\w*|rabatt\w*)\b|[€$£]\s?\d|\d\s?(€|eur|usd)\b)", _F)),
    ("guarantee or SLA commitment",
     re.compile(r"\b(guarantee\w*|warrant(y|ies)|sla\b|service level|uptime commitment|"
                r"garantie\w*|garantier\w*|zusicher\w*|gew[äa]hrleist\w*)", _F),
     re.compile(r"\b(guarantee\w*|warrant(y|ies)|sla\b|service level|garantie\w*|garantier\w*|"
                r"gew[äa]hrleist\w*)", _F)),
]

LEGAL_PATTERN = re.compile(
    r"\b(lawsuit|sue (you|your)|legal action|lawyer|attorney|court|liabilit\w*|litigation|"
    r"breach of contract|small claims|"
    r"anwalt|rechtsanwalt|klage\w*|verklagen|gericht\w*|rechtliche schritte|haftung\w*)\b", _F)

INJECTION_PATTERN = re.compile(
    r"(ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts?|rules)|"
    r"disregard (all |any )?(the )?(previous|prior|above|your) (instructions|rules)|"
    r"forget (all |your )?(previous )?instructions|"
    r"(reveal|print|show|repeat) (me )?(your|the) (system )?prompt|system prompt|"
    r"you are now (a|an|in)\b|act as (a|an) |developer mode|jailbreak|"
    r"ignoriere (alle )?(vorherigen|bisherigen|obigen) (anweisungen|regeln)|"
    r"vergiss (alle )?(vorherigen )?anweisungen|systemprompt|system-prompt)", _F)


# --- Individual rules (pure; each returns a (code, reason) or None) -----------------------------

def rule_high_incident(tri: dict):
    if str(tri.get("priority", "")).lower() == "high" and str(tri.get("type", "")).lower() == "incident":
        return "HIGH_INCIDENT", "High-priority Incident: immediate on-call escalation"
    return None


def rule_low_confidence(tri: dict, threshold: float = CONF_THRESHOLD):
    conf = tri.get("overall_conf")
    conf = 0.0 if conf is None else float(conf)
    if conf < threshold:
        return "LOW_CONFIDENCE", f"Low classifier confidence ({conf:.2f} < {threshold:.2f})"
    return None


def top_score(hits: list[dict]) -> float:
    return max((float(h.get("score") or 0.0) for h in hits), default=0.0)


def rule_no_precedent(hits: list[dict], threshold: float = SIM_THRESHOLD):
    if not hits:
        return "NO_PRECEDENT", "No historical precedent retrieved"
    top = top_score(hits)
    if top < threshold:
        return "NO_PRECEDENT", f"No close historical precedent (top similarity {top:.2f} < {threshold:.2f})"
    return None


def unsupported_requests(text: str, hits: list[dict]) -> list[str]:
    """Topic labels the ticket asks for that no retrieved answer covers."""
    evidence = " ".join(str(h.get("answer") or "") for h in hits)
    return [label for label, ask, ev in UNSUPPORTED_TOPICS if ask.search(text) and not ev.search(evidence)]


def rule_unsupported_ask(text: str, hits: list[dict]):
    missing = unsupported_requests(text, hits)
    if missing:
        return "UNSUPPORTED_ASK", "Asks for " + ", ".join(missing) + " not covered by any historical resolution"
    return None


def rule_legal(text: str):
    m = LEGAL_PATTERN.search(text)
    if m:
        return "LEGAL", f"Legal threat or legal commitment requested (\"{m.group(0)}\"): needs a human"
    return None


def rule_prompt_injection(text: str):
    m = INJECTION_PATTERN.search(text)
    if m:
        return "PROMPT_INJECTION", f"Possible prompt injection (\"{m.group(0)}\")"
    return None


def rule_empty(text: str):
    if len(text.replace(NO_SUBJECT, "").strip()) < MIN_TEXT_CHARS:
        return "EMPTY_TICKET", "Ticket text is empty or too short to triage"
    return None


def decide(triage_out: dict, hits: list[dict], text: str) -> dict:
    tri = triage_out or {}
    hits = hits or []
    text = text if isinstance(text, str) else ""
    fired = [r for r in (
        rule_high_incident(tri),
        rule_empty(text),
        rule_prompt_injection(text),
        rule_legal(text),
        rule_unsupported_ask(text, hits),
        rule_low_confidence(tri),
        rule_no_precedent(hits),
    ) if r]
    return {"escalate": bool(fired), "reasons": [r[1] for r in fired], "codes": [r[0] for r in fired]}
