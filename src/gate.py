"""Owner: S4. Deterministic, unit-testable escalation rules.

decide(triage_out, hits, text) -> {
  "escalate": bool,
  "reasons": [str],   # human-readable, shown in the UI badge
  "codes":   [str],   # machine-readable rule ids (additive key; evaluate.py can break escalations down by rule)
}

Rules (any one fires -> escalate). Every rule is a pure function of its inputs.
  HIGH_INCIDENT     priority == high and type == Incident
  SECURITY_INCIDENT ransomware, data leak/breach, hacked or compromised systems (en + de). The dataset
                    rarely labels these urgent (breach 24%, ransomware 15% high), so triage alone misses them.
  LOW_CONFIDENCE    overall_conf < CONF_THRESHOLD
  NO_PRECEDENT      no hits, or top retrieval score < SIM_THRESHOLD
  UNSUPPORTED_ASK   ticket asks for a refund / compensation / price / guarantee / SLA credit and
                    no retrieved resolution covers that topic
  LEGAL             legal threat or legal commitment request (always a human)
  PROMPT_INJECTION  text tries to steer the assistant ("ignore previous instructions", ...)
  EMPTY_TICKET      body too short to triage
  UNREADABLE        garbage or keyboard mash: no recognisable English/German word. Retrieval cannot
                    catch this (S3: "???" and mash score like real short tickets), so the gate does.
"""
from __future__ import annotations

import re

from .config import CONF_THRESHOLD, NO_SUBJECT, SIM_THRESHOLD

MIN_TEXT_CHARS = 15
MIN_LETTERS = 8

# --- Readability (garbage / keyboard mash) -----------------------------------------------------
# A real ticket, even a terse one ("VPN connection failure since update"), contains at least one
# common English/German word or IT-support term. Keyboard mash and symbol strings contain none.
_WORD = re.compile(r"[^\W\d_]{2,}")  # runs of 2+ unicode letters (umlauts and ß included)
COMMON_WORDS = frozenset("""
the an and or but if to of in on at by for from with without about into over after before since until
during between is are was were be been am do does did done have has had can could will would should may
might must not no yes it its this that these those there here what which who when where why how all any
some each every more most other only same so than too very just also again still already yet me my mine
we us our you your he him his she her they them their please thanks thank hello hi dear regards help need
want like get got make made work working works use using used issue issues problem problems error errors
question request urgent today yesterday since morning
vpn login log logon password account user users access server servers network internet wifi email mail
outlook system systems software hardware app application update upgrade install installed laptop computer
pc printer phone device data database file files backup cloud service services connection connect
connected down outage crash crashed slow fail failed failure broken invoice invoices billing bill payment
payments charge charged refund price pricing subscription license licence order product products support
ticket customer security breach report reset portal website site page browser integration api
der die das den dem des ein eine einen einem einer und oder aber wenn dass zu von mit für auf im an bei
nach seit vor über unter aus ist sind war waren wird werden hat haben hatte kann können konnte muss müssen
soll sollte möchte möchten bitte danke hallo guten tag sehr geehrte geehrter ich mich mir mein meine
meinen wir uns unser unsere sie ihr ihre ihnen es er nicht kein keine keinen noch schon auch nur wie was
warum wann wo welche dieser diese dieses hier dort gibt geht läuft funktioniert probleme fehler frage
anfrage hilfe dringend heute gestern morgen
zugang passwort konto benutzer netzwerk rechner drucker anwendung daten datei dateien sicherung
verbindung ausfall störung absturz langsam rechnung zahlung abrechnung rückerstattung erstattung preis
lizenz bestellung produkt kunde kunden sicherheit seite
""".split())

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
    # "quote" needs a word end (German "Engagement-Quoten" is not a quote request). "Angebot" only as the whole
    # word: "angebotenen" / "Angebote" (offered / offerings) matched 220 ordinary KB tickets; 37 real asks remain.
    ("specific pricing or discount",
     re.compile(r"\b(how much (does|will|would|is)|price\w*|pricing|quotes?\b|quotations?\b|discount\w*|"
                r"cost (of|for)|preis\w*|kosten f[üu]r|rabatt\w*|angebot\b|kostenvoranschl\w*|(was|wie ?viel) kostet)", _F),
     re.compile(r"(\b(price\w*|pricing|discount\w*|preis\w*|rabatt\w*)\b|[€$£]\s?\d|\d\s?(€|eur|usd)\b)", _F)),
    ("guarantee or SLA commitment",  # an "SLA dashboard/report" question asks for no commitment
     re.compile(r"\b(guarantee\w*|warrant(y|ies)|sla\b(?![- ](?:dashboard|report\w*|metric\w*|monitor\w*|status))|"
                r"service level|uptime commitment|"
                r"garantie\w*|garantier\w*|zusicher\w*|gew[äa]hrleist\w*)", _F),
     re.compile(r"\b(guarantee\w*|warrant(y|ies)|sla\b|service level|garantie\w*|garantier\w*|"
                r"gew[äa]hrleist\w*)", _F)),
]

# "court" only in legal phrasing (not "food court" or a court-booking app). "gericht" excludes "gerichtet(e)"
# ("directed at"), which made up 11 of the 12 KB tickets the old "gericht\w*" matched.
LEGAL_PATTERN = re.compile(
    r"\b(lawsuit|sue (you|your)|legal action|lawyer|attorney|(?:to|in|into|before) court|"
    r"court (?:order|case|action|claim|proceedings?|hearing)|liabilit\w*|litigation|"
    r"breach of contract|small claims|"
    r"anwalt|rechtsanwalt|klage\w*|verklagen|gericht(?!et)\w*|rechtliche schritte|haftung\w*)\b", _F)

# Security incidents (S1/S2 request). Deliberately narrow: bare "encrypt" / "verschlüssel" do NOT match,
# because encryption questions are routine. "breach of contract" is LEGAL's job and "SLA breach" is an
# SLA complaint, so both are excluded here.
SECURITY_PATTERN = re.compile(
    r"\b(ransom\w*|files? (?:are|were|got|have been) encrypted|encrypted (?:my|our|all) files|"
    r"data (?:leak\w*|breach\w*)|(?<!sla )breach(?:ed|es)?\b(?! of contract)|hacked|compromised|"
    r"datenleck\w*|datenpanne\w*|erpress\w*|l(?:ö|oe)segeld\w*|gehackt\w*|sicherheitsvorfall\w*|"
    r"(?:dateien|daten) (?:sind|wurden) (?:\w+ ){0,2}verschl(?:ü|ue)sselt)", _F)

INJECTION_PATTERN = re.compile(
    r"(ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts?|rules)|"
    r"disregard (all |any )?(the )?(previous|prior|above|your) (instructions|rules)|"
    r"forget (all |your )?(previous )?instructions|"
    r"(reveal|print|show|repeat) (me )?(your|the) (system )?prompt|system prompt|"
    r"you are now (a|an|in)\b|"
    # "act as a ..." only as an order to the assistant (start of a line/sentence, or "you (will) act as"),
    # not "can the gateway act as a proxy"; "developer mode" only together with dropping rules/restrictions.
    r"(?:(?:^|[\n.!?:])\s*|\byou\s+(?:will\s+|must\s+|should\s+|shall\s+|now\s+)?)act as (?:a|an)\b|"
    r"developer mode\b(?=[^.!?\n]*\b(?:ignore|unrestricted|uncensored|no (?:restrictions|rules|filters|limits)|"
    r"without (?:any )?(?:restrictions|rules|filters))\b)|jailbreak|"
    r"ignoriere (alle )?(vorherigen|bisherigen|obigen) (anweisungen|regeln)|"
    r"vergiss (alle )?(vorherigen )?anweisungen|systemprompt|system-prompt)", _F)


# --- Individual rules (pure; each returns a (code, reason) or None) -----------------------------

def rule_high_incident(tri: dict):
    if str(tri.get("priority", "")).lower() == "high" and str(tri.get("type", "")).lower() == "incident":
        return "HIGH_INCIDENT", "High-priority Incident: immediate on-call escalation"
    return None


def rule_security_incident(text: str):
    m = SECURITY_PATTERN.search(text)
    if m:
        return "SECURITY_INCIDENT", f"Possible security incident (\"{m.group(0)}\"): immediate on-call escalation"
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


def rule_unreadable(text: str):
    t = text.replace(NO_SUBJECT, "").strip()
    if len(t) < MIN_TEXT_CHARS:  # EMPTY_TICKET already covers it; don't report twice
        return None
    words = [w.lower() for w in _WORD.findall(t)]
    if sum(map(len, words)) < MIN_LETTERS or not any(w in COMMON_WORDS for w in words):
        return "UNREADABLE", "Ticket text has no recognisable words (garbage or keyboard mash)"
    return None


def decide(triage_out: dict, hits: list[dict], text: str) -> dict:
    tri = triage_out or {}
    hits = hits or []
    text = text if isinstance(text, str) else ""
    fired = [r for r in (
        rule_high_incident(tri),
        rule_security_incident(text),
        rule_empty(text),
        rule_unreadable(text),
        rule_prompt_injection(text),
        rule_legal(text),
        rule_unsupported_ask(text, hits),
        rule_low_confidence(tri),
        rule_no_precedent(hits),
    ) if r]
    return {"escalate": bool(fired), "reasons": [r[1] for r in fired], "codes": [r[0] for r in fired]}
