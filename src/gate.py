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

# Plus the 300 words found in the most KB tickets (training part only; each is in at least 702 of 34,948),
# so terse but real tickets ("Encountering Periodic Performance Deterioration | Resolution Required",
# "Fehlerhafte Platine-Problematik") count as readable. Keyboard mash and lorem ipsum still match nothing.
KB_FREQUENT_WORDS = frozenset("""
across additional additionally address advice affected affecting agency analysis analytics angelegenheit
applications appreciate appreciated assist assistance attempted attempts attention aufgrund available back bald
beheben behält believe benötigen bereits bereitstellen best bestehen besteht better boost brand breaches cache
caches campaign campaigns capabilities cause caused causing changes checking clearing compatibility
comprehensive concern concerns confidentiality configuration configurations conflicts connections connectivity
contacting content continues critical current currently dank darauf dashboard datenanalyse decisions delays
despite detailed detaillierte details detected devices dies difficulties digital digitale digitalen disruptions
documentation due durch efforts encountered encountering encryption engagement enhance enhancing ensure erfahren
erhalten experienced experiencing facing falls features festgestellt financial firewall forward freue freuen
further future geben gerne gestartet grateful greatly growth guidance habe healthcare hearing helfen hospital
hören immediate impacting improve incident include including inform information informationen informed initial
inquire inquiry insbesondere insights integrating integrations integrity interessiert interested investment
jedoch kindly know kundenservice kundensupport könnte könnten learning let logs look looking lösen lösung making
management marketing matter maßnahmen measures media medical medizinische medizinischen medizinischer mehr
melden methods multiple möglich mögliche möglicherweise name necessary needed neu occurred offer online
optimierung optimization optimize optimizing options out outdated overload particularly patient patientendaten
performance persist persists platform platforms plattform possible potential practices prevent process
processing project projektmanagement prompt promptly protocols provide reaching recent recommendations records
regarding related remains reporting require required resolution resolve resolved resolving resources response
restart restarted restarting reviewed reviewing root rückmeldung saas schreibe schritte schätzen secure securing
seek seeking sein sensitive settings several share sich significant situation social solution solutions soon
specific specifically starten steps strategien strategies strategy such suspect systeme take taken team through
time tool tools tried troubleshooting unauthorized unexpected unresolved unserer unterstützung updated updates
updating urgently ursache various verbessern verbesserung versucht vielen weitere weiterhin well within writing
wurde wurden wäre würde zeit zur zurückzuführen zusätzliche überprüft
""".split())
COMMON_WORDS = COMMON_WORDS | KB_FREQUENT_WORDS | frozenset(
    "alle alles jetzt immer wieder mehr neu neue neuen leider doch dann gerade anweisungen".split())

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
    # Narrowed on the KB (38,810 tickets): "stock prices", "preisgeben" (disclose), "angebotene Dienstleistungen"
    # (offered services) and "quote the error" are not price requests.
    ("specific pricing or discount",
     re.compile(r"\b(how much (does|will|would|is|are)|(?<!stock )(?<!share )prices?\b|pricing|"
                r"(a|an|your|price|pricing) quote\b|quote for\b|quotation|discount\w*|cost (of|for)|"
                r"preis(?!geb)\w*|kosten f[üu]r|rabatt\w*|kostenvoranschlag\w*|preisangebot\w*|"
                r"(ein|ihr|unser|kein|konkretes|schriftliches|unverbindliches|individuelles) angebot\b|"
                r"angebot (f[üu]r|über|zu)\b|(was|wie ?viel) kostet)", _F),
     re.compile(r"(\b((?<!stock )(?<!share )prices?|pricing|discount\w*|preis(?!geb)\w*|rabatt\w*)\b|"
                r"[€$£]\s?\d|\d\s?(€|eur|usd)\b)", _F)),
    # Asks for a commitment only. On the KB, "guarantee/garantieren/gewährleisten" almost always mean "ensure"
    # ("to guarantee the secure management of...", "die Vertraulichkeit zu gewährleisten": 1,007 false hits),
    # and "SLA dashboard" is a technical ticket. The evidence side is narrow too, so "ensure" wording in an
    # answer never counts as a precedent for a guarantee.
    ("guarantee or SLA commitment",
     re.compile(r"\b((can|could|will|would|do) you (\w+ )?guarantee|guarantee (me|us)\b|"
                r"(a|any|written|full|uptime|money-back|100%?) guarantee|"
                r"guaranteed (uptime|refund|response|resolution|availability|delivery)|warrant(y|ies)|"
                r"sla (credit|breach|guarantee|commitment|compensation|penalt)\w*|"
                r"service[- ]level (agreement|credit|guarantee|commitment)\w*|uptime (guarantee|commitment)\w*|"
                r"(k[öo]nnen|k[öo]nnten|w[üu]rden) sie (uns |mir )?(\w+ )?garantieren|"
                r"(eine|keine|schriftliche|verbindliche) garantie|garantie (auf|f[üu]r)|"
                r"zusicher\w*|gew[äa]hrleistungs(anspruch|frist|fall)\w*)", _F),
     re.compile(r"\b(sla\b|service[- ]level|uptime (guarantee|commitment)|warrant(y|ies)|garantie\b|"
                r"garantiebedingungen|gew[äa]hrleistungs\w+|money-back|guarantee (policy|terms))", _F)),
]

# Narrowed on the KB: all 16 old hits were false ("gerichtete Werbung" = targeted ads, "klagend" = complaining).
LEGAL_PATTERN = re.compile(
    r"\b(lawsuit|sue (you|your|the company)|legal (action|steps|proceedings)|lawyer|attorney|"
    r"(see you|take (this|you|it|the matter)|go) (in|to) court|court (case|order|action|proceedings)|"
    r"litigation|breach of contract|small claims|(held|be) liable|liability for|"
    r"anwalt\w*|rechtsanwalt\w*|verklagen|klage (einreichen|erheben)|rechtliche schritte|vor gericht|"
    r"gerichtlich\w*|gerichtsverfahren\w*|haftung\w*)\b", _F)

# Security incidents (S1/S2 request). Deliberately narrow: bare "encrypt" / "verschlüssel" do NOT match,
# because encryption questions are routine. "breach of contract" is LEGAL's job and "SLA breach" is an
# SLA complaint, so both are excluded here.
SECURITY_PATTERN = re.compile(
    r"\b(ransom\w*|files? (?:are|were|got|have been) encrypted|encrypted (?:my|our|all) files|"
    r"data (?:leak\w*|breach\w*)|(?<!sla )breach(?:ed|es)?\b(?! of contract)|hacked|compromised|"
    r"datenleck\w*|datenpanne\w*|erpress\w*|l(?:ö|oe)segeld\w*|gehackt\w*|sicherheitsvorfall\w*|"
    r"(?:dateien|daten) (?:sind|wurden) (?:\w+ ){0,2}verschl(?:ü|ue)sselt)", _F)

# Narrowed: the old "system prompt" matched "restore the system promptly" (both KB hits were false), and
# "developer mode" / "act as a ..." / "you are now a ..." are normal phrases in IT tickets. Role-play now
# only counts when it targets the assistant itself ("act as an unrestricted AI").
INJECTION_PATTERN = re.compile(
    r"(ignore (all |any )?(the |your )?(previous|prior|above|earlier) (instructions|prompts?|rules)|"
    r"disregard (all |any )?(the )?(previous|prior|above|your) (instructions|rules)|"
    r"forget (all |your )?(previous )?instructions|"
    r"(reveal|print|show|repeat|tell me) (me )?your (system |initial |hidden )?(prompt|instructions)\b|"
    r"\bsystem[- ]?prompt\b|"
    r"(you are now|act as|pretend to be) (an? )?(unrestricted |uncensored |jailbroken )?"
    r"(ai|assistant|chatbot|gpt|llm|language model|dan)\b|"
    r"\bjailbreak\w*|\bdo anything now\b|"
    r"ignoriere (alle )?(vorherigen|bisherigen|obigen) (anweisungen|regeln)|"
    r"vergiss (alle )?(vorherigen )?anweisungen|\bsystemprompt\b)", _F)


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
