"""Owner: Niti. Contract-shaped stand-ins used ONLY when a real stage is not ready.

pipeline.py calls these when a teammate stage raises NotImplementedError (or any error in
non-strict mode), or when SENTINELDESK_FORCE_MOCKS names the stage. They are deliberately simple
keyword heuristics: good enough to demo the flow, never reported as model results.
Mock precedents use ids "MOCK-xxxx" so they can never be confused with real TKT- ids.
"""
from __future__ import annotations

import re

from .config import SIM_THRESHOLD, TOP_K
from .text import build_text

_DE_HINTS = {"der", "die", "das", "und", "nicht", "ist", "ich", "wir", "bitte", "seit", "kann", "keine", "mit",
             "unser", "ihr", "funktioniert", "rechnung", "hallo", "danke", "mein", "meine", "wurde", "wurden",
             "fuer", "für", "sie", "mir", "den", "dem", "uns", "ein", "eine", "auch", "oder", "aber", "haben",
             "sind", "werden", "bei", "zweimal", "dieselbe", "doppelt", "kein", "nach", "wie", "bestellung"}

_RULES = [  # (keywords, type, queue, priority)
    (("outage", "down", "ausfall", "offline", "not reachable", "nicht erreichbar", "breach", "hacked",
      "ransomware", "data loss", "datenverlust", "critical", "kritisch"),
     "Incident", "Technical Support", "high"),
    (("invoice", "billing", "refund", "charge", "payment", "rechnung", "zahlung", "erstattung"),
     "Request", "Billing and Payments", "medium"),
    (("error", "bug", "crash", "fails", "failing", "broken", "fehler", "absturz", "vpn", "login"),
     "Problem", "Technical Support", "medium"),
    (("how do i", "how can i", "question", "wie kann", "frage", "information"),
     "Request", "Customer Service", "low"),
    (("cancel", "upgrade", "subscription", "kündigen", "abo", "vertrag"),
     "Change", "Returns and Exchanges", "low"),
]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zäöüß]+", text.lower()))


def mock_language(text: str) -> str:
    try:  # same library S2's stub uses; the hint list below is only a fallback
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0
        return "de" if detect(text) == "de" else "en"
    except Exception:  # noqa: BLE001
        pass
    toks = _tokens(text)
    return "de" if len(toks & _DE_HINTS) >= 2 or re.search(r"[äöüß]", text.lower()) else "en"


def mock_triage(subject, body) -> dict:
    text = build_text(subject, body)
    low = text.lower()
    typ, queue, prio, conf = "Request", "Customer Service", "low", 0.40
    for kws, t, q, pr in _RULES:
        if any(k in low for k in kws):
            typ, queue, prio, conf = t, q, pr, 0.62
            break
    c = {"type": conf, "queue": conf - 0.05, "priority": conf - 0.02}
    return {"language": mock_language(text), "type": typ, "queue": queue, "priority": prio,
            "conf": c, "overall_conf": min(c["type"], c["queue"])}


_KB = [
    ("MOCK-0001", "VPN connection drops", "Technical Support", "Problem", "en",
     "Please reinstall the VPN client (version 5.2 or later), clear cached credentials and reconnect. "
     "If the issue persists, send us the client log from Settings > Diagnostics."),
    ("MOCK-0002", "Invoice shows wrong amount", "Billing and Payments", "Request", "en",
     "We have reviewed your invoice. Our billing team will check the line items and send a corrected "
     "invoice to the billing contact on file once confirmed."),
    ("MOCK-0003", "Service unreachable for all users", "Technical Support", "Incident", "en",
     "Our engineers are investigating a service disruption. Please follow the status page for updates; "
     "we will notify you as soon as service is restored."),
    ("MOCK-0004", "Passwort zurücksetzen", "Technical Support", "Problem", "de",
     "Bitte nutzen Sie die Funktion 'Passwort vergessen' auf der Anmeldeseite. Der Link ist 30 Minuten gültig."),
    ("MOCK-0005", "Rechnung fehlerhaft", "Billing and Payments", "Request", "de",
     "Vielen Dank für Ihren Hinweis. Unser Abrechnungsteam prüft die Rechnung und meldet sich bei Ihnen."),
    ("MOCK-0006", "How to export reports", "Customer Service", "Request", "en",
     "You can export reports from Dashboard > Reports > Export. CSV and PDF formats are supported."),
]


_STOP = {"the", "and", "for", "are", "our", "you", "your", "with", "from", "this", "that", "can", "please",
         "since", "not", "have", "has", "get", "all", "any", "will", "der", "die", "das", "und", "ist", "ich",
         "wir", "bitte", "sie", "uns", "mit", "ein", "eine", "den", "dem", "auf", "von", "für", "nicht", "ihr"}


def _content(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zäöüß0-9]+", text.lower()) if len(w) > 2 and w not in _STOP}


def mock_retrieve(text: str, k: int = TOP_K) -> list[dict]:
    q = _content(text)
    hits = []
    for tid, subj, queue, typ, lang, ans in _KB:
        d = _content(subj + " " + ans)
        cos = len(q & d) / ((len(q) * len(d)) ** 0.5 or 1)
        # stretched relative to config.SIM_THRESHOLD so good matches clear it (0.55 today, 0.85 after S4 merges)
        score = min(0.99, SIM_THRESHOLD - 0.25 + 2 * cos) if cos else 0.0
        hits.append({"ticket_id": tid, "score": round(min(score, 0.99), 4), "subject": subj, "answer": ans,
                     "snippet": ans[:300], "queue": queue, "type": typ, "language": lang})
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:k]


def mock_decide(triage_out: dict, hits: list[dict], text: str) -> dict:
    from .config import CONF_THRESHOLD, SIM_THRESHOLD
    reasons = []
    if triage_out["priority"] == "high" and triage_out["type"] == "Incident":
        reasons.append("High-priority Incident: immediate human escalation")
    if triage_out["overall_conf"] < CONF_THRESHOLD:
        reasons.append(f"Low classifier confidence ({triage_out['overall_conf']:.2f} < {CONF_THRESHOLD})")
    top = hits[0]["score"] if hits else 0.0
    if top < SIM_THRESHOLD:
        reasons.append(f"No close historical precedent (top similarity {top:.2f} < {SIM_THRESHOLD})")
    return {"escalate": bool(reasons), "reasons": reasons}


def mock_draft(subject, body, lang: str, hits: list[dict]) -> dict:
    if not hits:
        text = ("Danke für Ihre Nachricht. Wir leiten Ihr Anliegen an einen Mitarbeiter weiter."
                if lang == "de" else "Thank you for contacting us. We are forwarding your request to a human agent.")
        return {"text": text, "cited_ids": [], "warnings": [], "used_fallback": True}
    top = hits[0]
    intro = "Basierend auf einem ähnlichen Fall" if lang == "de" else "Based on a similar past case"
    return {"text": f"{intro} [{top['ticket_id']}]: {top['answer']}", "cited_ids": [top["ticket_id"]],
            "warnings": [], "used_fallback": True}
