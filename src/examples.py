"""Owner: Niti. Hand-written demo tickets (NOT from the dataset, so no leakage risk).

EXAMPLES feed the console's example picker and `python -m src.pipeline --check`.
ADVERSARIAL_DEMO_IDS are S1's adversarial cases (eval/adversarial.json, also hand-written, not dataset rows) that S1
asked to show in the console; they are appended to EXAMPLES from that file so the wording has one source.
OUTAGE_BURST is a run of near-duplicate reports (en + de) used to demo major-incident detection.
"""

EXAMPLES = [
    {"name": "VPN keeps dropping (en)", "subject": "VPN disconnects every 10 minutes",
     "body": "Since this morning my VPN connection drops every 10 minutes and I have to log in again. "
             "Other apps work fine. Laptop is Windows 11."},
    {"name": "Production outage (en, high)", "subject": "URGENT: production API down",
     "body": "Our production API has been down for 20 minutes, all customers get HTTP 503. "
             "This is a critical outage, please escalate immediately."},
    {"name": "Wrong invoice (en)", "subject": "Invoice INV-2291 amount incorrect",
     "body": "Hello, our latest invoice charges us for 40 seats but we only have 25. "
             "Can you correct the invoice and refund the difference?"},
    {"name": "Rechnung falsch (de)", "subject": "Rechnung fehlerhaft",
     "body": "Hallo, die letzte Rechnung ist leider fehlerhaft. Wir wurden doppelt belastet. "
             "Bitte prüfen Sie das und senden Sie uns eine korrigierte Rechnung."},
    {"name": "No subject + PII (en)", "subject": None,
     "body": "Hi, I can't log in. My email is jane.doe@example.com and my phone is +1 617 555 0199. "
             "The server at 10.20.30.40 rejects my password."},
    {"name": "How-to question (en)", "subject": "Export reports",
     "body": "How do I export the monthly usage report as a PDF?"},
    # Cross-lingual demo (S3, wording from scripts/cross_lingual.py PAIRS): German tickets whose precedents
    # include English tickets.
    # expect_precedent = S3's expected top hit with the real index (checked by `--check` when the index is built).
    {"name": "VPN ausgefallen (de, cross-lingual)", "subject": "VPN ausgefallen",
     "body": "Unser Site-to-Site-VPN ist seit heute Morgen ausgefallen. Mitarbeiter im Homeoffice erreichen das Klinik-EMR-System nicht.",
     "expect_precedent": "TKT-26021"},
    {"name": "App stürzt ab (de, cross-lingual)", "subject": "App stürzt ab",
     "body": "Nach dem letzten Software-Update stürzt die Anwendung beim Start ab.",
     "expect_precedent": "TKT-41866"},
    {"name": "Doppelt belastet (de, cross-lingual)", "subject": "Doppelt belastet",
     "body": "Ich wurde diesen Monat doppelt für mein Abonnement belastet, bitte prüfen Sie meine Rechnung.",
     "expect_precedent": "TKT-20534"},
    {"name": "Datenleck (de, cross-lingual)", "subject": "Datenleck",
     "body": "Wir vermuten ein Datenleck: unbekannte Anmeldungen an unseren Admin-Konten aus dem Ausland.",
     "expect_precedent": "TKT-14289"},
]

ADVERSARIAL_DEMO_IDS = ["ADV-01", "ADV-02", "ADV-05", "ADV-07"]


def _adversarial_examples() -> list[dict]:
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "eval" / "adversarial.json"
    try:
        cases = {c["id"]: c for c in json.loads(path.read_text(encoding="utf-8"))}
    except Exception:  # noqa: BLE001  (file missing/changed: the console still works without these)
        return []
    return [{"name": f"{i}: {cases[i]['note']}", "subject": cases[i]["subject"],
             "body": cases[i]["body"] * cases[i].get("repeat", 1)} for i in ADVERSARIAL_DEMO_IDS if i in cases]


EXAMPLES += _adversarial_examples()

OUTAGE_BURST = [
    {"subject": "Cannot log in", "body": "Login page shows error 500 since 9:05, nobody on our team can log in."},
    {"subject": "Login error 500", "body": "Getting error 500 on the login page, can't log in at all."},
    {"subject": None, "body": "Login broken? Error 500 when I try to log in since about 9am."},
    {"subject": "Anmeldung nicht möglich", "body": "Seit 9 Uhr kommt beim Login ein Fehler 500, wir können uns nicht anmelden."},
    {"subject": "Login down", "body": "The login page returns error 500 for everyone in our office."},
]
