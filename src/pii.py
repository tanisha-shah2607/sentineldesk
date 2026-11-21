"""Owner: S4 (bonus). Mask PII before any text goes to an external LLM.

mask_pii(text) -> str               masked text
mask_pii_report(text) -> (str, {label: count})

Masks: emails, URLs with credentials, IBANs, payment card numbers (Luhn-checked), IPv4/IPv6,
labelled account/customer/contract numbers (en + de), and phone numbers.
Leaves ticket ids (TKT-00012), dates, times, version numbers and short numbers alone.
"""
from __future__ import annotations

import re
from collections import Counter

_F = re.IGNORECASE

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
URL_CRED = re.compile(r"\b(https?://)[^\s/:@]+:[^\s/@]+@", _F)
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,3})?\b")
CARD = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
IPV4 = re.compile(r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b")
IPV6 = re.compile(r"\b(?:[0-9a-f]{1,4}:){7}[0-9a-f]{1,4}\b|\b(?:[0-9a-f]{1,4}:){1,6}:(?:[0-9a-f]{1,4}:?){1,6}\b", _F)
ACCOUNT = re.compile(
    r"\b((?:account|acct|customer|client|contract|member|policy|order|invoice)\s*"
    r"(?:no\.?|number|nr\.?|id|#)?\s*[:#]?\s*"
    r"|(?:konto|kunden|vertrags|mitglieds|bestell|rechnungs)(?:nummer|nr\.?)\s*[:#]?\s*)"
    r"([A-Z]{0,3}[-]?\d[\d-]{4,})", _F)
PHONE = re.compile(r"(?<![\w-])(?:\+|00)?\(?\d[\d ()/.-]{6,}\d(?![\w-])")
DATE_LIKE = re.compile(r"^\d{1,4}[./-]\d{1,2}[./-]\d{1,4}$")


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def mask_pii_report(text) -> tuple[str, dict]:
    if not isinstance(text, str) or not text:
        return ("" if not isinstance(text, str) else text), {}
    counts: Counter = Counter()

    def sub(pattern, label, s, check=None, group=None):
        def repl(m):
            if check and not check(m):
                return m.group(0)
            counts[label] += 1
            if group:  # keep the label text ("Account no: "), mask only the value
                return m.group(0)[: m.start(group) - m.start(0)] + f"[{label}]"
            return f"[{label}]"
        return pattern.sub(repl, s)

    def cred(m):  # keep scheme and host readable, drop user:password
        counts["CREDENTIALS"] += 1
        return m.group(1) + "[CREDENTIALS]@"

    s = URL_CRED.sub(cred, text)
    s = sub(EMAIL, "EMAIL", s)
    s = sub(IBAN, "IBAN", s, check=lambda m: sum(c.isdigit() for c in m.group(0)) >= 10)
    s = sub(CARD, "CARD", s, check=lambda m: _luhn_ok(re.sub(r"\D", "", m.group(0))))
    s = sub(IPV4, "IP", s)
    s = sub(IPV6, "IP", s, check=lambda m: m.group(0).count(":") >= 2 and any(c.isdigit() for c in m.group(0)))
    s = sub(ACCOUNT, "ACCOUNT", s, group=2)

    def phone_ok(m):
        raw = m.group(0).strip()
        digits = re.sub(r"\D", "", raw)
        if DATE_LIKE.match(raw) or not 8 <= len(digits) <= 15:
            return False
        # version strings like 10.2.3.4 or 2.14.0 are dot-only; phones use spaces, dashes, slashes, +, ()
        return raw.startswith(("+", "00", "(")) or bool(re.search(r"[ /()-]", raw))

    s = sub(PHONE, "PHONE", s, check=phone_ok)
    return s, dict(counts)


def mask_pii(text) -> str:
    return mask_pii_report(text)[0]
