"""Shared input-text builder (S2 and S3 must both use this). No pandas import, cheap to load.
Retrieval-specific prefixes (e.g. e5 'query:'/'passage:') stay inside retrieval.py."""
from .config import NO_SUBJECT


def unescape(text) -> str:
    """The raw CSV stores ~21% of bodies (and 3% of answers) with the two characters backslash+n
    instead of real line breaks, so '\\n\\nich' would tokenize as the word 'nich'. Fix them.
    Use on answers too when displaying or prompting."""
    if not isinstance(text, str):
        return ""
    return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", " ")


def build_text(subject, body) -> str:
    """Canonical model input. Handles null/empty/NaN subject and body, and literal \\n escapes."""
    s = unescape(subject).strip() or NO_SUBJECT
    b = unescape(body)
    return f"{s}\n{b}".strip()
