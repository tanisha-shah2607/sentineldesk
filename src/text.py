"""Shared input-text builder (S2 and S3 must both use this). No pandas import, cheap to load.
Retrieval-specific prefixes (e.g. e5 'query:'/'passage:') stay inside retrieval.py."""
from .config import NO_SUBJECT


def build_text(subject, body) -> str:
    """Canonical model input. Handles null/empty/NaN subject and body."""
    s = subject if isinstance(subject, str) and subject.strip() else NO_SUBJECT
    b = body if isinstance(body, str) else ""
    return f"{s}\n{b}".strip()
