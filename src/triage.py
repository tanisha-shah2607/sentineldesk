"""Owner: S2. Replace the STUB bodies; keep signatures and return shape.

triage(subject, body) -> {
  "language": "en"|"de",
  "type": str, "queue": str, "priority": "high"|"medium"|"low",
  "conf": {"type": float, "queue": float, "priority": float},   # calibrated prob of the PREDICTED class
  "overall_conf": float
}
CONTRACT (decided by S2):
  - overall_conf = min(conf.type, conf.queue); the gate compares THIS to CONF_THRESHOLD.
    Priority is excluded (its confidence is low on most tickets); high+Incident still escalates via the gate.
  - conf.priority = probability of the label actually output, AFTER the recall-tuned threshold
    (a ticket flagged "high" at 0.35 shows 0.35).
  - Train only on load_kb_clean(part="train"); calibrate on part="val".
triage_batch(items) -> [triage dict, ...]  for items = [(subject, body), ...]. Used by evaluate.py on the
  full holdout; vectorize it (the default below just loops).
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

from .text import unescape

SUPPORTED_LANGUAGES = ("en", "de")
DEFAULT_LANGUAGE = "en"     # used only when the text has no letters to go on
_LANG_SEED = 0              # plain langdetect is random; a fixed seed makes it repeatable
_LANG_MAX_CHARS = 1000      # the language is clear long before 1,000 characters
_HAS_LETTER = re.compile(r"[^\W\d_]")


@lru_cache(maxsize=1)
def _language_factory():
    """langdetect loaded with only the English and German profiles.

    With just these two, every answer is "en" or "de", so a short German ticket
    can't come back as Dutch or Afrikaans the way it can with all 55 profiles."""
    from langdetect.detector_factory import PROFILES_DIRECTORY, DetectorFactory

    profiles = []
    for code in SUPPORTED_LANGUAGES:
        with open(os.path.join(PROFILES_DIRECTORY, code), encoding="utf-8") as f:
            profiles.append(f.read())
    factory = DetectorFactory()
    factory.load_json_profile(profiles)
    factory.set_seed(_LANG_SEED)
    return factory


def language_text(subject, body) -> str:
    """Subject + body as seen by language detection only.

    Unescaped like build_text (shared unescape), but without build_text's
    "[no subject]" placeholder: those English words tilt short German tickets."""
    return f"{unescape(subject).strip()}\n{unescape(body).strip()}".strip()


def detect_language_with_prob(text) -> tuple[str, float]:
    """Return (language, probability); language is always "en" or "de".

    Text with no letters (empty, digits, punctuation) returns
    (DEFAULT_LANGUAGE, 0.0): a guess, marked by the zero probability.
    The probability is overconfident on very short text; don't use it as a score."""
    if not isinstance(text, str) or not _HAS_LETTER.search(text):
        return DEFAULT_LANGUAGE, 0.0
    from langdetect.lang_detect_exception import LangDetectException

    detector = _language_factory().create()
    detector.set_max_text_length(_LANG_MAX_CHARS)
    detector.append(text)
    try:
        best = detector.get_probabilities()[0]
    except (LangDetectException, IndexError):
        return DEFAULT_LANGUAGE, 0.0
    return best.lang, float(best.prob)


def detect_language(text) -> str:
    return detect_language_with_prob(text)[0]


def triage(subject, body) -> dict:
    # STUB labels: S2 replaces these with real models loaded from MODEL_DIR
    return {
        "language": detect_language(language_text(subject, body)),
        "type": "Incident", "queue": "Technical Support", "priority": "medium",
        "conf": {"type": 0.5, "queue": 0.5, "priority": 0.5},
        "overall_conf": 0.5,
    }


def triage_batch(items) -> list[dict]:
    return [triage(s, b) for s, b in items]
