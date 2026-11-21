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
  - Outage rule: clear outage wording ("VPN down", "outage", "ausgefallen", "nicht erreichbar", ...)
    makes the ticket Incident + high, so the gate's high+Incident rule escalates it. The extra key
    "outage_terms" lists the wording found (empty list if none), for the UI's escalation reason.
triage_batch(items) -> [triage dict, ...]  for items = [(subject, body), ...]. Used by evaluate.py on the
  full holdout; one vectorizer pass for the whole batch.

The model is models/triage.joblib, built by `python scripts/train_triage.py`. Without it,
triage() still answers (placeholder labels, with a warning) so contract tests run anywhere.
"""
from __future__ import annotations

import os
import re
import warnings
from functools import lru_cache

from .config import MODEL_DIR
from .text import build_text, unescape

MODEL_PATH = MODEL_DIR / "triage.joblib"
SUPPORTED_LANGUAGES = ("en", "de")
DEFAULT_LANGUAGE = "en"     # used only when the text has no letters to go on
_LANG_SEED = 0              # plain langdetect is random; a fixed seed makes it repeatable
_LANG_MAX_CHARS = 1000      # the language is clear long before 1,000 characters
_HAS_LETTER = re.compile(r"[^\W\d_]")
_PLACEHOLDER = {"type": "Incident", "queue": "Technical Support", "priority": "medium",
                "conf": {"type": 0.5, "queue": 0.5, "priority": 0.5}, "overall_conf": 0.5}


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


@lru_cache(maxsize=1)
def _load_model():
    """The trained model, loaded once per process. None if it hasn't been trained here."""
    if not MODEL_PATH.exists():
        warnings.warn(f"{MODEL_PATH} not found, so triage() returns placeholder labels. "
                      "Run: python scripts/train_triage.py", stacklevel=2)
        return None
    import joblib

    return joblib.load(MODEL_PATH)


def model_info() -> dict | None:
    """When the model was trained and on what (for the UI or README); None without a model."""
    model = _load_model()
    return None if model is None else dict(model["meta"], priority_threshold=model["priority_threshold"])


def triage_batch(items) -> list[dict]:
    items = list(items)
    if not items:
        return []
    languages = [detect_language(language_text(s, b)) for s, b in items]
    model = _load_model()
    texts = [build_text(s, b) for s, b in items]
    if model is None:
        from .triage_model import outage_terms

        labels = [dict(_PLACEHOLDER, conf=dict(_PLACEHOLDER["conf"]), outage_terms=outage_terms(t))
                  for t in texts]
    else:
        from .triage_model import predict

        labels = predict(model, texts)
    return [{"language": lang, **lab} for lang, lab in zip(languages, labels)]


def triage(subject, body) -> dict:
    return triage_batch([(subject, body)])[0]
