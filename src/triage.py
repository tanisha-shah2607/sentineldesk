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
from .text import build_text


def detect_language(text: str) -> str:
    try:
        from langdetect import detect
        return "de" if detect(text) == "de" else "en"
    except Exception:
        return "en"


def triage(subject, body) -> dict:
    text = build_text(subject, body)
    # STUB: S2 replaces with real models loaded from MODEL_DIR
    return {
        "language": detect_language(text),
        "type": "Incident", "queue": "Technical Support", "priority": "medium",
        "conf": {"type": 0.5, "queue": 0.5, "priority": 0.5},
        "overall_conf": 0.5,
    }


def triage_batch(items) -> list[dict]:
    return [triage(s, b) for s, b in items]
