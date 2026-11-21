"""Owner: S2. Replace the STUB body; keep the signature and return shape.

triage(subject, body) -> {
  "language": "en"|"de",
  "type": str, "queue": str, "priority": "high"|"medium"|"low",
  "conf": {"type": float, "queue": float, "priority": float},
  "overall_conf": float   # calibrated, 0..1
}
"""
from .data import build_text


def detect_language(text: str) -> str:
    try:
        from langdetect import detect
        lang = detect(text)
        return "de" if lang == "de" else "en"
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
