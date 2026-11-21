"""Owner: S5. Glue. Never reads labels or the holdout; only takes raw subject/body.

Latency keys (ms): triage_ms, retrieve_ms, gate_ms, generate_ms, total_ms  (config.LATENCY_KEYS)
Two-stage escalation: decide() runs before generation; after generation, any draft warning
(invalid citation / stripped unsupported claim) also escalates the ticket.
"""
import time

from .gate import decide
from .generate import draft_reply
from .retrieval import retrieve
from .text import build_text
from .triage import triage


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


def run_ticket(subject, body) -> dict:
    t0 = time.perf_counter()
    lat = {}

    t = time.perf_counter()
    tri = triage(subject, body)
    lat["triage_ms"] = _ms(t)

    text = build_text(subject, body)
    t = time.perf_counter()
    hits = retrieve(text)
    lat["retrieve_ms"] = _ms(t)

    t = time.perf_counter()
    gate = decide(tri, hits, text)
    lat["gate_ms"] = _ms(t)

    t = time.perf_counter()
    draft = draft_reply(subject, body, tri["language"], hits)
    lat["generate_ms"] = _ms(t)

    if draft.get("warnings"):
        gate = {"escalate": True,
                "reasons": gate["reasons"] + ["Draft failed grounding check: " + "; ".join(draft["warnings"])]}
    draft = {**draft, "for_review": gate["escalate"]}

    lat["total_ms"] = _ms(t0)
    return {"triage": tri, "hits": hits, "gate": gate, "draft": draft, "latency": lat}
