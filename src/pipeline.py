"""Owner: S5. Glue. Never reads labels or the holdout; only takes raw subject/body.

Latency keys (ms): triage_ms, retrieve_ms, gate_ms, generate_ms, total_ms  (config.LATENCY_KEYS)
Two-stage escalation: decide() runs before generation; after generation, any draft warning
(invalid citation / stripped unsupported claim) also escalates the ticket, and so does a hand-off
(draft["abstained"]: the LLM correctly declined because no precedent covers the request), with its own reason.
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

    extra, extra_codes = [], []
    if draft.get("abstained"):
        extra.append("Draft hands off to a specialist: no retrieved precedent covers this request")
        extra_codes.append("DRAFT_HANDOFF")
    if draft.get("warnings"):
        extra.append("Draft failed grounding check: " + "; ".join(draft["warnings"]))
        extra_codes.append("DRAFT_GROUNDING")
    if extra:
        gate = {**gate, "escalate": True, "reasons": gate["reasons"] + extra}  # keep S4's extra keys (e.g. "codes")
        if "codes" in gate:
            gate["codes"] = list(gate["codes"]) + extra_codes
    draft = {**draft, "for_review": gate["escalate"]}

    lat["total_ms"] = _ms(t0)
    return {"triage": tri, "hits": hits, "gate": gate, "draft": draft, "latency": lat}
