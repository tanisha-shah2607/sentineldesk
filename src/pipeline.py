"""Owner: S5. Glue. Never reads labels or the holdout; only takes raw subject/body."""
import time

from .data import build_text
from .gate import decide
from .generate import draft_reply
from .retrieval import retrieve
from .triage import triage


def run_ticket(subject, body) -> dict:
    t0 = time.perf_counter()
    lat = {}

    t = time.perf_counter()
    tri = triage(subject, body)
    lat["triage_ms"] = (time.perf_counter() - t) * 1000

    text = build_text(subject, body)
    t = time.perf_counter()
    hits = retrieve(text)
    lat["retrieval_ms"] = (time.perf_counter() - t) * 1000

    gate = decide(tri, hits, text)

    t = time.perf_counter()
    draft = draft_reply(subject, body, tri["language"], hits)
    lat["generation_ms"] = (time.perf_counter() - t) * 1000

    lat["total_ms"] = (time.perf_counter() - t0) * 1000
    return {"triage": tri, "hits": hits, "gate": gate, "draft": draft, "latency": lat}
