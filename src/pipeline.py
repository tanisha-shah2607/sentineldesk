"""Owner: Niti. Glue. Never reads labels or the holdout; only takes raw subject/body.

run_ticket(subject, body) -> {triage, hits, gate, draft(+for_review), latency}
Latency keys (ms): triage_ms, retrieve_ms, gate_ms, generate_ms, total_ms  (config.LATENCY_KEYS)
Two-stage escalation: decide() runs before generation; after generation, any draft warning
(invalid citation / stripped unsupported claim) also escalates the ticket.

Resilience (so the console works before every stage is finished):
  - A stage that raises NotImplementedError is replaced by its mock from src/mocks.py.
  - Any other stage error also falls back to the mock, unless strict mode is on
    (SENTINELDESK_STRICT=1, or run_ticket(..., strict=True)), in which case it is re-raised.
  - SENTINELDESK_FORCE_MOCKS=triage,retrieve,gate,generate (or "all") forces mocks for UI rehearsal.
  - A stage that answers but with a teammate's documented placeholder (triage without models/triage.joblib,
    retrieval in RETRIEVAL_STUB mode) is reported as "placeholder (...)", never as "live".
  Status values: "live" | "placeholder (...)" | "mock (...)". Anything but "live" makes the console show a banner;
  evaluate.py / run_adversarial.py run with SENTINELDESK_STRICT=1 and warn on any non-live stage.
  run_ticket_verbose() also returns which implementation each stage used; run_ticket() keeps the frozen shape.

PII: subject/body are masked with src.pii.mask_pii before draft_reply (the only stage that may call an
external LLM). Triage and retrieval run locally on the raw text.

CLI: python -m src.pipeline --check   runs src/examples.py through the pipeline and validates every contract.
"""
from __future__ import annotations

import os
import time

from . import mocks
from .contracts import validate_result
from .pii import mask_pii
from .text import build_text

STAGES = ("triage", "retrieve", "gate", "generate")
MASK_PII_FOR_LLM = True


def _real(stage: str):
    """Import lazily so a broken teammate module degrades one stage instead of the whole app."""
    if stage == "triage":
        from .triage import triage
        return triage
    if stage == "retrieve":
        from .retrieval import retrieve
        return retrieve
    if stage == "gate":
        from .gate import decide
        return decide
    from .generate import draft_reply
    return draft_reply


_MOCK = {"triage": mocks.mock_triage, "retrieve": mocks.mock_retrieve,
         "gate": mocks.mock_decide, "generate": mocks.mock_draft}


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def _forced() -> set[str]:
    v = os.getenv("SENTINELDESK_FORCE_MOCKS", "").strip().lower()
    return set(STAGES) if v == "all" else {s.strip() for s in v.split(",") if s.strip()}


def _placeholder(stage: str, out) -> str | None:
    """Detect teammates' documented placeholder modes, so they can't pass as live."""
    try:
        if stage == "triage":
            from .triage import MODEL_PATH
            if not MODEL_PATH.exists():
                return "placeholder (triage model not trained: run python scripts/train_triage.py)"
        if stage == "retrieve":
            stub = os.getenv("RETRIEVAL_STUB") == "1" or any(
                isinstance(h, dict) and h.get("subject") == "(stub)" for h in (out or []))
            if stub:
                return "placeholder (retrieval stub: run python scripts/build_index.py)"
    except Exception:  # noqa: BLE001
        pass
    return None


def _call(stage: str, args: tuple, status: dict, strict: bool):
    """Run the real stage, or its mock. Records 'live' | 'mock (...)' in status[stage]."""
    if stage in _forced():
        status[stage] = "mock (forced)"
        return _MOCK[stage](*args)
    try:
        out = _real(stage)(*args)
        status[stage] = _placeholder(stage, out) or "live"
        return out
    except NotImplementedError:
        status[stage] = "mock (not implemented)"
    except Exception as e:  # noqa: BLE001
        if strict:
            raise
        status[stage] = f"mock (error: {type(e).__name__}: {e})"
    return _MOCK[stage](*args)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


def run_ticket_verbose(subject, body, strict: bool | None = None) -> tuple[dict, dict]:
    """Like run_ticket, plus {stage: 'live' | 'mock (...)'} for the console's status panel."""
    strict = _env_flag("SENTINELDESK_STRICT") if strict is None else strict
    status: dict[str, str] = {}
    t0 = time.perf_counter()
    lat = {}

    t = time.perf_counter()
    tri = _call("triage", (subject, body), status, strict)
    lat["triage_ms"] = _ms(t)

    text = build_text(subject, body)
    t = time.perf_counter()
    hits = _call("retrieve", (text,), status, strict)
    lat["retrieve_ms"] = _ms(t)

    t = time.perf_counter()
    gate = _call("gate", (tri, hits, text), status, strict)
    lat["gate_ms"] = _ms(t)

    s_llm, b_llm = subject, body
    if MASK_PII_FOR_LLM:
        s_llm = mask_pii(subject) if isinstance(subject, str) else subject
        b_llm = mask_pii(body) if isinstance(body, str) else body
    t = time.perf_counter()
    draft = _call("generate", (s_llm, b_llm, tri["language"], hits), status, strict)
    lat["generate_ms"] = _ms(t)

    if draft.get("warnings"):
        gate = {**gate, "escalate": True,   # keep S4's extra keys (e.g. "codes")
                "reasons": gate["reasons"] + ["Draft failed grounding check: " + "; ".join(draft["warnings"])]}
        if "codes" in gate:
            gate["codes"] = list(gate["codes"]) + ["DRAFT_GROUNDING"]
    draft = {**draft, "for_review": gate["escalate"]}

    lat["total_ms"] = _ms(t0)
    return {"triage": tri, "hits": hits, "gate": gate, "draft": draft, "latency": lat}, status


def run_ticket(subject, body) -> dict:
    return run_ticket_verbose(subject, body)[0]


def check() -> int:
    """Smoke test: every example through the pipeline, every contract validated. Returns exit code."""
    from .examples import EXAMPLES, OUTAGE_BURST

    items = [(e["name"], e["subject"], e["body"], e.get("expect_precedent")) for e in EXAMPLES]
    items += [(f"burst #{i + 1}", e["subject"], e["body"], None) for i, e in enumerate(OUTAGE_BURST)]
    items += [("edge: NaN subject, None body", float("nan"), None, None), ("edge: empty", "", "", None)]
    try:  # load S3's model/index once so per-ticket latencies are realistic (the app does this at startup)
        from .retrieval import index_ready, warmup
        if index_ready():
            warmup()
    except Exception:  # noqa: BLE001
        pass
    failures, statuses, notes = 0, {}, []
    for name, s, b, expect in items:
        try:
            r, st = run_ticket_verbose(s, b)
            problems = validate_result(r)
        except Exception as e:  # noqa: BLE001
            r, st, problems = None, {}, [f"crashed: {type(e).__name__}: {e}"]
        for k, v in st.items():
            statuses.setdefault(k, set()).add(v)
        flag = "FAIL" if problems else "ok  "
        failures += bool(problems)
        summary = ""
        if r:
            t = r["triage"]
            summary = (f"{t['language']} {t['type']}/{t['priority']} conf={t['overall_conf']:.2f} "
                       f"{'ESCALATE' if r['gate']['escalate'] else 'auto-draft'} {r['latency']['total_ms']:.0f}ms")
        if expect and r and st.get("retrieve") == "live":
            ids = [h["ticket_id"] for h in r["hits"]]
            where = f"rank {ids.index(expect) + 1}" if expect in ids else "NOT in top-k"
            notes.append(f"{name}: expected precedent {expect} -> {where} (top: {ids[:3]})")
        print(f"[{flag}] {name:38s} {summary}")
        for p in problems:
            print(f"        - {p}")
    print("\nStage implementations used:")
    for k in STAGES:
        print(f"  {k:9s} {', '.join(sorted(statuses.get(k, {'?'})))}")
    if notes:
        print("\nCross-lingual demo precedents (informational):")
        for n in notes:
            print(f"  {n}")
    print(f"\n{len(items) - failures}/{len(items)} passed contract checks")
    return 1 if failures else 0


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="SentinelDesk pipeline")
    ap.add_argument("--check", action="store_true", help="run examples and validate all contracts")
    ap.add_argument("--subject", default=None)
    ap.add_argument("--body", default=None)
    a = ap.parse_args()
    if a.check or a.body is None:
        raise SystemExit(check())
    r, st = run_ticket_verbose(a.subject, a.body)
    print(json.dumps({"result": r, "stages": st}, indent=2, ensure_ascii=False))
