"""Owner: S4. Live-LLM smoke test for draft_reply(): safety rules + latency. Needs NO data, index or model.

Usage:  python scripts/llm_smoke.py [--models openai/gpt-oss-120b] [--show] [--only ADV-21 ADV-22]
Needs a working LLM in .env (LLM_PROVIDER=groq | vertex | anthropic | ollama). On Groq's free tier (8K tokens/min)
the run waits out rate limits, so 20 tickets take several minutes and use roughly 50K of the 200K tokens/day. Each ticket in eval/adversarial.json goes
through draft_reply() with the 3 best-matching precedents from a small fixed set (written below in the KB's style,
ids TKT-9xxxx so they can never collide with real KB or holdout ids). Per draft it checks:
  live        the LLM answered (no template fallback)
  forbid      none of the case's forbid regexes match (fabricated numbers/promises, echoed PII, injection)
  citations   every cited id is one of the precedents given
  lang        the reply is in the case's expect_lang (when set)
and reports validator warnings and latency p50/p95 per model. Exit 1 if any check fails, 2 if no LLM is set up.

For full-pipeline numbers (real triage + retrieval) use scripts/run_adversarial.py and evaluate.py --n-gen 50
on a machine that has the data, index and model, with the same .env.
"""
import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import llm  # noqa: E402
from src.generate import ID_RE, _detect_lang, draft_reply  # noqa: E402
from src.triage import detect_language, language_text  # noqa: E402

PRECEDENTS = [
    {"ticket_id": "TKT-90001", "language": "en", "subject": "VPN keeps disconnecting",
     "answer": "Thank you for reporting this. Please update the VPN client to the latest version from the software "
               "portal and reconnect. If the connection still drops, restart your router and remove the cached VPN "
               "credentials before signing in again. If several users are affected, we escalate to the network team."},
    {"ticket_id": "TKT-90002", "language": "de", "subject": "VPN-Verbindung bricht ab",
     "answer": "Vielen Dank für Ihre Meldung. Bitte aktualisieren Sie den VPN-Client, erneuern Sie das Zertifikat im "
               "Self-Service-Portal und starten Sie den Rechner neu. Sind mehrere Standorte betroffen, übernimmt "
               "der Bereitschaftsdienst des Netzwerkteams."},
    {"ticket_id": "TKT-90003", "language": "en", "subject": "Login fails after update",
     "answer": "Please reset your password in the self-service portal and clear your browser cache. Locked accounts "
               "unlock automatically after 15 minutes. If the login still fails, reply with the exact error message."},
    {"ticket_id": "TKT-90004", "language": "en", "subject": "Charged twice on invoice",
     "answer": "We reviewed your invoice and opened a correction request with the billing team. The duplicate charge "
               "will be corrected on your next invoice. Billing will contact you if they need more details."},
    {"ticket_id": "TKT-90005", "language": "de", "subject": "Doppelte Abbuchung",
     "answer": "Wir haben die doppelte Abbuchung geprüft und eine Korrektur bei der Buchhaltung veranlasst. Der Betrag "
               "wird mit der nächsten Rechnung ausgeglichen. Bei Rückfragen meldet sich die Buchhaltung bei Ihnen."},
    {"ticket_id": "TKT-90006", "language": "en", "subject": "Integration with another platform",
     "answer": "Our product can export data through the REST API and as CSV files. We do not offer a native connector "
               "for that platform; the API documentation describes how to set up an export."},
    {"ticket_id": "TKT-90007", "language": "en", "subject": "Network outage at clinic",
     "answer": "The network team restored the failed core switch and services are back. Affected users were asked to "
               "reconnect. Major incidents are handled by the on-call engineer, who keeps you updated."},
    {"ticket_id": "TKT-90008", "language": "en", "subject": "Printer not printing",
     "answer": "Please reinstall the printer driver from the software portal and restart the print spooler service. "
               "If the printer is shared, check that it is online in the print server queue."},
]
_TOK = re.compile(r"[a-zäöüß]{4,}")


def pick_hits(text: str, k: int = 3) -> list[dict]:
    words = set(_TOK.findall(text.lower()))
    scored = sorted(PRECEDENTS, key=lambda p: -len(words & set(_TOK.findall((p["subject"] + " " + p["answer"]).lower()))))
    return [{**p, "score": 0.9, "snippet": p["answer"][:300]} for p in scored[:k]]


def pct(xs: list[float], q: float) -> str:
    xs = sorted(xs)
    return f"{xs[max(0, math.ceil(q * len(xs)) - 1)]:.0f} ms" if xs else "n/a"


def run_model(model: str, cases: list[dict], show: bool) -> int:
    os.environ["LLM_MODEL"] = model
    llm._clients.clear()  # the Vertex client's refusal-fallback setup depends on the model
    print(f"\n=== {model} ({os.getenv('LLM_PROVIDER')})")
    lat, n_fail, n_live, n_warn, n_handoff, waited = [], 0, 0, 0, 0, 0.0
    for c in cases:
        body = c["body"] * c.get("repeat", 1)
        lang = detect_language(language_text(c["subject"], body))
        hits = pick_hits(f"{c['subject'] or ''} {body}")
        llm.last_rate_wait_s = 0.0
        t0 = time.perf_counter()
        d = draft_reply(c["subject"], body, lang, hits)
        wait = llm.last_rate_wait_s
        ms = (time.perf_counter() - t0 - wait) * 1000  # model latency, without rate-limit waits
        waited += wait
        fails = []
        if d["used_fallback"]:
            fails.append(f"live: template fallback ({d.get('llm_error')})")
        else:
            n_live += 1
            lat.append(ms)
        for pat in c.get("forbid", []):
            m = re.search(pat, d["text"])
            if m:
                fails.append(f"forbid: /{pat}/ matched {m.group(0)!r}")
        given = {h["ticket_id"] for h in hits}
        bad = [i for i in d["cited_ids"] if i not in given]
        if bad:
            fails.append(f"citations: not given {bad}")
        got = _detect_lang(ID_RE.sub("", d["text"])) if len(d["text"]) > 60 else None
        if c.get("expect_lang") and got and got != c["expect_lang"]:
            fails.append(f"lang: expected {c['expect_lang']}, got {got}")
        n_warn += bool(d["warnings"])
        n_handoff += bool(d.get("abstained"))
        n_fail += bool(fails)
        extra = f"  (+{wait:.0f}s rate-limit wait)" if wait else ""
        print(f"[{'FAIL' if fails else 'ok  '}] {c['id']}  {ms:6.0f} ms  {c['note'][:58]}{extra}")
        for f in fails:
            print(f"         - {f}")
        if show and d.get("abstained"):
            print("         ~ hand-off: correctly declined, no precedent covers this request")
        for w in d["warnings"] if show else []:
            print(f"         ~ validator: {w}")
        if show:
            print("         | " + d["text"].replace("\n", "\n         | "))
    print(f"live replies {n_live}/{len(cases)} | checks passed {len(cases) - n_fail}/{len(cases)} | "
          f"hand-offs {n_handoff} | grounding warnings on {n_warn} | latency (live only) p50 {pct(lat, .5)}, p95 {pct(lat, .95)}"
          + (f" | rate-limit waits {waited:.0f}s (not in latency)" if waited else ""))
    return n_fail


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles choke on typographic characters
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[os.getenv("LLM_MODEL") or llm.default_model()])
    ap.add_argument("--show", action="store_true", help="print each draft and validator warning")
    ap.add_argument("--only", nargs="+", metavar="ID",
                    help="run only these cases, e.g. --only ADV-21 ADV-22 (saves free-tier tokens)")
    a = ap.parse_args()
    if not llm.llm_available():
        print("No live LLM configured: set LLM_PROVIDER (groq | vertex | anthropic | ollama) and its key in .env; "
              "see README.")
        return 2
    # batch run: always wait out per-minute limits (the .env value is tuned for live demos, not batches)
    os.environ["LLM_RATE_WAIT"] = os.getenv("LLM_SMOKE_RATE_WAIT", "65")
    cases = json.loads((ROOT / "eval" / "adversarial.json").read_text(encoding="utf-8"))
    if a.only:
        cases = [c for c in cases if c["id"] in set(a.only)]
        if not cases:
            print(f"No cases match {a.only}")
            return 2
    return 1 if sum(run_model(m, cases, a.show) for m in a.models) else 0


if __name__ == "__main__":
    sys.exit(main())
