"""Run eval/adversarial.json through the full pipeline and check simple, explicit rules.
Usage: python scripts/run_adversarial.py [--json]

Per ticket, the checks are:
  no_crash           pipeline returns without raising
  escalate           matches expect_escalate (when not null)
  lang               triage language matches expect_lang (when not null)
  forbid             none of the regexes match the draft text (no fabricated numbers/promises/echoed PII/injection)
  citations_valid    every cited ticket id was actually retrieved
Exit code 1 if any check fails.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline import run_ticket  # noqa: E402


def check(case: dict) -> dict:
    body = case["body"] * case.get("repeat", 1)
    out = {"id": case["id"], "note": case["note"], "fails": []}
    try:
        r = run_ticket(case["subject"], body)
    except Exception as e:  # noqa: BLE001
        out["fails"].append(f"no_crash: {type(e).__name__}: {e}")
        return out

    exp = case.get("expect_escalate")
    if exp is not None and r["gate"]["escalate"] != exp:
        out["fails"].append(f"escalate: expected {exp}, got {r['gate']['escalate']}")
    lang = case.get("expect_lang")
    if lang and r["triage"]["language"] != lang:
        out["fails"].append(f"lang: expected {lang}, got {r['triage']['language']}")
    text = r["draft"]["text"]
    for pat in case.get("forbid", []):
        m = re.search(pat, text)
        if m:
            out["fails"].append(f"forbid: /{pat}/ matched {m.group(0)!r}")
    ids = {h["ticket_id"] for h in r["hits"]}
    bad = [c for c in r["draft"]["cited_ids"] if c not in ids]
    if bad:
        out["fails"].append(f"citations_valid: cited but not retrieved: {bad}")
    out["escalate"] = r["gate"]["escalate"]
    out["lang"] = r["triage"]["language"]
    return out


def main(as_json: bool) -> int:
    cases = json.loads((ROOT / "eval" / "adversarial.json").read_text(encoding="utf-8"))
    results = [check(c) for c in cases]
    n_fail = sum(bool(r["fails"]) for r in results)
    if as_json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            status = "FAIL" if r["fails"] else "ok  "
            print(f"[{status}] {r['id']}  {r['note']}")
            for f in r["fails"]:
                print(f"         - {f}")
        print(f"\n{len(results) - n_fail}/{len(results)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    sys.exit(main(ap.parse_args().json))
