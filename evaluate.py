"""Owner: S1. Usage: python evaluate.py [--n-cls N] [--n-gen 50]

Three stages, so we never run the LLM over ~10k tickets:
  A. Classification (triage_batch, FULL clean holdout): Macro-F1 type/queue/priority, language accuracy,
     High-Priority Incident Recall (classifier).
  B. Retrieval + gate (retrieve_batch + decide, FULL clean holdout, NO LLM): escalation rate,
     High-Priority Incident Recall (escalated), retrieval hit rates.
  C. End-to-end generation (run_ticket with the LLM, SAMPLE of --n-gen tickets): citation validity,
     warning rate, fallback rate, latency p50/p95.

DEFINITIONS (state these in the demo):
  High-Priority Incident Recall (escalated)  = of holdout tickets whose TRUE label is Incident + high,
      the fraction the system routes to human escalation.  <- the headline safety number
  High-Priority Incident Recall (classifier) = of those same tickets, the fraction for which triage
      predicts BOTH type=Incident and priority=high.

Inputs go through the pipeline (subject/body only). Labels are loaded separately, for scoring only.
"""
import argparse
import json
import time

import numpy as np
from sklearn.metrics import f1_score

from src.config import RESULTS_DIR, TARGETS
from src.data import load_test_inputs, load_test_labels
from src.gate import decide
from src.pipeline import run_ticket
from src.retrieval import retrieve_batch
from src.text import build_text
from src.triage import triage_batch


def _rate(mask, sel):
    sel = np.asarray(sel, dtype=bool)
    return float(np.asarray(mask)[sel].mean()) if sel.any() else None


def main(n_cls: int | None, n_gen: int):
    labels = load_test_labels()                      # clean rows only (task label scheme)
    if n_cls:
        labels = labels.head(n_cls)
    inputs = load_test_inputs().set_index("ticket_id").loc[labels.ticket_id]
    labels = labels.set_index("ticket_id")
    items = list(zip(inputs["subject"], inputs["body"]))
    texts = [build_text(s, b) for s, b in items]
    res = {"n_eval": len(items)}

    true_hi = ((labels["type"] == "Incident") & (labels["priority"] == "high")).values
    res["n_true_high_incident"] = int(true_hi.sum())

    # ---- Stage A: classification (full holdout)
    t = time.perf_counter()
    preds = triage_batch(items)
    res["triage_batch_seconds"] = round(time.perf_counter() - t, 2)
    for tgt in TARGETS:
        res[f"macro_f1_{tgt}"] = float(f1_score(labels[tgt], [p[tgt] for p in preds],
                                                average="macro", zero_division=0))
    res["language_accuracy"] = float(np.mean([p["language"] == l for p, l in zip(preds, labels["language"])]))
    pred_hi = np.array([p["type"] == "Incident" and p["priority"] == "high" for p in preds])
    res["hp_incident_recall_classifier"] = _rate(pred_hi, true_hi)

    # ---- Stage B: retrieval + gate (full holdout, no LLM)
    hits = retrieve_batch(texts)
    gates = [decide(p, h, tx) for p, h, tx in zip(preds, hits, texts)]
    esc = np.array([g["escalate"] for g in gates])
    res["escalation_rate"] = float(esc.mean())
    res["hp_incident_recall_escalated"] = _rate(esc, true_hi)
    res["auto_draft_queue_accuracy"] = _rate([p["queue"] == q for p, q in zip(preds, labels["queue"])], ~esc)
    q, ty = labels["queue"].values, labels["type"].values
    res["retrieval_top1_queue_match"] = float(np.mean([bool(h) and h[0]["queue"] == a for h, a in zip(hits, q)]))
    res["retrieval_anyk_queue_match"] = float(np.mean([any(x["queue"] == a for x in h) for h, a in zip(hits, q)]))
    res["retrieval_anyk_type_match"] = float(np.mean([any(x["type"] == a for x in h) for h, a in zip(hits, ty)]))

    # ---- Stage C: end-to-end generation (sample)
    lats, warn, fb, bad_cite = [], [], [], []
    for s, b in items[:n_gen]:
        r = run_ticket(s, b)
        lats.append(r["latency"]["total_ms"])
        d, ids = r["draft"], {h["ticket_id"] for h in r["hits"]}
        warn.append(bool(d["warnings"]))
        fb.append(bool(d["used_fallback"]))
        bad_cite.append(any(c not in ids for c in d["cited_ids"]))
    if lats:
        res.update({"gen_sample_n": len(lats), "gen_invalid_citation_rate": float(np.mean(bad_cite)),
                    "gen_warning_rate": float(np.mean(warn)), "gen_fallback_rate": float(np.mean(fb)),
                    "latency_p50_ms": float(np.percentile(lats, 50)),
                    "latency_p95_ms": float(np.percentile(lats, 95))})

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "results.json").write_text(json.dumps(res, indent=2))
    (RESULTS_DIR / "results.md").write_text(
        "| metric | value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in res.items()) + "\n",
        encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cls", type=int, default=None, help="limit stage A/B to first N clean holdout rows")
    ap.add_argument("--n-gen", type=int, default=50, help="sample size for stage C (LLM generation)")
    a = ap.parse_args()
    main(a.n_cls, a.n_gen)
