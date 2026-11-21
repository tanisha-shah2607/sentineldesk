"""Owner: S1. Usage: python evaluate.py [--n-cls N] [--n-gen 50]

Three stages, so we never run the LLM over ~10k tickets:
  A. Classification (triage_batch, FULL clean holdout): Macro-F1 type/queue/priority,
     High-Priority Incident Recall (classifier).
  B. Retrieval + gate (retrieve_batch + decide, FULL clean holdout, NO LLM): escalation rate,
     High-Priority Incident Recall (escalated), retrieval hit rates.
  C. End-to-end generation (run_ticket with the LLM, SAMPLE of --n-gen tickets): citation validity,
     warning rate, fallback rate, latency p50/p95.

HONEST REPORTING (the dataset contains duplicates):
  ~27% of clean holdout tickets have an exact-body twin in the KB, and twins share labels ~99.7% of the
  time. Every stage A/B number is therefore reported THREE ways:  all_*  (headline for the brief),
  never_seen_*  (holdout tickets with no twin in the KB; the fair estimate for new tickets), and, for
  Macro-F1, by DETECTED language (the dataset's `language` label is wrong on ~11% of tickets, so we
  group by triage()["language"], never by the label).

DEFINITIONS (state these in the demo):
  High-Priority Incident Recall (escalated)  = of holdout tickets whose TRUE label is Incident + high,
      the fraction the system routes to human escalation.  <- the headline safety number
  High-Priority Incident Recall (classifier) = of those same tickets, the fraction for which triage
      predicts BOTH type=Incident and priority=high.

Inputs go through the pipeline (subject/body only). Labels are loaded separately, for scoring only.
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.metrics import f1_score

# Strict mode: stage C must RAISE on a stage error instead of silently substituting a mock,
# otherwise the generation numbers would be measured on fake data.
os.environ.setdefault("SENTINELDESK_STRICT", "1")

from src import pipeline  # noqa: E402
from src.config import RESULTS_DIR, TARGETS  # noqa: E402
from src.data import holdout_seen_ids, load_test_inputs, load_test_labels  # noqa: E402
from src.gate import decide  # noqa: E402
from src.retrieval import retrieve_batch  # noqa: E402
from src.text import build_text
from src.triage import triage_batch


def _rate(flag, sel):
    sel = np.asarray(sel, dtype=bool)
    return float(np.asarray(flag)[sel].mean()) if sel.any() else None


def _f1(y_true, y_pred, sel):
    sel = np.asarray(sel, dtype=bool)
    if not sel.any():
        return None
    return float(f1_score(np.asarray(y_true)[sel], np.asarray(y_pred)[sel], average="macro", zero_division=0))


def main(n_cls: int | None, n_gen: int):
    labels = load_test_labels()                      # clean rows only (task label scheme)
    if n_cls:
        labels = labels.head(n_cls)
    inputs = load_test_inputs().set_index("ticket_id").loc[labels.ticket_id]
    seen_ids = holdout_seen_ids()
    labels = labels.set_index("ticket_id")
    items = list(zip(inputs["subject"], inputs["body"]))
    texts = [build_text(s, b) for s, b in items]
    seen = np.array([t in seen_ids for t in labels.index])
    subsets = {"all": np.ones(len(items), dtype=bool), "never_seen": ~seen}

    true_hi = ((labels["type"] == "Incident") & (labels["priority"] == "high")).values
    res = {"n_eval": len(items), "n_twin_in_kb": int(seen.sum()), "n_never_seen": int((~seen).sum()),
           "n_true_high_incident": int(true_hi.sum()),
           "n_true_high_incident_never_seen": int((true_hi & ~seen).sum())}

    # ---- Stage A: classification (full holdout)
    t = time.perf_counter()
    preds = triage_batch(items)
    res["triage_batch_seconds"] = round(time.perf_counter() - t, 2)
    det_lang = np.array([p["language"] for p in preds])
    pred_hi = np.array([p["type"] == "Incident" and p["priority"] == "high" for p in preds])
    for name, sel in subsets.items():
        for tgt in TARGETS:
            res[f"{name}_macro_f1_{tgt}"] = _f1(labels[tgt].values, [p[tgt] for p in preds], sel)
        res[f"{name}_hp_incident_recall_classifier"] = _rate(pred_hi, true_hi & sel)
    for lang in ("en", "de"):                        # grouped by DETECTED language, not the label
        sel = det_lang == lang
        res[f"detected_{lang}_n"] = int(sel.sum())
        for tgt in TARGETS:
            res[f"detected_{lang}_macro_f1_{tgt}"] = _f1(labels[tgt].values, [p[tgt] for p in preds], sel)
    res["language_label_agreement_NOT_ACCURACY"] = float(np.mean(det_lang == labels["language"].values))

    # ---- Stage B: retrieval + gate (full holdout, no LLM)
    hits = retrieve_batch(texts)
    gates = [decide(p, h, tx) for p, h, tx in zip(preds, hits, texts)]
    esc = np.array([g["escalate"] for g in gates])
    # which gate rule fired (S4's decide() returns machine-readable codes); a ticket can fire several
    code_counts = {}
    for g in gates:
        for c in g.get("codes", []):
            code_counts[c] = code_counts.get(c, 0) + 1
    res["gate_rule_share_of_tickets"] = {c: round(n / len(gates), 3) for c, n in sorted(code_counts.items())}
    q, ty = labels["queue"].values, labels["type"].values
    q_ok = np.array([p["queue"] == a for p, a in zip(preds, q)])
    top1_q = np.array([bool(h) and h[0]["queue"] == a for h, a in zip(hits, q)])
    anyk_q = np.array([any(x["queue"] == a for x in h) for h, a in zip(hits, q)])
    anyk_t = np.array([any(x["type"] == a for x in h) for h, a in zip(hits, ty)])
    for name, sel in subsets.items():
        res[f"{name}_escalation_rate"] = _rate(esc, sel)
        res[f"{name}_hp_incident_recall_escalated"] = _rate(esc, true_hi & sel)
        res[f"{name}_auto_draft_queue_accuracy"] = _rate(q_ok, ~esc & sel)
        res[f"{name}_retrieval_top1_queue_match"] = _rate(top1_q, sel)
        res[f"{name}_retrieval_anyk_queue_match"] = _rate(anyk_q, sel)
        res[f"{name}_retrieval_anyk_type_match"] = _rate(anyk_t, sel)

    # ---- Stage C: end-to-end generation (sample)
    lats, warn, fb, bad_cite, stage_status, abst, llm_errs = [], [], [], [], {}, [], {}
    for s, b in items[:n_gen]:
        if hasattr(pipeline, "run_ticket_verbose"):
            r, status = pipeline.run_ticket_verbose(s, b)
            for k, v in status.items():
                stage_status.setdefault(k, set()).add(v)
        else:
            r = pipeline.run_ticket(s, b)
        lats.append(r["latency"]["total_ms"])
        d, ids = r["draft"], {h["ticket_id"] for h in r["hits"]}
        warn.append(bool(d["warnings"]))          # real grounding failure (DRAFT_GROUNDING)
        abst.append(bool(d.get("abstained")))     # correct refusal / hand-off (DRAFT_HANDOFF): NOT a failure
        fb.append(bool(d["used_fallback"]))
        if d.get("llm_error"):
            e = str(d["llm_error"])[:60]
            llm_errs[e] = llm_errs.get(e, 0) + 1
        bad_cite.append(any(c not in ids for c in d["cited_ids"]))
    if lats:
        res.update({"gen_sample_n": len(lats), "gen_invalid_citation_rate": float(np.mean(bad_cite)),
                    "gen_grounding_failure_rate": float(np.mean(warn)),
                    "gen_abstain_rate": float(np.mean(abst)),
                    "gen_fallback_rate": float(np.mean(fb)), "gen_llm_errors": llm_errs,
                    "latency_p50_ms": float(np.percentile(lats, 50)),
                    "latency_p95_ms": float(np.percentile(lats, 95))})
        res["gen_stage_status"] = {k: sorted(v) for k, v in stage_status.items()}
        mocked = {k: sorted(v) for k, v in stage_status.items() if v != {"live"}}
        if mocked:
            res["WARNING_mock_stages_used"] = mocked
            print(f"\n*** WARNING: stage C used mocks {mocked}; generation numbers are NOT real ***\n")
        if res["gen_fallback_rate"] > 0:
            print(f"NOTE: {res['gen_fallback_rate']:.0%} of sampled drafts used the template fallback "
                  "(no live LLM); citation/warning rates then describe the template, not the model.")

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else v

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "results.json").write_text(json.dumps(res, indent=2))
    (RESULTS_DIR / "results.md").write_text(
        "| metric | value |\n|---|---|\n" + "\n".join(f"| {k} | {fmt(v)} |" for k, v in res.items()) + "\n",
        encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cls", type=int, default=None, help="limit stage A/B to first N clean holdout rows")
    ap.add_argument("--n-gen", type=int, default=50, help="sample size for stage C (LLM generation)")
    a = ap.parse_args()
    main(a.n_cls, a.n_gen)
