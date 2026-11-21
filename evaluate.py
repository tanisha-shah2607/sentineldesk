"""Owner: S1. Usage: python evaluate.py [--n 500]

Inputs go through the pipeline (subject/body only). Labels are loaded separately
and used ONLY for scoring.
"""
import argparse
import json
import numpy as np
from sklearn.metrics import f1_score

from src.config import RESULTS_DIR
from src.data import load_test_inputs, load_test_labels
from src.pipeline import run_ticket


def main(n: int):
    labels = load_test_labels().head(n)          # clean rows only (task label scheme)
    inputs = load_test_inputs().set_index("ticket_id").loc[labels.ticket_id].reset_index()
    labels = labels.set_index("ticket_id")

    preds, lats, esc = [], [], []
    for row in inputs.itertuples():
        r = run_ticket(row.subject, row.body)
        preds.append(r["triage"])
        lats.append(r["latency"]["total_ms"])
        esc.append(r["gate"]["escalate"])

    res = {"n": len(inputs)}
    for tgt in ["type", "queue", "priority"]:
        y_pred = [p[tgt] for p in preds]
        res[f"macro_f1_{tgt}"] = f1_score(labels[tgt], y_pred, average="macro", zero_division=0)

    hi_inc = (labels["priority"] == "high") & (labels["type"] == "Incident")
    flagged = np.array([p["priority"] == "high" and p["type"] == "Incident" for p in preds])
    res["high_priority_incident_recall"] = float(flagged[hi_inc.values].mean()) if hi_inc.any() else None
    res["escalation_rate"] = float(np.mean(esc))
    res["latency_p50_ms"] = float(np.percentile(lats, 50))
    res["latency_p95_ms"] = float(np.percentile(lats, 95))
    # TODO(S1): retrieval hit-rate, escalation recall on high-priority incidents, per-class report

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    main(ap.parse_args().n)
