"""Owner: S1. Escalation-rate vs safety trade-off.  Usage: python scripts/tune_gate.py

Question: today the gate escalates about half of all tickets. Which (priority bar, CONF_THRESHOLD) setting
keeps High-Priority Incident recall high while auto-drafting more routine tickets?

Data: the shared VALIDATION slice only (load_kb_clean(part="val")), never the holdout. Labels from the KB are
used for scoring only. S2's model was trained on part="train" and calibrated on val, so val is slightly
optimistic; confirm the chosen setting on the holdout with evaluate.py afterwards.

Gate rules simulated (all text/triage-only): HIGH_INCIDENT, LOW_CONFIDENCE(T1), EMPTY, UNREADABLE,
PROMPT_INJECTION, LEGAL. NOT simulated (need retrieval): NO_PRECEDENT (S3: ~0% of val at 0.85) and
UNSUPPORTED_ASK. So the escalation rates below are a LOWER BOUND on the real ones.

Definitions: "routine" = true label is NOT (Incident + high). auto-draft rate = share of routine tickets the
gate does not escalate. escalated recall = share of true Incident+high tickets the gate escalates.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from src import gate  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402
from src.data import load_kb_clean, norm_body  # noqa: E402
from src.text import build_text  # noqa: E402
from src.triage import MODEL_PATH  # noqa: E402
from src.triage_model import final_labels, head_proba, outage_terms  # noqa: E402

BARS = [0.30, 0.40, 0.50, 0.60]
T1S = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]


def main():
    if not MODEL_PATH.exists():
        sys.exit("models/triage.joblib missing: run python scripts/train_triage.py")
    art = joblib.load(MODEL_PATH)
    val = load_kb_clean(part="val")
    train_keys = set(norm_body(load_kb_clean(part="train")["body"]))
    never_seen = ~norm_body(val["body"]).isin(train_keys).to_numpy()
    texts = [build_text(s, b) for s, b in zip(val["subject"], val["body"])]

    X = art["vectorizer"].transform(texts)
    probas = {n: head_proba(art, n, X) for n in ("type", "queue", "priority")}
    classes = {n: art["heads"][n]["clf"].classes_ for n in probas}
    outage = np.array([bool(outage_terms(t)) for t in texts])

    text_rule = np.array([any(r(t) for r in (gate.rule_empty, gate.rule_unreadable,
                                             gate.rule_prompt_injection, gate.rule_legal)) for t in texts])
    true_hi_inc = ((val["type"] == "Incident") & (val["priority"] == "high")).to_numpy()
    routine = ~true_hi_inc
    q_true = val["queue"].to_numpy()

    rows = []
    for bar in BARS:
        labels, confs = final_labels(probas, classes, bar, outage)
        high_inc = (labels["priority"] == "high") & (labels["type"] == "Incident")
        overall = np.minimum(confs["type"], confs["queue"])
        q_ok = labels["queue"] == q_true
        for t1 in T1S:
            esc = high_inc | (overall < t1) | text_rule
            for name, sel in (("all", np.ones(len(val), bool)), ("never_seen", never_seen)):
                auto = ~esc & sel
                rows.append({
                    "priority_bar": bar, "T1": t1, "subset": name, "n": int(sel.sum()),
                    "escalation_rate": round(float(esc[sel].mean()), 3),
                    "escalated_recall_hp_incident": round(float(esc[sel & true_hi_inc].mean()), 3),
                    "auto_draft_rate_routine": round(float((~esc)[sel & routine].mean()), 3),
                    "queue_acc_of_auto_drafts": round(float(q_ok[auto].mean()), 3) if auto.any() else None,
                    "share_from_high_incident_rule": round(float(high_inc[sel].mean()), 3),
                    "share_predicted_high_among_routine": round(float(high_inc[sel & routine].mean()), 3),
                })

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "gate_tradeoff.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"val n={len(val)} (never-seen {int(never_seen.sum())}); true Incident+high: {int(true_hi_inc.sum())}; "
          f"current model bar = {art['priority_threshold']}\n")
    print("subset=never_seen (the honest one).  esc = escalation rate | recall = escalated recall of true "
          "Incident+high\nauto = share of routine tickets auto-drafted | qacc = queue accuracy of auto-drafts\n")
    print(f"{'bar':>5} {'T1':>5} | {'esc':>6} {'recall':>7} {'auto':>6} {'qacc':>6} | high-Inc rule fires on routine")
    for r in rows:
        if r["subset"] != "never_seen":
            continue
        mark = "  <- current" if (r["priority_bar"] == 0.30 and r["T1"] == 0.5) else ""
        print(f"{r['priority_bar']:5.2f} {r['T1']:5.2f} | {r['escalation_rate']:6.3f} "
              f"{r['escalated_recall_hp_incident']:7.3f} {r['auto_draft_rate_routine']:6.3f} "
              f"{r['queue_acc_of_auto_drafts'] or 0:6.3f} | {r['share_predicted_high_among_routine']:.3f}{mark}")
    print(f"\nfull table (both subsets): {out}")


if __name__ == "__main__":
    main()
