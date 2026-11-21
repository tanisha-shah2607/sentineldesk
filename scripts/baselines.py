"""Owner: S1. Floor numbers to put S2's and S3's results in context.  Usage: python scripts/baselines.py

  majority   : always predict the most common class (what "no model" scores)
  tfidf_svm  : plain TF-IDF + linear SVM, no calibration, no outage rule (what a 5-minute model scores)
  tfidf_retr : TF-IDF cosine retrieval over the clean KB (what keyword search scores), same
               top1/any-5 queue-match metrics as evaluate.py

Trained/indexed on load_kb_clean() only; scored on the clean holdout, all and never-seen.
Holdout labels are loaded for scoring only (same rule as evaluate.py).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from src.config import RESULTS_DIR, TARGETS  # noqa: E402
from src.data import (holdout_seen_ids, load_kb_clean, load_test_inputs,  # noqa: E402
                      load_test_labels, norm_body)
from src.text import build_text  # noqa: E402


def f1(y, p, sel):
    sel = np.asarray(sel)
    return float(f1_score(np.asarray(y)[sel], np.asarray(p)[sel], average="macro", zero_division=0))


def main():
    kb = load_kb_clean(need_answer=True)
    lab = load_test_labels()
    te = load_test_inputs().set_index("ticket_id").loc[lab.ticket_id]
    lab = lab.set_index("ticket_id")
    seen_ids = holdout_seen_ids()
    seen = np.array([t in seen_ids for t in lab.index])
    subsets = {"all": np.ones(len(lab), bool), "never_seen": ~seen}

    kb_text = [build_text(s, b) for s, b in zip(kb["subject"], kb["body"])]
    te_text = [build_text(s, b) for s, b in zip(te["subject"], te["body"])]
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=200_000)
    Xkb, Xte = vec.fit_transform(kb_text), vec.transform(te_text)

    preds = {"majority": {}, "tfidf_svm": {}}
    for t in TARGETS:
        preds["majority"][t] = np.array([kb[t].mode()[0]] * len(lab))
        preds["tfidf_svm"][t] = LinearSVC(C=0.5, class_weight="balanced").fit(Xkb, kb[t]).predict(Xte)

    true_hi = ((lab["type"] == "Incident") & (lab["priority"] == "high")).values
    res = {"n_train": len(kb), "n_eval": len(lab), "n_never_seen": int((~seen).sum())}
    for name, p in preds.items():
        pred_hi = (p["type"] == "Incident") & (p["priority"] == "high")
        for sub, sel in subsets.items():
            for t in TARGETS:
                res[f"{name}_{sub}_macro_f1_{t}"] = f1(lab[t].values, p[t], sel)
            hi = true_hi & sel
            res[f"{name}_{sub}_hp_incident_recall"] = float(pred_hi[hi].mean()) if hi.any() else None

    # TF-IDF retrieval: kb deduped by normalised body (same as S3's index), top-5 cosine
    keep = ~norm_body(kb["body"]).duplicated().values
    Xr, kbq, kbt = Xkb[keep], kb["queue"].values[keep], kb["type"].values[keep]
    q_true, t_true = lab["queue"].values, lab["type"].values
    top1_q, anyk_q, anyk_t = [], [], []
    for i in range(0, Xte.shape[0], 512):
        sims = (Xte[i:i + 512] @ Xr.T).toarray()
        top = np.argsort(-sims, axis=1)[:, :5]
        for j, row in enumerate(top):
            top1_q.append(kbq[row[0]] == q_true[i + j])
            anyk_q.append(any(kbq[r] == q_true[i + j] for r in row))
            anyk_t.append(any(kbt[r] == t_true[i + j] for r in row))
    for sub, sel in subsets.items():
        res[f"tfidf_retr_{sub}_top1_queue_match"] = float(np.array(top1_q)[sel].mean())
        res[f"tfidf_retr_{sub}_anyk_queue_match"] = float(np.array(anyk_q)[sel].mean())
        res[f"tfidf_retr_{sub}_anyk_type_match"] = float(np.array(anyk_t)[sel].mean())

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "baselines.json").write_text(json.dumps(res, indent=2))

    print(f"train n={res['n_train']}  eval n={res['n_eval']} (never-seen {res['n_never_seen']})\n")
    print(f"{'':<22}{'all':>26}{'never-seen':>26}")
    print(f"{'model / metric':<22}" + "   type  queue  prio   HPinc" * 2)
    for name in preds:
        row = f"{name:<22}"
        for sub in subsets:
            v = [res[f"{name}_{sub}_macro_f1_{t}"] for t in TARGETS] + [res[f"{name}_{sub}_hp_incident_recall"]]
            row += "".join(f"{x:7.3f}" for x in v) + "   "
        print(row)
    print("\nTF-IDF retrieval (queue/type match of top-5 hits):")
    for sub in subsets:
        print(f"  {sub:<11} top1 queue {res[f'tfidf_retr_{sub}_top1_queue_match']:.3f} | "
              f"any-5 queue {res[f'tfidf_retr_{sub}_anyk_queue_match']:.3f} | "
              f"any-5 type {res[f'tfidf_retr_{sub}_anyk_type_match']:.3f}")


if __name__ == "__main__":
    main()
