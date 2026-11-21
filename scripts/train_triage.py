"""Train the triage classifier (owner: S2). One command: python scripts/train_triage.py

Reads data ONLY through src.data.load_kb_clean(part="train" / part="val"); never opens
the holdout files. It:
  - fits the shared TF-IDF and the type / queue / priority heads on part="train"
  - calibrates each head (temperature scaling) and tunes the priority "high" bar on part="val",
    with the outage rule applied, so both share one Macro-F1 budget
  - prints validation metrics (all of val, and val tickets whose body never appears in train)
    and writes models/triage.joblib + results/triage_val_metrics.json
Optional: --C queue=64 overrides one head's regularisation for an experiment.
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from src import config as C  # noqa: E402
from src.data import load_kb_clean  # noqa: E402
from src.text import build_text, unescape  # noqa: E402
from src.triage_model import (HEADS, HIGH, INCIDENT, OUTAGE_PATTERN,  # noqa: E402
                              expected_calibration_error, final_labels, fit_temperature, logits,
                              make_head, make_vectorizer, outage_terms, softmax_t)

# Regularisation per head, picked by validation Macro-F1 (word-only features,
# C in {1, 4, 16, 32, 64, 128}; scores rise steeply up to 16, then flatten).
BEST_C = {"type": 32.0, "queue": 16.0, "priority": 16.0}
MAX_F1_DROP = 0.02   # priority Macro-F1 we accept losing (bar + outage rule) to catch more incidents
MODEL_PATH = C.MODEL_DIR / "triage.joblib"
METRICS_PATH = C.RESULTS_DIR / "triage_val_metrics.json"


def texts(df) -> list[str]:
    return [build_text(s, b) for s, b in zip(df.subject, df.body)]


def norm_body(body) -> str:
    """Same normalisation S3 uses to find duplicate tickets."""
    return re.sub(r"\s+", " ", unescape(body)).strip().lower()


def cross_fitted(scores: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    """Calibrate each half of val with a temperature fitted on the other half, so the
    reported calibration error isn't measured on the rows the temperature was fitted to."""
    half = np.arange(len(y_idx)) % 2 == 0
    out = np.empty_like(scores)
    out[half] = softmax_t(scores[half], fit_temperature(scores[~half], y_idx[~half]))
    out[~half] = softmax_t(scores[~half], fit_temperature(scores[half], y_idx[half]))
    return out


def hpi_recall(y_type, y_prio, p_type, p_prio, also_caught=None) -> float:
    """evaluate.py's definition: of true high + Incident tickets, the share predicted high + Incident."""
    true = (y_type == INCIDENT) & (y_prio == HIGH)
    caught = (p_type == INCIDENT) & (p_prio == HIGH)
    if also_caught is not None:
        caught = caught | also_caught
    return float(caught[true].mean())


def parse_c_overrides(pairs: list[str]) -> dict:
    best_c = dict(BEST_C)
    for pair in pairs:
        name, _, value = pair.partition("=")
        if name not in best_c or not value:
            raise SystemExit(f"--C expects head=value with head in {list(best_c)}, got {pair!r}")
        best_c[name] = float(value)
    return best_c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--C", action="append", default=[], metavar="HEAD=VALUE",
                    help="override a head's regularisation, e.g. --C queue=64")
    best_c = parse_c_overrides(ap.parse_args().C)
    t_start = time.perf_counter()
    train, val = load_kb_clean(part="train"), load_kb_clean(part="val")
    train_texts, val_texts = texts(train), texts(val)
    never_seen = ~val["body"].map(norm_body).isin(set(train["body"].map(norm_body))).to_numpy()
    print(f"Training on {len(train):,} tickets, validating on {len(val):,} (shared slice; "
          f"{never_seen.sum():,} of them never appear in train)", flush=True)

    t0 = time.perf_counter()
    vec = make_vectorizer().fit(train_texts)
    Xt, Xv = vec.transform(train_texts), vec.transform(val_texts)
    for _, v in vec.transformer_list:
        v.stop_words_ = None   # sklearn keeps every pruned n-gram here; not needed, huge on disk
    print(f"  features: {Xt.shape[1]:,} columns ({time.perf_counter() - t0:.0f}s)", flush=True)

    heads, proba, classes, report = {}, {}, {}, {}
    for name in HEADS:
        t0 = time.perf_counter()
        clf = make_head(best_c[name]).fit(Xt, train[name])
        y_idx = np.searchsorted(clf.classes_, val[name].to_numpy())
        scores = logits(clf, Xv)
        temperature = fit_temperature(scores, y_idx)
        raw = softmax_t(scores)
        pred = clf.classes_[raw.argmax(axis=1)]
        heads[name] = {"clf": clf, "temperature": temperature}
        proba[name], classes[name] = softmax_t(scores, temperature), clf.classes_
        per_class = f1_score(val[name], pred, average=None, labels=clf.classes_)
        report[name] = {
            "C": best_c[name],
            "macro_f1": float(f1_score(val[name], pred, average="macro")),
            "accuracy": float(accuracy_score(val[name], pred)),
            "temperature": temperature,
            "calibration_error_before": expected_calibration_error(raw, y_idx),
            "calibration_error_after": expected_calibration_error(cross_fitted(scores, y_idx), y_idx),
            "per_class_f1": {str(k): float(v) for k, v in zip(clf.classes_, per_class)},
        }
        print(f"  {name} head trained ({time.perf_counter() - t0:.0f}s)", flush=True)

    y = {name: val[name].to_numpy() for name in HEADS}
    outage = np.array([bool(outage_terms(t)) for t in val_texts], dtype=bool)

    # Model alone (most likely label everywhere, no rule) is the baseline for the F1 budget.
    alone = {name: np.asarray(classes[name][proba[name].argmax(axis=1)], dtype=object) for name in HEADS}
    base_f1 = f1_score(y["priority"], alone["priority"], average="macro")
    base_recall = hpi_recall(y["type"], y["priority"], alone["type"], alone["priority"])

    # Priority bar: the lowest P(high) threshold that, together with the outage rule, keeps
    # priority Macro-F1 within MAX_F1_DROP of the model alone.
    sweep = []
    for t in np.round(np.arange(0.05, 0.601, 0.01), 2):
        lab, _ = final_labels(proba, classes, float(t), outage)
        sweep.append((float(t), float(f1_score(y["priority"], lab["priority"], average="macro")),
                      hpi_recall(y["type"], y["priority"], lab["type"], lab["priority"])))
    feasible = [row for row in sweep if row[1] >= base_f1 - MAX_F1_DROP] or [max(sweep, key=lambda r: r[1])]
    threshold = min(feasible, key=lambda r: r[0])[0]

    final, conf = final_labels(proba, classes, threshold, outage)
    overall = np.minimum(conf["type"], conf["queue"])
    low_conf = overall < C.CONF_THRESHOLD
    routing_ok = (final["type"] == y["type"]) & (final["queue"] == y["queue"])
    already_hpi = (alone["type"] == INCIDENT) & (alone["priority"] == HIGH)
    true_hpi = (y["type"] == INCIDENT) & (y["priority"] == HIGH)

    def summary(m: np.ndarray) -> dict:
        return {
            "tickets": int(m.sum()),
            "macro_f1": {name: float(f1_score(y[name][m], final[name][m], average="macro")) for name in HEADS},
            "hpi_recall_model_alone": hpi_recall(y["type"][m], y["priority"][m],
                                                 alone["type"][m], alone["priority"][m]),
            "hpi_recall": hpi_recall(y["type"][m], y["priority"][m], final["type"][m], final["priority"][m]),
            "hpi_recall_or_low_conf": hpi_recall(y["type"][m], y["priority"][m], final["type"][m],
                                                 final["priority"][m], also_caught=low_conf[m]),
            "share_below_conf_threshold": float(low_conf[m].mean()),
            "routing_correct_below_conf_threshold": float(routing_ok[m & low_conf].mean()) if (m & low_conf).any() else None,
        }

    results = {"all_val": summary(np.ones(len(val), bool)), "never_seen": summary(never_seen)}
    bands = []
    for lo, hi in [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]:
        m = (overall >= lo) & (overall < hi)
        bands.append({"overall_conf": f"{lo:.1f}-{min(hi, 1.0):.1f}", "share": float(m.mean()),
                      "routing_correct": float(routing_ok[m].mean()) if m.any() else None})
    rule = {"matched": int(outage.sum()),
            "newly_flagged": int((outage & ~already_hpi).sum()),
            "newly_flagged_wrongly": int((outage & ~already_hpi & ~true_hpi).sum())}

    artifact = {
        "format": 1,
        "vectorizer": vec,
        "heads": heads,
        "priority_threshold": threshold,
        "outage_rule": True,
        "meta": {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sklearn": sklearn.__version__,
            "n_train": len(train),
            "n_val": len(val),
            "best_c": best_c,
            "text_builder": "src.text.build_text",
            "outage_pattern": OUTAGE_PATTERN.pattern,
        },
    }
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH, compress=3)

    metrics = {
        "n_train": len(train), "n_val": len(val), "n_val_never_seen": int(never_seen.sum()),
        "heads_model_alone": report, "priority_threshold": threshold,
        "priority_macro_f1_model_alone": float(base_f1), "hpi_recall_model_alone": base_recall,
        "final": results, "outage_rule": rule, "overall_conf_bands": bands,
        "priority_threshold_sweep": [{"threshold": t, "macro_f1": f, "hpi_recall": r} for t, f, r in sweep],
    }
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"\nModel alone on validation ({len(val):,} tickets; C and temperature were tuned on them)")
    print(f"  {'head':<9}{'Macro-F1':>9}{'accuracy':>10}{'calibration error':>22}{'temperature':>13}")
    for name in HEADS:
        r = report[name]
        cal = f"{r['calibration_error_before']:.3f} -> {r['calibration_error_after']:.3f}"
        print(f"  {name:<9}{r['macro_f1']:>9.3f}{r['accuracy']:>10.3f}{cal:>22}{r['temperature']:>13.2f}")
    print(f"\nFinal labels, as triage() outputs them: priority bar P(high) >= {threshold:.2f} + outage rule")
    print(f"  outage rule matched {rule['matched']} val tickets; {rule['newly_flagged']} newly flagged, "
          f"{rule['newly_flagged_wrongly']} of them wrongly")
    a, n = results["all_val"], results["never_seen"]
    print(f"  {'':<38}{'all val':>9}{'never-seen':>12}")
    for name in HEADS:
        print(f"  {name + ' Macro-F1':<38}{a['macro_f1'][name]:>9.3f}{n['macro_f1'][name]:>12.3f}")
    print(f"  {'high-priority incident recall':<38}{a['hpi_recall']:>9.3f}{n['hpi_recall']:>12.3f}"
          f"   (model alone: {a['hpi_recall_model_alone']:.3f} / {n['hpi_recall_model_alone']:.3f})")
    print(f"  {'...or overall_conf < ' + format(C.CONF_THRESHOLD, '.2f'):<38}{a['hpi_recall_or_low_conf']:>9.3f}"
          f"{n['hpi_recall_or_low_conf']:>12.3f}   (the gate also escalates low confidence)")
    print(f"  {'share below T1, routing right there':<38}{a['share_below_conf_threshold']:>8.0%} "
          f"{n['share_below_conf_threshold']:>8.0%}    ({a['routing_correct_below_conf_threshold']:.0%} / "
          f"{n['routing_correct_below_conf_threshold']:.0%} right)")
    print(f"\n  overall_conf on all validation tickets (what the gate's T1 = {C.CONF_THRESHOLD:.2f} compares against)")
    print(f"    {'overall_conf':<14}{'tickets':>8}{'routing correct':>18}")
    for b in bands:
        rc = "-" if b["routing_correct"] is None else f"{b['routing_correct']:.0%}"
        print(f"    {b['overall_conf']:<14}{b['share']:>8.0%}{rc:>18}")
    weakest = sorted(report["queue"]["per_class_f1"].items(), key=lambda kv: kv[1])[:3]
    print("\n  weakest queues (F1, model alone): " + ", ".join(f"{k} {v:.2f}" for k, v in weakest))
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"\nSaved {MODEL_PATH.relative_to(C.ROOT).as_posix()} ({size_mb:.0f} MB) and "
          f"{METRICS_PATH.relative_to(C.ROOT).as_posix()} in {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
