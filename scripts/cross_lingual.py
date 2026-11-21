"""Owner: S3. Step 12: does cross-lingual precedent matching work (bonus feature)?

Usage: python scripts/cross_lingual.py   -> results/cross_lingual.json (results/ is gitignored)

Leakage-safe: KB part="val" queries against the part="train" index; holdout never touched.
Languages (queries and index) are DETECTED with src.triage, not the dataset label (wrong on ~11%).
Val tickets with an exact body copy in train are excluded (they would match themselves).

Three questions:
  A. Language mix: how often are top-1 / top-5 hits in the other language?
  B. Quality by language: for German queries, how good are the best ENGLISH precedents compared
     with the best GERMAN ones (and vice versa)? Measured by top-1 / any-of-top-5 queue and type
     match. If "other language only" is close to "same language only", cross-lingual matching works.
  C. Translation pairs: the same ticket written in English and German. Do both versions retrieve
     the same precedents (overlap of top-5), and how similar are the two query vectors?
  D. The fix: retrieve() reserves CROSS_LINGUAL_SLOTS of the k hits for the under-represented
     language. Compares xl_slots = 0 / 1 / 2 on the same val queries, and lists the mixed hits for
     the German pair queries on the FINAL (part="all") index as demo candidates.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402

from src.config import RESULTS_DIR  # noqa: E402
from src.data import load_kb_clean  # noqa: E402
from src.retrieval import _load, _norm_body, _top_k, embed_texts, retrieve_batch  # noqa: E402
from src.text import build_text  # noqa: E402
from src.triage import detect_language, language_text  # noqa: E402

PAIRS = [  # (subject_en, body_en, subject_de, body_de) written for this check
    ("VPN down", "Our site-to-site VPN is down since this morning. Remote staff cannot reach the hospital EMR system.",
     "VPN ausgefallen", "Unser Site-to-Site-VPN ist seit heute Morgen ausgefallen. Mitarbeiter im Homeoffice erreichen das Klinik-EMR-System nicht."),
    ("Password reset", "How do I reset my password? The reset link in the email has expired.",
     "Passwort zurücksetzen", "Wie setze ich mein Passwort zurück? Der Link in der E-Mail ist abgelaufen."),
    ("Charged twice", "I was charged twice for my subscription this month, please check my invoice.",
     "Doppelt belastet", "Ich wurde diesen Monat doppelt für mein Abonnement belastet, bitte prüfen Sie meine Rechnung."),
    ("Printer offline", "Our network printer shows offline since today and nobody can print.",
     "Drucker offline", "Unser Netzwerkdrucker wird seit heute als offline angezeigt und niemand kann drucken."),
    ("App crashes", "After the latest software update the application crashes on startup.",
     "App stürzt ab", "Nach dem letzten Software-Update stürzt die Anwendung beim Start ab."),
    ("Data export", "How can I export all our project data to CSV?",
     "Datenexport", "Wie kann ich alle unsere Projektdaten als CSV exportieren?"),
    ("Upgrade plan", "We want to upgrade from the basic plan to the premium plan. What are the steps?",
     "Tarif upgraden", "Wir möchten vom Basis-Tarif auf den Premium-Tarif wechseln. Was sind die Schritte?"),
    ("Data breach", "We suspect a data breach: unknown logins to our admin accounts from abroad.",
     "Datenleck", "Wir vermuten ein Datenleck: unbekannte Anmeldungen an unseren Admin-Konten aus dem Ausland."),
    ("Slow website", "Our e-commerce website has become very slow and pages time out during checkout.",
     "Langsame Website", "Unsere E-Commerce-Website ist sehr langsam geworden und Seiten brechen beim Checkout ab."),
    ("CRM integration", "Is there an API to integrate your analytics platform with our CRM?",
     "CRM-Integration", "Gibt es eine API, um Ihre Analyseplattform mit unserem CRM zu integrieren?"),
]


def main() -> None:
    vecs, recs = _load("train")
    r_lang = np.array([r["language"] for r in recs])
    r_queue = np.array([r["queue"] for r in recs])
    r_type = np.array([r["type"] for r in recs])

    val = load_kb_clean(part="val")
    train_ids = {r["ticket_id"] for r in recs}
    kb_train = load_kb_clean(need_answer=True, part="train")
    train_bodies = set(_norm_body(kb_train[kb_train.ticket_id.isin(train_ids)]["body"]))
    val = val[~_norm_body(val["body"]).isin(train_bodies).to_numpy()].reset_index(drop=True)
    q = embed_texts([build_text(s, b) for s, b in zip(val["subject"], val["body"])])
    # Query language = DETECTED language (the dataset label is wrong on ~11% of tickets).
    v_lang = np.array([detect_language(language_text(s, b)) for s, b in zip(val["subject"], val["body"])])
    v_queue, v_type = val["queue"].to_numpy(), val["type"].to_numpy()

    def search(mask):
        sub = np.flatnonzero(mask)
        s, i = _top_k(vecs[sub], q, 5)
        return s, sub[i]

    res = {"n_queries": int(len(val)), "index": "train", "A_language_mix": {}, "B_quality": {}}
    s_all, i_all = search(np.ones(len(recs), bool))
    for lang in ("en", "de"):
        m = v_lang == lang
        other = r_lang[i_all[m]] != lang
        res["A_language_mix"][lang] = {
            "n": int(m.sum()),
            "top1_other_language": round(float(other[:, 0].mean()), 4),
            "top5_share_other_language": round(float(other.mean()), 4),
            "queries_with_any_other_language_in_top5": round(float(other.any(1).mean()), 4)}

    for lang in ("en", "de"):
        m = v_lang == lang
        out = {}
        for name, mask in (("all_languages", np.ones(len(recs), bool)),
                           ("same_language_only", r_lang == lang),
                           ("other_language_only", r_lang != lang)):
            s, i = search(mask)
            s, i = s[m], i[m]
            out[name] = {"top1_queue": round(float((r_queue[i[:, 0]] == v_queue[m]).mean()), 4),
                         "any5_queue": round(float((r_queue[i] == v_queue[m][:, None]).any(1).mean()), 4),
                         "top1_type": round(float((r_type[i[:, 0]] == v_type[m]).mean()), 4),
                         "top1_score_p50": round(float(np.median(s[:, 0])), 4)}
        res["B_quality"][lang] = out

    pairs = []
    en_q = embed_texts([build_text(a, b) for a, b, _, _ in PAIRS])
    de_q = embed_texts([build_text(c, d) for _, _, c, d in PAIRS])
    se, ie = _top_k(vecs, en_q, 5)
    sd, idd = _top_k(vecs, de_q, 5)
    for n, (a, _, c, _) in enumerate(PAIRS):
        ids_en, ids_de = [recs[j]["ticket_id"] for j in ie[n]], [recs[j]["ticket_id"] for j in idd[n]]
        de_en_only = _top_k(vecs[r_lang == "en"], de_q[n:n + 1], 1)
        best_en_for_de = recs[np.flatnonzero(r_lang == "en")[de_en_only[1][0, 0]]]
        pairs.append({
            "en": a, "de": c,
            "query_cosine_en_de": round(float(en_q[n] @ de_q[n]), 4),
            "top5_overlap": len(set(ids_en) & set(ids_de)),
            "top5_languages_en_query": "".join("E" if r_lang[j] == "en" else "D" for j in ie[n]),
            "top5_languages_de_query": "".join("E" if r_lang[j] == "en" else "D" for j in idd[n]),
            "de_query_top1": f"{sd[n, 0]:.3f} {recs[idd[n, 0]]['ticket_id']} [{r_lang[idd[n, 0]]}] {recs[idd[n, 0]]['subject'][:60]}",
            "de_query_best_english": f"{de_en_only[0][0, 0]:.3f} {best_en_for_de['ticket_id']} {best_en_for_de['subject'][:60]}",
        })
    res["C_translation_pairs"] = pairs
    res["C_summary"] = {
        "mean_query_cosine_en_de": round(float(np.mean([p["query_cosine_en_de"] for p in pairs])), 4),
        "mean_top5_overlap": round(float(np.mean([p["top5_overlap"] for p in pairs])), 2)}

    # D. Reserved cross-lingual slots (what retrieve() now does by default)
    texts = [build_text(s, b) for s, b in zip(val["subject"], val["body"])]
    res["D_reserved_slots"] = {}
    for r in (0, 1, 2):
        hits = retrieve_batch(texts, k=5, part="train", xl_slots=r)
        row = {}
        for lang in ("en", "de"):
            m = np.flatnonzero(v_lang == lang)
            hs = [hits[i] for i in m]
            row[lang] = {
                "top1_queue": round(float(np.mean([h[0]["queue"] == v_queue[i] for h, i in zip(hs, m)])), 4),
                "any5_queue": round(float(np.mean([any(x["queue"] == v_queue[i] for x in h) for h, i in zip(hs, m)])), 4),
                "any5_type": round(float(np.mean([any(x["type"] == v_type[i] for x in h) for h, i in zip(hs, m)])), 4),
                "queries_with_other_language_hit": round(float(np.mean([any(x["language"] != lang for x in h) for h in hs])), 4)}
        res["D_reserved_slots"][f"xl_slots={r}"] = row

    demo = retrieve_batch([build_text(c, d) for _, _, c, d in PAIRS], k=5, part="all")
    res["D_demo_german_queries_final_index"] = {
        c: [f"{h['score']:.3f} {h['ticket_id']} [{h['language']}] {h['queue']} | {h['subject'][:60]}" for h in hs]
        for (_, _, c, _), hs in zip(PAIRS, demo)}

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "cross_lingual.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("A_language_mix", "B_quality", "C_summary", "D_reserved_slots",
                                          "D_demo_german_queries_final_index")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
