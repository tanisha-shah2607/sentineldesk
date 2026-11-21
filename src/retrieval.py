"""Owner: S3. Multilingual dense retrieval over the historical KB (exact cosine search).

Search is exact inner product on L2-normalised e5 vectors (same results as faiss.IndexFlatIP),
done in NumPy. FAISS is not used: on macOS its wheel and torch's wheel each bundle libomp, and
loading both aborts the process ("OMP: Error #15"). At ~33k vectors exact NumPy search is
a few ms per query, so nothing is lost.

retrieve(text, k=5, part="all", xl_slots=None) -> [{
  "ticket_id": "TKT-00012", "score": float (cosine; e5 scores sit in a narrow high band, ~0.75-0.95),
  "subject": str, "answer": str, "snippet": str (answer[:300]),
  "queue": str, "type": str, "language": str ("en"|"de", DETECTED by src.triage, not the dataset label)
}]  sorted by score desc; [] for empty text
retrieve_batch(texts, k, part="all", xl_slots=None) -> [list_of_hits, ...]   (evaluate.py runs this on the full holdout)

RULES:
  - Index load_kb_clean(need_answer=True) only (never anything from the holdout).
  - Two index builds: part="train" (for tuning SIM_THRESHOLD on part="val" queries, so a val
    ticket never retrieves itself) and part="all" (the final index used by the app).
  - Duplicate bodies are collapsed to one ticket (lowest ticket_id) so top-k hits are distinct.
  - Cross-lingual mix (bonus): e5 gives same-language tickets a small score bonus, so a German
    ticket usually gets 5 German hits even when English precedents are as good. With k > slots,
    the top (k - CROSS_LINGUAL_SLOTS) hits are the best overall and the remaining slots go to the
    best tickets in whichever language is under-represented among them. Hits stay sorted by their
    real cosine score, so hits[0] (what the gate's SIM_THRESHOLD uses) never changes.
    On val: German any-of-top-5 queue match 82.1% -> 86.3%, English 88.0% -> 89.6%
    (scripts/cross_lingual.py). Set CROSS_LINGUAL_SLOTS=0 (env) or xl_slots=0 to disable.
  - e5 query/passage prefixes live here; the input text comes from src.text.build_text.

Build: python scripts/build_index.py   (embeds once into index/, then builds both indexes)
Tests/UI without an index: set RETRIEVAL_STUB=1 to get the fixed stub hit (conftest.py does this
automatically when index/ has not been built).
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

import numpy as np

from .config import INDEX_DIR, TOP_K
from . import text as _text_mod
from .text import build_text, unescape

MODEL_NAME = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
QUERY_PREFIX, PASSAGE_PREFIX = "query: ", "passage: "   # required by e5 models
BATCH_SIZE = 64
SEARCH_CHUNK = 512        # queries per matrix multiply in retrieve_batch (caps memory at ~70 MB)
CROSS_LINGUAL_SLOTS = int(os.getenv("CROSS_LINGUAL_SLOTS", "2"))   # of k hits, reserved for the other language
SNIPPET_CHARS = 300
PARTS = ("all", "train")

EMB_PATH = INDEX_DIR / "kb_emb.npy"          # float32 (n, d), L2-normalised, row-aligned with META_PATH
META_PATH = INDEX_DIR / "kb_meta.parquet"    # ticket_id, subject, answer, queue, type, language
MODEL_TXT = INDEX_DIR / "model.txt"          # cache key (model + text-builder version) of EMB_PATH
META_COLS = ["ticket_id", "subject", "answer", "queue", "type", "language"]


def _vec_path(part: str):
    return INDEX_DIR / f"vecs_{part}.npy"


def _meta_path(part: str):
    return INDEX_DIR / f"meta_{part}.parquet"


# ---------------------------------------------------------------- embedding

def _device() -> str:
    """EMBED_DEVICE overrides; otherwise cuda > mps (Apple GPU) > cpu (e.g. Cloud Run)."""
    if os.getenv("EMBED_DEVICE"):
        return os.environ["EMBED_DEVICE"]
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME, device=_device())


def embed_texts(texts: list[str], is_query: bool = True, show_progress: bool = False) -> np.ndarray:
    """L2-normalised float32 embeddings. Shared with S2 (classifier) and S5 (burst.py).
    Pass texts built with build_text(); the e5 prefix is added here."""
    prefix = QUERY_PREFIX if is_query else PASSAGE_PREFIX
    return _model().encode([prefix + t for t in texts], batch_size=BATCH_SIZE,
                           normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=show_progress).astype(np.float32)


@lru_cache(maxsize=1)
def _cache_key() -> str:
    """Model name + hash of src/text.py. Any change to build_text()/unescape() changes the key, so a
    stale embedding cache is detected instead of silently mismatching query-time text."""
    import hashlib
    from pathlib import Path
    src = Path(_text_mod.__file__).read_bytes()
    return f"{MODEL_NAME}|text:{hashlib.sha256(src).hexdigest()[:12]}"


def _stored_key() -> str | None:
    return MODEL_TXT.read_text().strip() if MODEL_TXT.exists() else None


def index_ready() -> bool:
    """True if the embedding cache exists for this model AND this build_text() version
    (the search index can then be built in seconds)."""
    return EMB_PATH.exists() and META_PATH.exists() and _stored_key() == _cache_key()


def _norm_body(body: "pd.Series") -> "pd.Series":  # noqa: F821
    """Normalised body used to detect duplicate tickets (unescaped, whitespace-collapsed, lowercase)."""
    return body.map(unescape).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()


def embed_kb(force: bool = False) -> None:
    """Step 4: embed every clean KB row with an answer ONCE and cache to INDEX_DIR.
    Keyed by ticket_id, so the train/all indexes are just subsets (no re-embedding)."""
    if not force and index_ready():
        print(f"Embedding cache already exists ({EMB_PATH}); use --force to rebuild.")
        return
    from .data import load_kb_clean   # data access stays in src/data.py
    kb = load_kb_clean(need_answer=True, part="all")
    texts = [build_text(s, b) for s, b in zip(kb["subject"], kb["body"])]
    print(f"Embedding {len(texts)} KB tickets with {MODEL_NAME} on {_device()} ...")
    emb = embed_texts(texts, is_query=False, show_progress=True)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_PATH, emb)
    kb[META_COLS].reset_index(drop=True).to_parquet(META_PATH, index=False)
    MODEL_TXT.write_text(_cache_key() + "\n")
    print(f"Saved {emb.shape} -> {EMB_PATH}\nSaved metadata -> {META_PATH}")


def load_kb_embeddings() -> tuple[np.ndarray, "pd.DataFrame"]:  # noqa: F821
    """(embeddings, metadata) from the cache, row-aligned. For S3's index build and S2's classifier."""
    import pandas as pd
    if not index_ready():
        raise FileNotFoundError(f"No up-to-date embedding cache in {INDEX_DIR} (missing, or the model or "
                                "src/text.py changed). "
                                "Run: python scripts/build_index.py")
    return np.load(EMB_PATH), pd.read_parquet(META_PATH)


# ---------------------------------------------------------------- index

# --- Generic answers --------------------------------------------------------------------------
# Some KB "resolutions" contain no resolution: "call", "Sorry", "To better assist you, could you please
# provide more details about the issue you are facing?" (TKT-07586), "I will address the issue and call
# you back at <tel_num> shortly" (TKT-33547). Their vague text sits close to many vague queries, so they
# end up as top hits and the reply quotes nothing useful. They are left out of the index (~6-7% of it).
MIN_ANSWER_CHARS = 80        # below this, no answer in the KB holds an actual resolution
GENERIC_MAX_CHARS = 150      # short answers that only ask for details / promise contact are generic too
_GENERIC_ANSWER = re.compile(
    r"\b((provide|share|send|give|add)( us)? (more|additional|further) (details|information|info)|"
    r"(more|further|additional) (details|information)|could you (please )?(clarify|describe|explain)|"
    r"call you( back)?|contact (us|you)|reach (out|us)|get back to you|get in touch|follow up|"
    r"will (address|look into|investigate|review|check)|"
    r"(weitere|mehr|zusätzliche|genauere|nähere) (details|informationen|angaben)|"
    r"(können|könnten) sie .{0,40}(beschreiben|erläutern|mitteilen|angeben)|"
    r"(rufen|melden) (wir|sie|uns)|rückruf|kontaktieren sie|kontaktieren uns|anrufen|"
    r"wird (geprüft|bearbeitet|untersucht)|werden .{0,30}(prüfen|untersuchen))|<tel_num>", re.IGNORECASE)


def is_generic_answer(answer) -> bool:
    """True if a KB answer holds no actual resolution (too short, or short and only asks for details /
    promises a call). Such tickets are excluded from the index."""
    text = unescape(answer).strip()
    return len(text) < MIN_ANSWER_CHARS or (len(text) < GENERIC_MAX_CHARS and bool(_GENERIC_ANSWER.search(text)))


def _dedupe_positions(meta) -> np.ndarray:
    """Row positions to keep: one ticket per normalised body (`_body`), the lowest ticket_id.
    `meta` must have a 0..n-1 RangeIndex so index labels are row positions."""
    first = meta.sort_values("ticket_id").drop_duplicates("_body")
    return np.sort(first.index.to_numpy())


def build_index(part: str = "all") -> dict:
    """Steps 5-6: select cached rows for `part`, drop duplicate bodies, and persist the
    index (normalised vectors + row-aligned metadata) for exact cosine search."""
    from .data import load_kb_clean   # data access stays in src/data.py
    assert part in PARTS, part
    emb, meta = load_kb_embeddings()

    kb = load_kb_clean(need_answer=True, part=part)[["ticket_id", "body"]]
    sel = meta["ticket_id"].isin(set(kb["ticket_id"])).to_numpy()
    emb, meta = emb[sel], meta[sel].reset_index(drop=True)
    body = meta[["ticket_id"]].merge(kb, on="ticket_id", how="left")["body"]
    meta["_body"] = _norm_body(body).to_numpy()
    meta["_raw_body"] = body.to_numpy()

    # Drop generic answers BEFORE de-duplicating, so a duplicate with a real answer can take its place.
    informative = ~meta["answer"].map(is_generic_answer).to_numpy()
    n_generic = int((~informative).sum())
    emb, meta = emb[informative], meta[informative].reset_index(drop=True)

    keep = _dedupe_positions(meta)
    emb = np.ascontiguousarray(emb[keep])
    meta = meta.iloc[keep].reset_index(drop=True)
    # `language` = detected language (S2's detector), not the dataset label: the label is wrong on
    # ~11% of KB tickets (26% of those labelled "de" are written in English). Used for the hits'
    # `language` field and for the cross-lingual mix. ~9 s for the full index.
    from .triage import detect_language, language_text
    meta["language"] = [detect_language(language_text(s, b))
                        for s, b in zip(meta["subject"], meta["_raw_body"])]
    meta = meta.drop(columns=["_body", "_raw_body"])

    np.save(_vec_path(part), emb)
    meta.to_parquet(_meta_path(part), index=False)
    _load.cache_clear()
    _lang_positions.cache_clear()
    stats = {"part": part, "selected": int(sel.sum()), "indexed": int(len(emb)),
             "generic_answers_dropped": n_generic,
             "duplicates_dropped": int(sel.sum() - n_generic - len(emb))}
    print(f"Built {_vec_path(part).name}: {stats}")
    return stats


@lru_cache(maxsize=len(PARTS))
def _load(part: str):
    """Load (vectors, metadata records) once per process. Builds the index from the embedding
    cache if it is missing (seconds); never embeds the KB inside a request."""
    import pandas as pd
    assert part in PARTS, part
    if not index_ready():
        raise FileNotFoundError(f"No up-to-date embedding cache in {INDEX_DIR} (missing, or the model or "
                                "src/text.py changed). "
                                "Run: python scripts/build_index.py")
    if not (_vec_path(part).exists() and _meta_path(part).exists()):
        build_index(part)
    vecs = np.load(_vec_path(part))
    meta = pd.read_parquet(_meta_path(part))
    for c in META_COLS:                       # contract: plain str, never NaN/None (8.7% null subjects)
        meta[c] = meta[c].fillna("").astype(str)
    for c in ("subject", "answer"):           # literal "\n" escapes -> real line breaks (see text.unescape)
        meta[c] = meta[c].map(unescape)
    meta["snippet"] = meta["answer"].str[:SNIPPET_CHARS]
    return vecs, meta[META_COLS + ["snippet"]].to_dict("records")


def _top_k(vecs: np.ndarray, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k by inner product for each row of q. Returns (scores, ids), both (len(q), k)."""
    k = min(k, len(vecs))
    all_s, all_i = [], []
    for start in range(0, len(q), SEARCH_CHUNK):
        s = q[start:start + SEARCH_CHUNK] @ vecs.T
        i = np.argpartition(-s, k - 1, axis=1)[:, :k]
        top = np.take_along_axis(s, i, axis=1)
        order = np.argsort(-top, axis=1, kind="stable")
        all_i.append(np.take_along_axis(i, order, axis=1))
        all_s.append(np.take_along_axis(top, order, axis=1))
    return np.vstack(all_s), np.vstack(all_i)


@lru_cache(maxsize=len(PARTS))
def _lang_positions(part: str) -> dict[str, np.ndarray]:
    """Row positions of each language in the index for `part`."""
    langs = np.array([r["language"] for r in _load(part)[1]])
    return {lang: np.flatnonzero(langs == lang) for lang in sorted(set(langs))}


def _search(part: str, q: np.ndarray, k: int, slots: int) -> list[list[tuple[float, int]]]:
    """Per query: [(score, row), ...] sorted by score desc. Applies the cross-lingual mix."""
    vecs, records = _load(part)
    S, I = _top_k(vecs, q, k)
    if slots <= 0 or k <= slots:
        return [list(zip(S[n].tolist(), I[n].tolist())) for n in range(len(q))]
    pos = _lang_positions(part)
    per_lang = {}
    for lang, p in pos.items():
        s, i = _top_k(vecs[p], q, min(k, len(p)))
        per_lang[lang] = (s, p[i])
    rows = []
    for n in range(len(q)):
        base = list(zip(S[n, :k - slots].tolist(), I[n, :k - slots].tolist()))
        taken = {j for _, j in base}
        counts = {lang: 0 for lang in pos}
        for _, j in base:
            counts[records[j]["language"]] += 1
        other = min(pos, key=lambda lang: (counts[lang], lang != "en"))   # under-represented language
        s_o, i_o = per_lang[other]
        extra = [(float(s), int(j)) for s, j in zip(s_o[n], i_o[n]) if int(j) not in taken][:slots]
        if len(extra) < slots:                                             # tiny index: fill from overall
            extra += [(float(s), int(j)) for s, j in zip(S[n], I[n])
                      if int(j) not in taken and int(j) not in {e[1] for e in extra}][:slots - len(extra)]
        rows.append(sorted(base + extra, key=lambda t: -t[0]))
    return rows


def warmup(part: str = "all") -> None:
    """Load model + index and run one query so the first real request is fast (call at app start)."""
    retrieve("warmup", k=1, part=part)


# ---------------------------------------------------------------- search

_STUB_HIT = {
    "ticket_id": "TKT-00000", "score": 0.0, "subject": "(stub)",
    "answer": "(stub answer)", "snippet": "(stub answer)", "queue": "Technical Support",
    "type": "Incident", "language": "en",
}


def _stub_mode() -> bool:
    return os.getenv("RETRIEVAL_STUB") == "1"


def retrieve_batch(texts: list[str], k: int = TOP_K, part: str = "all",
                   xl_slots: int | None = None) -> list[list[dict]]:
    if _stub_mode():
        return [[dict(_STUB_HIT)][:k] if t and t.strip() else [] for t in texts]
    out: list[list[dict]] = [[] for _ in texts]
    live = [i for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
    if not live or k <= 0:
        return out
    records = _load(part)[1]
    q = embed_texts([texts[i] for i in live], is_query=True)
    slots = CROSS_LINGUAL_SLOTS if xl_slots is None else xl_slots
    for row, i in zip(_search(part, q, k, slots), live):
        out[i] = [{**records[j], "score": s} for s, j in row]
    return out


def retrieve(text: str, k: int = TOP_K, part: str = "all", xl_slots: int | None = None) -> list[dict]:
    return retrieve_batch([text], k=k, part=part, xl_slots=xl_slots)[0]
