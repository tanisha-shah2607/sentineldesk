"""S3 steps 1-2: compare embedding models on speed and a quick quality check.

Usage: python scripts/bench_embed.py
Quality = top-1 queue match of 200 val queries against 5,000 train docs. About a quarter of val
tickets have an exact copy in train, so the absolute numbers are inflated; use them only to compare models.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from src.data import load_kb_clean  # noqa: E402
from src.text import build_text  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODELS = {  # name: (query prefix, passage prefix)
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": ("", ""),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
}

train = load_kb_clean(need_answer=True, part="train").sample(5000, random_state=42)
val = load_kb_clean(need_answer=True, part="val").sample(200, random_state=42)
docs = [build_text(s, b) for s, b in zip(train.subject, train.body)]
qs = [build_text(s, b) for s, b in zip(val.subject, val.body)]

# Val queries with no exact body copy in the train sample: a fairer quality signal.
norm = lambda s: s.fillna("").str.strip().str.lower()
unseen = ~norm(val.body).isin(set(norm(train.body))).values

print(f"device={DEVICE} | docs={len(docs)} | queries={len(qs)} ({unseen.sum()} without a copy in docs)\n")

for name, (qp, pp) in MODELS.items():
    m = SentenceTransformer(name, device=DEVICE)
    m.encode(["warmup"])

    def enc(xs, p):
        return m.encode([p + x for x in xs], batch_size=64, normalize_embeddings=True)

    t = time.perf_counter()
    D = enc(docs[:1000], pp)
    per_1k = time.perf_counter() - t
    D = np.vstack([D, enc(docs[1000:], pp)])

    single = []
    for q in qs[:20]:
        t = time.perf_counter()
        enc([q], qp)
        single.append((time.perf_counter() - t) * 1000)

    S = enc(qs, qp) @ D.T
    top1 = S.argmax(1)
    hit = train.queue.values[top1] == val.queue.values
    top5 = np.argsort(-S, axis=1)[:, :5]
    any5 = np.array([val.queue.values[i] in train.queue.values[top5[i]] for i in range(len(qs))])
    cross = np.mean([train.language.values[top1[i]] != val.language.values[i] for i in range(len(qs))])

    print(f"{name}\n"
          f"  dim={D.shape[1]} | max_seq_len={m.max_seq_length}\n"
          f"  1k docs: {per_1k:.1f}s  (~{per_1k * 38.8 / 60:.1f} min for full KB of 38,810)\n"
          f"  single query: median {np.median(single):.0f} ms, max {np.max(single):.0f} ms\n"
          f"  top-1 queue match: {hit.mean():.1%} (all) | {hit[unseen].mean():.1%} (no copy in docs)\n"
          f"  top-5 any queue match: {any5.mean():.1%} | top-1 in other language: {cross:.1%}\n")
