"""Owner: S3. Build the retrieval artifacts in index/ (gitignored).

Usage: python scripts/build_index.py [--force]
  1. Embed every clean KB ticket once and cache it (index/kb_emb.npy + kb_meta.parquet). Skipped if
     the cache for the current model exists; --force re-embeds.
  2. Always rebuild both indexes from the cache (seconds): vecs_all (app) and vecs_train
     (threshold tuning on part="val" queries). Re-run after any change to the val slice.
Run scripts/make_split.py first.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retrieval import PARTS, build_index, embed_kb  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-embed even if the cache exists")
    a = ap.parse_args()
    t = time.perf_counter()
    embed_kb(force=a.force)
    for part in PARTS:
        build_index(part)
    print(f"Done in {time.perf_counter() - t:.0f}s")
