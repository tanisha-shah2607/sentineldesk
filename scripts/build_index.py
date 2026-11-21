"""Owner: S3. Build the retrieval artifacts in index/ (gitignored).

Usage: python scripts/build_index.py [--force]
Step 4 (now): embed every clean KB ticket once and cache it (index/kb_emb.npy + kb_meta.parquet).
Steps 5-6 (next): dedupe + FAISS indexes for part="train" and part="all" will be added here.
Run scripts/make_split.py first.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retrieval import embed_kb  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-embed even if the cache exists")
    a = ap.parse_args()
    t = time.perf_counter()
    embed_kb(force=a.force)
    print(f"Done in {time.perf_counter() - t:.0f}s")
