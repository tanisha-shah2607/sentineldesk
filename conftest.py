"""Repo-root pytest config.

- Being at the repo root, this file makes pytest put the root on sys.path, so plain `pytest`
  (the README command) can import `src` (before, only `python -m pytest` worked).
- If the retrieval index has not been built on this machine (python scripts/build_index.py),
  retrieval runs in stub mode so contract tests still pass without index/.
"""
import os

from src.retrieval import index_ready

if not index_ready():
    os.environ.setdefault("RETRIEVAL_STUB", "1")
