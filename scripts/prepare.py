"""Owner: Niti. One-shot setup check before a demo:  python scripts/prepare.py [--split] [--index]

  1. Python version and key packages
  2. Data split present?          (--split runs make_split if missing)
  3. Retrieval index buildable?   (--index calls retrieval.build_index("all"))
  4. Pipeline contract check on the demo examples (same as `python -m src.pipeline --check`)
Never edits teammate code; it only calls their public functions.
"""
import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402


def step(msg):
    print(f"\n== {msg}")


def main(do_split: bool, do_index: bool) -> int:
    ok = True
    step("Python and packages")
    v = sys.version_info
    print(f"python {v.major}.{v.minor}.{v.micro}" + ("" if (3, 11) <= v[:2] <= (3, 12) else
                                                     "  WARNING: team targets 3.11/3.12"))
    for mod in ("pandas", "sklearn", "streamlit", "dotenv", "langdetect", "sentence_transformers", "faiss"):
        try:
            importlib.import_module(mod)
            print(f"  ok      {mod}")
        except Exception as e:  # noqa: BLE001
            print(f"  MISSING {mod} ({type(e).__name__})")

    step("Data split")
    # Only the KB file is checked by name: holdout paths are reserved for data.py/evaluate.py (leakage audit).
    have = C.KB_PATH.exists()
    n = len(list(C.DATA_DIR.glob("*.parquet"))) if C.DATA_DIR.exists() else 0
    print(f"  KB present, {n} parquet file(s) in {C.DATA_DIR}" if have else f"  missing under {C.DATA_DIR}")
    if not have and do_split:
        from src.data import make_split
        make_split()
        have = True

    step("Retrieval index")
    files = list(C.INDEX_DIR.glob("*")) if C.INDEX_DIR.exists() else []
    print(f"  {len(files)} file(s) in {C.INDEX_DIR}")
    if do_index:
        if not have:
            print("  skipped: no data split (run with --split)")
        else:
            from src.retrieval import build_index
            try:
                build_index("all")
                print("  built")
            except NotImplementedError:
                print("  build_index not implemented yet (S3); the console will use mock retrieval")

    step("Pipeline contract check")
    from src.pipeline import check
    ok &= check() == 0
    print("\nREADY" if ok else "\nNOT READY (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="store_true", help="download + split the dataset if missing")
    ap.add_argument("--index", action="store_true", help="build the retrieval index")
    a = ap.parse_args()
    raise SystemExit(main(a.split, a.index))
