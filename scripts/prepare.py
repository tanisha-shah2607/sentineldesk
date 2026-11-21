"""Owner: Niti. One-shot setup before a demo:  python scripts/prepare.py [--split] [--train] [--index]

  1. Python version and key packages
  2. Data split present?          (--split runs scripts/make_split.py if missing)
  3. Triage model trained?        (--train runs S2's scripts/train_triage.py, ~1.5 min)
  4. Retrieval index built?       (--index runs S3's scripts/build_index.py, ~2 min on a GPU)
  5. Pipeline contract check on the demo examples (same as `python -m src.pipeline --check`)
Fresh machine: python scripts/prepare.py --split --train --index
Never edits teammate code; it only calls their public functions.
"""
import argparse
import importlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402


def step(msg):
    print(f"\n== {msg}")


def run(script: str) -> bool:
    print(f"  running {script} …", flush=True)
    return subprocess.run([sys.executable, str(C.ROOT / script)], cwd=C.ROOT).returncode == 0


def skip(why: str) -> bool:
    print(f"  skipped: {why}")
    return False


def main(do_split: bool, do_train: bool, do_index: bool) -> int:
    ok = True
    step("Python and packages")
    v = sys.version_info
    print(f"python {v.major}.{v.minor}.{v.micro}" + ("" if (3, 11) <= v[:2] <= (3, 12) else
                                                     "  WARNING: team targets 3.11/3.12"))
    for mod in ("pandas", "sklearn", "streamlit", "dotenv", "langdetect", "joblib", "torch", "sentence_transformers"):
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
        ok &= run("scripts/make_split.py")
        have = C.KB_PATH.exists()

    step("Triage model (S2)")
    from src.triage import MODEL_PATH
    print(f"  {'present' if MODEL_PATH.exists() else 'missing'}: {MODEL_PATH}")
    if do_train:
        ok &= run("scripts/train_triage.py") if have else skip("no data split (add --split)")

    step("Retrieval index (S3)")
    from src.retrieval import index_ready
    print("  ready" if index_ready() else f"  not built (or stale) in {C.INDEX_DIR}")
    if do_index:
        ok &= run("scripts/build_index.py") if have else skip("no data split (add --split)")

    step("Pipeline contract check")
    from src.pipeline import check
    ok &= check() == 0
    print("\nREADY" if ok else "\nNOT READY (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="store_true", help="download + split the dataset if missing")
    ap.add_argument("--train", action="store_true", help="train the triage model (S2)")
    ap.add_argument("--index", action="store_true", help="build the retrieval index (S3)")
    a = ap.parse_args()
    raise SystemExit(main(a.split, a.train, a.index))
