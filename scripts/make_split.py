"""Usage: python scripts/make_split.py [--csv path/to/file.csv] [--allow-mismatch]
Prints row count, first ticket ids and a fingerprint; aborts if they differ from config.EXPECTED_*."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import make_split  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="local CSV instead of HF download")
    ap.add_argument("--allow-mismatch", action="store_true", help="only for the person re-pinning the dataset")
    a = ap.parse_args()
    make_split(a.csv, a.allow_mismatch)
