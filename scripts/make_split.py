"""Usage: python scripts/make_split.py [--csv path/to/file.csv]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import make_split  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="local CSV instead of HF download")
    make_split(ap.parse_args().csv)
