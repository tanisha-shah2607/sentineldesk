#!/usr/bin/env bash
# Owner: Niti. One command to launch the console on macOS/Linux:  ./run.sh
# First run on a machine: creates .venv (Python 3.12/3.11), installs requirements, then builds the data split,
# S2's triage model and S3's retrieval index (scripts/prepare.py) if they are missing. Later runs start in seconds.
# After `git pull`, run `./run.sh --rebuild` to retrain/re-index (needed when a teammate changes them).
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
  PY=""
  for c in python3.12 python3.11 python3; do command -v "$c" >/dev/null && { PY="$c"; break; }; done
  [[ -z "$PY" ]] && { echo "Python 3.11/3.12 not found"; exit 1; }
  echo "Creating .venv with $($PY --version)"
  "$PY" -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi
[[ -f .env ]] || { cp .env.example .env; echo "Created .env from .env.example"; }

REBUILD=0
if [[ "${1:-}" == "--rebuild" ]]; then REBUILD=1; shift; fi
if [[ $REBUILD == 1 || ! -f models/triage.joblib ]] || ! .venv/bin/python -c "from src.retrieval import index_ready; raise SystemExit(0 if index_ready() else 1)" 2>/dev/null; then
  echo "Preparing data split, triage model and retrieval index (first run: several minutes)..."
  .venv/bin/python scripts/prepare.py --split --train --index || echo "WARNING: setup incomplete; the console will use labelled stand-ins"
fi

exec .venv/bin/python -m streamlit run app.py "$@"
