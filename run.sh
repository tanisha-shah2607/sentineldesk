#!/usr/bin/env bash
# Owner: Niti. One command to launch the console on macOS/Linux:  ./run.sh
# Creates .venv (Python 3.12/3.11) and installs requirements on first run.
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

exec .venv/bin/python -m streamlit run app.py "$@"
