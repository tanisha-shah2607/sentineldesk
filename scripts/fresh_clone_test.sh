#!/usr/bin/env bash
# Owner: Niti. Proves the repo works from a clean checkout, the way a judge would run it.
# Usage: scripts/fresh_clone_test.sh [--light] [git-ref]
#   --light   install only what tests + console need (skips torch/sentence-transformers; triage + retrieval run as stubs)
#   git-ref   branch/commit to test (default: current HEAD, including only COMMITTED changes)
set -euo pipefail

LIGHT=0
if [[ "${1:-}" == "--light" ]]; then LIGHT=1; shift; fi
REPO="$(cd "$(dirname "$0")/.." && pwd)"
REF="${1:-$(git -C "$REPO" rev-parse HEAD)}"
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  for c in python3.12 python3.11 python3; do command -v "$c" >/dev/null && { PY="$c"; break; }; done
fi
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== cloning $REF into $TMP"
git clone -q "$REPO" "$TMP/repo"
git -C "$TMP/repo" checkout -q "$REF"
cd "$TMP/repo"

echo "== venv ($($PY --version))"
"$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
if [[ $LIGHT == 1 ]]; then
  .venv/bin/pip install -q pandas pyarrow fastparquet scikit-learn joblib numpy langdetect streamlit python-dotenv requests pytest
else
  .venv/bin/pip install -q -r requirements.txt
fi

echo "== pytest"
.venv/bin/python -m pytest -q
echo "== pipeline --check"
.venv/bin/python -m src.pipeline --check
echo "== streamlit boot"
.venv/bin/python -m streamlit run app.py --server.headless true --server.port 8599 >streamlit.log 2>&1 &
PID=$!
ok=0
for _ in $(seq 1 30); do
  if curl -fsS http://localhost:8599/_stcore/health >/dev/null 2>&1; then ok=1; break; fi
  sleep 1
done
kill $PID 2>/dev/null || true
if [[ $ok == 1 ]] && ! grep -qi "traceback" streamlit.log; then
  echo "streamlit OK"
else
  cat streamlit.log; echo "streamlit FAILED"; exit 1
fi
echo "== FRESH CLONE TEST PASSED"
