#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export STREAMLIT_BROWSER_GATHER_USAGE_STATS="${STREAMLIT_BROWSER_GATHER_USAGE_STATS:-false}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

if ! "$PYTHON_BIN" -c "import streamlit" >/dev/null 2>&1; then
  printf '%s\n' "Streamlit is not installed. Run:"
  printf '%s\n' "  cd /home/jin/local_customgui"
  printf '%s\n' "  python3 -m venv .venv"
  printf '%s\n' "  source .venv/bin/activate"
  printf '%s\n' "  python -m pip install -r requirements.txt"
  exit 1
fi

"$PYTHON_BIN" -m streamlit run streamlit_app.py \
  --server.address 127.0.0.1 \
  --server.port 8791
