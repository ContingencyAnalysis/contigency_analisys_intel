#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

exec "$PYTHON_BIN" -m uvicorn api:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}" --reload
