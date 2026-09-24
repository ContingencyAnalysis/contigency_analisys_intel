#!/usr/bin/env bash
set -euo pipefail

# Processa as métricas publicadas pelo backend para o mesmo recorte exibido no front.
# Use AGENT_LIMIT para controlar o tamanho do lote (padrão: 100).
export AGENT_SOURCE="${AGENT_SOURCE:-postgres}"
export AGENT_SOURCE_SYSTEM="${AGENT_SOURCE_SYSTEM:-Compras}"
export AGENT_POSTGRES_DATASET="${AGENT_POSTGRES_DATASET:-compras}"
export AGENT_LIMIT="${AGENT_LIMIT:-100}"
# Optional comma-separated IDs from the dashboard, e.g. AGENT_GROUP_IDS=254445,154054

PYTHON_BIN="${PYTHON_BIN:-../venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

exec "$PYTHON_BIN" query_result_ai.py
