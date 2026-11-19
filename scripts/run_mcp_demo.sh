#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/pathology-rag/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pathrag.mcp_calculators.network_server &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
trap 'exit 130' INT TERM
python -m streamlit run streamlit_app.py
