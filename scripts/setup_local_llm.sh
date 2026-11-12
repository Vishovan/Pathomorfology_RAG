#!/usr/bin/env bash
# Поднимает локальный OpenAI-совместимый сервер с Т-Банк LLM (T-Tech T-pro 2.1 / T-lite).
#
#   macOS (Apple Silicon)  -> Ollama (t-tech/T-pro-it-2.1:q4_K_M)
#   Linux + NVIDIA GPU     -> vLLM   (t-tech/T-pro-it-2.1 или t-tech/T-lite-it-1.0)
#
# Ollama на :11434 приложение находит само. Для vLLM или другого адреса экспортируйте
# VLLM_ENDPOINT и TBANK_MODEL (команда печатается в конце); адрес также пишется в /tmp/vllm_endpoint.txt.
set -euo pipefail

ENDPOINT_FILE="/tmp/vllm_endpoint.txt"
OS="$(uname -s)"

wait_ready() {  # $1 = base url
  for _ in $(seq 1 120); do
    if curl -sf "$1/models" >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  echo "Сервер не ответил за 4 минуты: $1" >&2; return 1
}

if [[ "$OS" == "Darwin" ]]; then
  command -v ollama >/dev/null || { echo "Установите Ollama: brew install ollama" >&2; exit 1; }
  MODEL="${LLM_MODEL:-t-tech/T-pro-it-2.1:q4_K_M}"
  curl -sf http://localhost:11434/api/version >/dev/null || { nohup ollama serve >/tmp/ollama.log 2>&1 & }
  sleep 2
  ollama pull "$MODEL"
  BASE="http://localhost:11434/v1"
else
  command -v nvidia-smi >/dev/null || { echo "vLLM требует NVIDIA GPU." >&2; exit 1; }
  MODEL="${LLM_MODEL:-t-tech/T-pro-it-2.1}"
  PORT="${PORT:-8000}"
  python -c "import vllm" 2>/dev/null || pip install -q vllm
  nohup python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --port "$PORT" --max-model-len 16384 >/tmp/vllm_server.log 2>&1 &
  BASE="http://localhost:${PORT}/v1"
fi

wait_ready "$BASE"
echo "$BASE" > "$ENDPOINT_FILE"
echo "OK: $MODEL -> $BASE (записано в $ENDPOINT_FILE)"
echo "Для приложения: export VLLM_ENDPOINT=$BASE TBANK_MODEL=$MODEL"
