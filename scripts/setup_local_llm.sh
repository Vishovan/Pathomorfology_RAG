#!/usr/bin/env bash
# Поднимает локальный OpenAI-совместимый сервер с Llama 3.1 8B Instruct.
#
#   macOS (Apple Silicon)  -> Ollama  (vLLM на macOS не поддерживает GPU Apple)
#   Linux + NVIDIA GPU     -> vLLM    (нужен HF_TOKEN с принятой лицензией Llama 3.1 на huggingface.co)
#
# Эндпоинт пишется в /tmp/vllm_endpoint.txt — streamlit_app.py подхватывает его автоматически
# при локальном запуске. Для Streamlit Cloud укажите публичный адрес сервера в Secrets:
#   VLLM_ENDPOINT = "https://<ваш-сервер>/v1"   VLLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
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
  MODEL="${LLM_MODEL:-llama3.1:8b}"
  curl -sf http://localhost:11434/api/version >/dev/null || { nohup ollama serve >/tmp/ollama.log 2>&1 & }
  sleep 2
  ollama pull "$MODEL"
  BASE="http://localhost:11434/v1"
else
  command -v nvidia-smi >/dev/null || { echo "vLLM требует NVIDIA GPU (>=16 ГБ VRAM для 8B в bf16)." >&2; exit 1; }
  : "${HF_TOKEN:?Задайте HF_TOKEN (модель meta-llama gated: примите лицензию на huggingface.co)}"
  MODEL="${LLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
  PORT="${PORT:-8000}"
  python -c "import vllm" 2>/dev/null || pip install -q vllm
  nohup python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --port "$PORT" --max-model-len 16384 \
      ${VLLM_API_KEY:+--api-key "$VLLM_API_KEY"} >/tmp/vllm_server.log 2>&1 &
  BASE="http://localhost:${PORT}/v1"
fi

wait_ready "$BASE"
echo "$BASE" > "$ENDPOINT_FILE"
echo "OK: $MODEL -> $BASE (записано в $ENDPOINT_FILE)"
