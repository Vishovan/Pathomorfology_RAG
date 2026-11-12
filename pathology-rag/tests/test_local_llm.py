"""Интеграционный тест локальной LLM (Ollama / vLLM). Пропускается, если сервер не запущен.

Запуск сервера: bash scripts/setup_local_llm.sh
"""
import os

import httpx
import pytest

from pathrag.agent import ask_agent
from pathrag.agent.llm import test_llm_connection as check_connection

ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://localhost:11434/v1")
MODEL = os.getenv("TBANK_MODEL", "t-tech/T-pro-it-2.1:q4_K_M")


def _server_up() -> bool:
    try:
        r = httpx.get(f"{ENDPOINT}/models", timeout=2)
        if r.status_code != 200:
            return False
        avail = [m.get("id") for m in r.json().get("data", [])]
        return MODEL in avail or any("t-tech" in m or "llama" in m for m in avail)
    except Exception:
        return False


pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(not _server_up(), reason=f"LLM сервер {ENDPOINT} не запущен"),
]


def test_connection():
    ok, msg = check_connection("tbank_local", endpoint_url=ENDPOINT, model_name=MODEL)
    assert ok, msg


def test_agent_answers_with_llm():
    res = ask_agent("Опишите морфологию казеозного некроза", llm_provider="tbank_local",
                    llm_endpoint=ENDPOINT, llm_model=MODEL, use_llm=True)
    answer = res.get("final_answer", "")
    assert len(answer) > 100
    assert res.get("citations")
