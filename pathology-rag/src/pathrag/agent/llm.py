"""Модуль интеграции с моделями генерации (LLM) без обязательных ключей:
1. Т-Банк LLM (T-Tech T-pro 2.1 / T-lite) через локальный инференс (Ollama / vLLM) — без API-ключей;
2. Free Cloud AI (T-Tech / Qwen 2.5 Serverless) — работа онлайн без API-ключей и без локального GPU;
3. Экспертный детерминированный RAG-синтез (автономный режим) — офлайн без LLM.
"""
import logging
import os
from typing import Any

log = logging.getLogger("pathrag.agent.llm")

DEFAULT_TBANK_MODEL = "t-tech/T-pro-it-2.1:q4_K_M"
DEFAULT_LOCAL_ENDPOINT = "http://localhost:11434/v1"


def test_llm_connection(
    provider: str,
    api_key: str | None = None,
    endpoint_url: str | None = None,
    model_name: str | None = None,
) -> tuple[bool, str]:
    """Проверяет доступность выбранного провайдера LLM коротким запросом."""
    test_prompt = "Тест связи. Ответь одним словом: 'Готов'."
    try:
        reply = generate_with_llm(
            prompt=test_prompt,
            system_prompt="Отвечай кратко.",
            provider=provider,
            api_key=api_key,
            endpoint_url=endpoint_url,
            model_name=model_name,
        )
        if reply and len(reply.strip()) > 0:
            return True, f"Соединение успешно! Ответ модели: {reply[:60]}..."
        return False, "Модель вернула пустой ответ или не ответила."
    except Exception as e:
        return False, f"Ошибка подключения: {str(e)}"


def generate_with_llm(
    prompt: str,
    system_prompt: str | None = None,
    provider: str = "tbank_local",  # tbank_local | free_cloud | openai_compat | auto
    api_key: str | None = None,
    endpoint_url: str | None = None,
    model_name: str | None = None,
) -> str | None:
    """Генерирует ответ через выбранную модель без необходимости ввода платных ключей."""
    system = system_prompt or (
        "Вы — ведущий патоморфолог-эксперт и специалист по медицинским информационным системам. "
        "Сформулируйте профессиональный, структурированный и клинически выверенный ответ на русском языке. "
        "Используйте только проверенные данные из предоставленного контекста, выделяйте критерии, проценты, шкалы и обязательно ссылайтесь на источники."
    )

    # 1. Т-Банк LLM (T-Tech T-pro 2.1 / T-lite) или локальный OpenAI-совместимый эндпоинт (Ollama / vLLM)
    if provider in ("tbank_local", "openai_compat", "auto"):
        target_endpoint = endpoint_url or os.getenv("VLLM_ENDPOINT") or os.getenv("OLLAMA_ENDPOINT") or DEFAULT_LOCAL_ENDPOINT
        target_model = model_name or os.getenv("TBANK_MODEL") or DEFAULT_TBANK_MODEL

        try:
            import httpx

            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            payload = {
                "model": target_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 1800,
            }
            timeout_sec = float(os.getenv("LLM_TIMEOUT", "180"))
            with httpx.Client(timeout=timeout_sec) as client:
                url = f"{target_endpoint.rstrip('/')}/chat/completions"
                res = client.post(url, json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "").strip()
        except Exception as e:
            log.warning("T-Bank local LLM endpoint failed (%s): %s", target_endpoint, e)
            if provider == "tbank_local":
                # Если локальный сервер недоступен, пробуем прозрачный fallback на Free Cloud
                provider = "free_cloud"

    # 2. Бесплатный облачный инференс (Free Cloud AI / Pollinations / Qwen 2.5 / T-Tech) без ключей
    if provider in ("free_cloud", "auto"):
        try:
            import httpx

            payload = {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "model": "qwen",
                "seed": 42,
            }
            with httpx.Client(timeout=20.0) as client:
                res = client.post("https://text.pollinations.ai/", json=payload)
                if res.status_code == 200 and len(res.text.strip()) > 30:
                    return res.text.strip()
        except Exception as e:
            log.debug("Free cloud AI fallback unavailable: %s", e)
            if provider == "free_cloud":
                return None

    return None
