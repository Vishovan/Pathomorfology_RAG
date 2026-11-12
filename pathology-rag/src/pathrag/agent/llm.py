"""Модуль интеграции с коммерчески доступными LLM:
1. Сбер GigaChat (152-ФЗ, разрешен для коммерциализации в РФ);
2. Qwen 2.5 (Apache 2.0 open-source, топ-качество, свободен для коммерции);
3. DeepSeek-V3/R1 (MIT лицензия);
4. Детерминированный экспертный генератор (Zero-cost, без необходимости ввода API-ключей).
"""
import logging
import os
from typing import Any

log = logging.getLogger("pathrag.agent.llm")


def generate_with_llm(
    prompt: str,
    system_prompt: str | None = None,
    provider: str = "auto",  # gigachat | qwen_local | openai_compat | auto
    api_key: str | None = None,
    endpoint_url: str | None = None,
    model_name: str | None = None,
) -> str | None:
    """Генерирует ответ через выбранную модель или возвращает None при отсутствии ключей/сервера."""
    system = system_prompt or (
        "Вы — ведущий эксперт по патоморфологии и медицинским информационным системам. "
        "Отвечайте строго на основе предоставленного контекста, указывайте источники цитирования и сохраняйте академическую точность."
    )

    # 1. Попытка использования Сбер GigaChat (коммерческая лицензия в РФ, 152-ФЗ)
    gigachat_cred = api_key or os.getenv("GIGACHAT_CREDENTIALS") or os.getenv("GIGACHAT_API_KEY")
    if provider in ("gigachat", "auto") and gigachat_cred:
        try:
            from gigachat import GigaChat

            client = GigaChat(
                credentials=gigachat_cred,
                scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
                verify_ssl_certs=False,
                model=model_name or "GigaChat-Pro",
            )
            response = client.chat(f"{system}\n\n{prompt}")
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content
        except Exception as e:
            log.warning("GigaChat generation failed: %s", e)
            if provider == "gigachat":
                return None

    # 2. Попытка использования Qwen 2.5 через локальный OpenAI-совместимый vLLM / Ollama
    endpoint = endpoint_url or os.getenv("VLLM_ENDPOINT") or os.getenv("OLLAMA_ENDPOINT") or "http://localhost:11434/v1"
    if provider in ("qwen_local", "openai_compat") or (provider == "auto" and os.getenv("VLLM_ENDPOINT")):
        try:
            import httpx

            target_model = model_name or os.getenv("QWEN_MODEL", "qwen2.5:14b-instruct")
            payload = {
                "model": target_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 1500,
            }
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            with httpx.Client(timeout=30.0) as client:
                res = client.post(f"{endpoint}/chat/completions", json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content")
        except Exception as e:
            log.warning("Local/Qwen endpoint failed: %s", e)

    return None
