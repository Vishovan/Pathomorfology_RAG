"""Модуль интеграции с коммерчески доступными LLM:
1. Сбер GigaChat (152-ФЗ, официальная лицензия для коммерциализации в РФ);
2. Groq Cloud (сверхбыстрый и бесплатный API для Qwen 2.5 / Llama 3.3);
3. OpenRouter / DeepSeek / OpenAI-совместимый API (Apache 2.0 / MIT);
4. Локальный/корпоративный vLLM / Ollama;
5. Free Cloud AI (Pollinations AI — работа без ключей при наличии сети);
6. Экспертный детерминированный синтез (автономный режим).
"""
import logging
import os
from typing import Any

log = logging.getLogger("pathrag.agent.llm")


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
    provider: str = "auto",  # gigachat | groq | free_cloud | qwen_local | openai_compat | auto
    api_key: str | None = None,
    endpoint_url: str | None = None,
    model_name: str | None = None,
) -> str | None:
    """Генерирует ответ через выбранную модель или возвращает None при отсутствии ключей/сервера."""
    system = system_prompt or (
        "Вы — ведущий патоморфолог-эксперт и специалист по медицинским информационным системам. "
        "Сформулируйте профессиональный, структурированный и клинически выверенный ответ на русском языке. "
        "Используйте только проверенные данные из предоставленного контекста, выделяйте критерии, проценты, шкалы и обязательно ссылайтесь на источники."
    )

    # 1. Попытка вызова Сбер GigaChat (152-ФЗ, коммерческая лицензия в РФ)
    gigachat_cred = api_key or os.getenv("GIGACHAT_CREDENTIALS") or os.getenv("GIGACHAT_API_KEY")
    if provider in ("gigachat", "auto") and gigachat_cred:
        try:
            from gigachat import GigaChat

            scope = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
            model = model_name or "GigaChat-Pro"
            with GigaChat(credentials=gigachat_cred, scope=scope, verify_ssl_certs=False, model=model) as giga:
                response = giga.chat(f"{system}\n\nКонтекст и вопрос:\n{prompt}")
                if response.choices and response.choices[0].message:
                    return response.choices[0].message.content.strip()
        except Exception as e:
            log.warning("GigaChat generation failed: %s", e)
            if provider == "gigachat":
                return None

    # 2. Попытка вызова через Groq Cloud (бесплатный ультрабыстрый API для Qwen 2.5 / Llama 3.3)
    groq_key = api_key or os.getenv("GROQ_API_KEY")
    if provider in ("groq", "auto") and groq_key:
        try:
            import httpx

            target_model = model_name or "llama-3.3-70b-versatile"
            headers = {
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": target_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 1800,
            }
            with httpx.Client(timeout=25.0) as client:
                res = client.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "").strip()
        except Exception as e:
            log.warning("Groq API failed: %s", e)
            if provider == "groq":
                return None

    # 3. Бесплатный облачный инференс (Free Cloud AI / Pollinations / Qwen 2.5) без ключей
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
            with httpx.Client(timeout=15.0) as client:
                res = client.post("https://text.pollinations.ai/", json=payload)
                if res.status_code == 200 and len(res.text.strip()) > 30:
                    return res.text.strip()
        except Exception as e:
            log.debug("Free cloud AI fallback unavailable: %s", e)
            if provider == "free_cloud":
                return None

    # 4. Попытка вызова локального / OpenAI-совместимого эндпоинта (vLLM / Ollama с Qwen 2.5 или OpenRouter)
    endpoint = endpoint_url or os.getenv("VLLM_ENDPOINT") or os.getenv("OLLAMA_ENDPOINT")
    if provider in ("qwen_local", "openai_compat", "openrouter") or (provider == "auto" and endpoint):
        target_endpoint = endpoint or "http://localhost:11434/v1"
        try:
            import httpx

            target_model = model_name or os.getenv("QWEN_MODEL", "qwen2.5:14b-instruct")
            headers = {"Content-Type": "application/json"}
            auth_token = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"

            payload = {
                "model": target_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 1800,
            }
            with httpx.Client(timeout=float(os.getenv("LLM_TIMEOUT", "180"))) as client:
                url = f"{target_endpoint.rstrip('/')}/chat/completions"
                res = client.post(url, json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "").strip()
        except Exception as e:
            log.warning("OpenAI-compatible / Qwen endpoint failed: %s", e)

    return None
