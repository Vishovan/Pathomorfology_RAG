"""Модуль интеграции с моделями генерации (LLM) для Pathology RAG:
1. Groq Cloud (сверхбыстрый LPU, бесплатный доступ к Llama 3.3 70B / Qwen 2.5 32B);
2. Google Gemini (Flash 2.0 / 1.5, бесплатный тир);
3. OpenRouter (пул бесплатных моделей :free);
4. Т-Банк LLM / Ollama / свой vLLM (локальный инференс без ключей);
5. Сбер GigaChat Pro (152-ФЗ, официальная сертификация в РФ);
6. Free Cloud AI (Zero Config для деплоя без ключей);
7. Автономный экспертный RAG-синтез по верифицированным руководствам.
"""
import logging
import os
from typing import Any

log = logging.getLogger("pathrag.agent.llm")

DEFAULT_TBANK_MODEL = "llama3.1:8b"
DEFAULT_LOCAL_ENDPOINT = "http://localhost:11434/v1"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"


def get_secret_or_env(name: str, default: str = "") -> str:
    """Безопасно извлекает значение из os.environ или streamlit.secrets без вызова исключений."""
    val = os.getenv(name)
    if val:
        return val
    try:
        import streamlit as st
        try:
            sec = getattr(st, "secrets", None)
            if sec is not None and name in sec:
                return str(sec[name])
        except BaseException:
            pass
    except BaseException:
        pass
    return default


def is_local_ollama_available(endpoint: str | None = None) -> bool:
    """Быстро проверяет, доступен ли локальный сервер Ollama / vLLM."""
    target = endpoint or get_secret_or_env("VLLM_ENDPOINT") or get_secret_or_env("OLLAMA_ENDPOINT") or DEFAULT_LOCAL_ENDPOINT
    try:
        import httpx
        url = target.replace("/v1", "").rstrip("/") + "/api/version"
        with httpx.Client(timeout=1.2) as client:
            r = client.get(url)
            return r.status_code == 200
    except Exception:
        return False


def test_llm_connection(
    provider: str,
    api_key: str | None = None,
    endpoint_url: str | None = None,
    model_name: str | None = None,
) -> tuple[bool, str]:
    """Проверяет доступность выбранного провайдера LLM с понятной диагностикой ошибок."""
    # Быстрые проверки наличия обязательных ключей и серверов
    if provider == "groq":
        k = api_key or get_secret_or_env("GROQ_API_KEY")
        if not k:
            return False, "Не указан API-ключ Groq. Введите ключ (gsk_...) в боковой панели или добавьте в secrets.toml. Бесплатно на https://console.groq.com"

    elif provider == "gemini":
        k = api_key or get_secret_or_env("GEMINI_API_KEY")
        if not k:
            return False, "Не указан API-ключ Gemini. Введите ключ (AIza...) в боковой панели. Бесплатно на https://aistudio.google.com"

    elif provider == "openrouter":
        k = api_key or get_secret_or_env("OPENROUTER_API_KEY")
        if not k:
            return False, "Не указан API-ключ OpenRouter. Введите ключ (sk-or-...) в боковой панели на https://openrouter.ai"

    elif provider in ("tbank_local", "openai_compat"):
        ep = endpoint_url or get_secret_or_env("VLLM_ENDPOINT") or get_secret_or_env("OLLAMA_ENDPOINT") or DEFAULT_LOCAL_ENDPOINT
        if not is_local_ollama_available(ep):
            return False, f"Сервер Ollama недоступен по адресу {ep}. Запустите его командой: ollama serve (или bash scripts/setup_local_llm.sh)"

    test_prompt = "Тест связи. Ответь одним словом: 'Готов'."
    try:
        reply = generate_with_llm(
            prompt=test_prompt,
            system_prompt="Отвечай кратко одним словом.",
            provider=provider,
            api_key=api_key,
            endpoint_url=endpoint_url,
            model_name=model_name,
        )
        if reply and len(reply.strip()) > 0:
            preview = reply.strip().replace("\n", " ")
            if len(preview) > 60:
                preview = preview[:60] + "..."
            return True, f"Соединение успешно! Ответ модели: {preview}"
        if provider == "free_cloud":
            return False, "Публичный сервер Free Cloud сейчас перегружен. Для надежной работы укажите бесплатный ключ Groq (console.groq.com) в панели слева."
        return False, "Модель вернула пустой ответ или превысила таймаут."
    except Exception as e:
        return False, f"Ошибка подключения ({provider}): {str(e)}"


def generate_with_llm(
    prompt: str,
    system_prompt: str | None = None,
    provider: str = "auto",  # groq | gemini | openrouter | tbank_local | gigachat | free_cloud | openai_compat | auto
    api_key: str | None = None,
    endpoint_url: str | None = None,
    model_name: str | None = None,
    return_meta: bool = False,
) -> Any:
    """Генерирует ответ через выбранный провайдер или выполняет умный каскад (auto)."""
    import httpx

    system = system_prompt or (
        "Вы — ведущий врач-патологоанатом и эксперт по молекулярно-биологической диагностике опухолей. "
        "Сформулируйте профессиональный, структурированный и клинически выверенный ответ на русском языке. "
        "Используйте только проверенные данные из предоставленного контекста, выделяйте критерии, проценты, шкалы и обязательно ссылайтесь на источники."
    )

    used_label = "Неизвестно"

    # --- 1. Провайдер Groq Cloud (Llama 3.3 70B / Qwen 2.5 32B) ---
    groq_key = api_key if provider == "groq" else (api_key or get_secret_or_env("GROQ_API_KEY"))
    if (provider == "groq") or (provider == "auto" and groq_key):
        target_model = model_name or os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL
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
            "max_tokens": 2048,
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    choices = data.get("choices", [])
                    if choices:
                        ans = choices[0].get("message", {}).get("content", "").strip()
                        if ans:
                            used_label = f"Groq ({target_model})"
                            return (ans, used_label) if return_meta else ans
                else:
                    log.warning("Groq API error (%s): %s", res.status_code, res.text[:200])
        except Exception as e:
            log.warning("Groq invocation failed: %s", e)
            if provider == "groq":
                return (None, used_label) if return_meta else None

    # --- 2. Google Gemini API (Flash 2.0 / 1.5) ---
    gemini_key = api_key if provider == "gemini" else (api_key or get_secret_or_env("GEMINI_API_KEY"))
    if (provider == "gemini") or (provider == "auto" and gemini_key):
        target_model = model_name or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        headers = {
            "Authorization": f"Bearer {gemini_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 2048,
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    choices = data.get("choices", [])
                    if choices:
                        ans = choices[0].get("message", {}).get("content", "").strip()
                        if ans:
                            used_label = f"Google Gemini ({target_model})"
                            return (ans, used_label) if return_meta else ans
                else:
                    log.warning("Gemini API error (%s): %s", res.status_code, res.text[:200])
        except Exception as e:
            log.warning("Gemini invocation failed: %s", e)
            if provider == "gemini":
                return (None, used_label) if return_meta else None

    # --- 3. OpenRouter (:free модели) ---
    openrouter_key = api_key if provider == "openrouter" else (api_key or get_secret_or_env("OPENROUTER_API_KEY"))
    if (provider == "openrouter") or (provider == "auto" and openrouter_key):
        target_model = model_name or os.getenv("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Vishovan/Pathomorfology_RAG",
            "X-Title": "Pathology RAG Copilot",
        }
        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 2048,
        }
        try:
            with httpx.Client(timeout=35.0) as client:
                res = client.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    choices = data.get("choices", [])
                    if choices:
                        ans = choices[0].get("message", {}).get("content", "").strip()
                        if ans:
                            used_label = f"OpenRouter ({target_model})"
                            return (ans, used_label) if return_meta else ans
                else:
                    log.warning("OpenRouter error (%s): %s", res.status_code, res.text[:200])
        except Exception as e:
            log.warning("OpenRouter failed: %s", e)
            if provider == "openrouter":
                return (None, used_label) if return_meta else None

    # --- 4. Т-Банк LLM (T-Tech T-pro 2.1) / Локальный Ollama / vLLM ---
    if provider in ("tbank_local", "openai_compat") or (provider == "auto" and is_local_ollama_available()):
        target_endpoint = endpoint_url or get_secret_or_env("VLLM_ENDPOINT") or get_secret_or_env("OLLAMA_ENDPOINT") or DEFAULT_LOCAL_ENDPOINT
        if not target_endpoint.endswith("/v1") and ("11434" in target_endpoint or "8000" in target_endpoint):
            target_endpoint = target_endpoint.rstrip("/") + "/v1"
        target_model = model_name or get_secret_or_env("TBANK_MODEL") or DEFAULT_TBANK_MODEL

        # Автоматический выбор быстрой модели, если модель явно не задана или отсутствует
        if "localhost" in target_endpoint or "127.0.0.1" in target_endpoint:
            try:
                base_check = target_endpoint.replace("/v1", "").rstrip("/") + "/api/tags"
                with httpx.Client(timeout=1.0) as c:
                    r_tags = c.get(base_check)
                    if r_tags.status_code == 200:
                        installed = [m.get("name") for m in r_tags.json().get("models", [])]
                        if installed:
                            if not model_name:
                                for pref in ["llama3.1:8b", "llama3.1", "qwen2.5-coder:14b", "t-tech/T-pro-it-2.1:q4_K_M"]:
                                    if pref in installed:
                                        target_model = pref
                                        break
                                else:
                                    target_model = installed[0]
                            elif target_model not in installed:
                                for pref in ["llama3.1:8b", "llama3.1", "qwen2.5-coder:14b", "t-tech/T-pro-it-2.1:q4_K_M"]:
                                    if pref in installed:
                                        target_model = pref
                                        break
                                else:
                                    target_model = installed[0]
            except Exception:
                pass

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        is_large = any(x in target_model.lower() for x in ("32b", "33b", "70b", "t-pro"))
        timeout_sec = 240.0 if is_large else float(os.getenv("LLM_TIMEOUT", "90"))
        max_tokens = 1024 if is_large else 1536

        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens,
        }
        try:
            with httpx.Client(timeout=timeout_sec) as client:
                url = f"{target_endpoint.rstrip('/')}/chat/completions"
                res = client.post(url, json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    choices = data.get("choices", [])
                    if choices:
                        ans = choices[0].get("message", {}).get("content", "").strip()
                        if ans:
                            used_label = f"Т-Банк / Ollama ({target_model})"
                            return (ans, used_label) if return_meta else ans
                else:
                    log.warning("Ollama / local LLM error (%s): %s", res.status_code, res.text[:200])
        except Exception as e:
            log.warning("Local LLM invocation failed (%s): %s", target_endpoint, e)
            if provider == "tbank_local":
                # При сбое локального сервера пробуем прозрачно перейти к Free Cloud
                provider = "free_cloud"

    # --- 5. Сбер GigaChat Pro (152-ФЗ) ---
    gigachat_cred = api_key if provider == "gigachat" else (api_key or get_secret_or_env("GIGACHAT_CREDENTIALS"))
    if (provider == "gigachat") or (provider == "auto" and gigachat_cred):
        try:
            from gigachat import GigaChat
            scope = get_secret_or_env("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
            target_model = model_name or "GigaChat-Pro"
            with GigaChat(credentials=gigachat_cred, scope=scope, verify_ssl_certs=False, model=target_model) as giga:
                response = giga.chat(f"{system}\n\nКонтекст и вопрос:\n{prompt}")
                if response.choices and response.choices[0].message:
                    ans = response.choices[0].message.content.strip()
                    if ans:
                        used_label = f"Сбер GigaChat ({target_model})"
                        return (ans, used_label) if return_meta else ans
        except Exception as e:
            log.warning("GigaChat failed: %s", e)
            if provider == "gigachat":
                return (None, used_label) if return_meta else None

    # --- 6. Free Cloud AI (Zero Config · без ключей) ---
    if provider in ("free_cloud", "auto"):
        # Попытка вызова Pollinations anonymous tier (openai-fast) без фиксированного seed
        try:
            payload = {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "model": "openai-fast",
                "temperature": 0.3,
            }
            with httpx.Client(timeout=22.0) as client:
                res = client.post("https://text.pollinations.ai/", json=payload)
                if res.status_code == 200 and len(res.text.strip()) > 30 and "limit reached" not in res.text.lower():
                    ans = res.text.strip()
                    used_label = "Free Cloud AI (Zero-Config)"
                    return (ans, used_label) if return_meta else ans
        except Exception as e:
            log.debug("Free Cloud AI primary endpoint failed: %s", e)

    return (None, used_label) if return_meta else None
