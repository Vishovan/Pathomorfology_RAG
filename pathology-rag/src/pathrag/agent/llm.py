"""Модуль интеграции с моделями генерации (LLM) для Pathology RAG.

Провайдеры:
1. Groq Cloud, Google Gemini, OpenRouter — облачные OpenAI-совместимые API (нужен ключ);
2. Т-Банк LLM / Ollama / vLLM — локальный OpenAI-совместимый сервер (tbank_local, openai_compat);
3. Сбер GigaChat (152-ФЗ) — через пакет `gigachat`;
4. Free Cloud AI (Pollinations, без ключей) — ТОЛЬКО при явном выборе free_cloud или режиме auto.

Правила безопасности:
- ключ, переданный в `api_key`, отправляется только выбранному провайдеру; в режиме auto каждый
  провайдер получает лишь собственный ключ из окружения / secrets;
- при сбое выбранного провайдера запрос НЕ перенаправляется в другой (тем более публичный) сервис.
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger("pathrag.agent.llm")

DEFAULT_TBANK_MODEL = "llama3.1:8b"
DEFAULT_LOCAL_ENDPOINT = "http://localhost:11434/v1"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
DEFAULT_GIGACHAT_MODEL = "GigaChat-Pro"

DEFAULT_SYSTEM_PROMPT = (
    "Вы — ведущий врач-патологоанатом и эксперт по молекулярно-биологической диагностике опухолей. "
    "Сформулируйте профессиональный, структурированный и клинически выверенный ответ на русском языке. "
    "Используйте только проверенные данные из предоставленного контекста, выделяйте критерии, проценты, "
    "шкалы и обязательно ссылайтесь на источники."
)

# Предпочтительные локальные модели (от быстрых к тяжёлым)
LOCAL_MODEL_PREFERENCE = ["llama3.1:8b", "llama3.1", "qwen2.5-coder:14b", "t-tech/T-pro-it-2.1:q4_K_M"]
_LARGE_MODEL_MARKERS = ("32b", "33b", "70b", "t-pro")

PROVIDERS_REQUIRING_LOCAL_SERVER = ("tbank_local", "openai_compat")


@dataclass(frozen=True)
class CloudProvider:
    """Облачный OpenAI-совместимый провайдер."""
    key: str
    label: str
    url: str
    key_env: str
    model_env: str
    default_model: str
    timeout: float = 30.0
    extra_headers: dict[str, str] = field(default_factory=dict)


CLOUD_PROVIDERS: dict[str, CloudProvider] = {
    "groq": CloudProvider(
        "groq", "Groq", "https://api.groq.com/openai/v1/chat/completions",
        "GROQ_API_KEY", "GROQ_MODEL", DEFAULT_GROQ_MODEL,
    ),
    "gemini": CloudProvider(
        "gemini", "Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "GEMINI_API_KEY", "GEMINI_MODEL", DEFAULT_GEMINI_MODEL,
    ),
    "openrouter": CloudProvider(
        "openrouter", "OpenRouter", "https://openrouter.ai/api/v1/chat/completions",
        "OPENROUTER_API_KEY", "OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL, timeout=35.0,
        extra_headers={"HTTP-Referer": "https://github.com/Vishovan/Pathomorfology_RAG", "X-Title": "Pathology RAG Copilot"},
    ),
}

# Подсказки для диагностики подключения (test_llm_connection)
_MISSING_KEY_HINTS = {
    "groq": "Не указан API-ключ Groq. Введите ключ (gsk_...) в боковой панели или добавьте в secrets.toml. Бесплатно на https://console.groq.com",
    "gemini": "Не указан API-ключ Gemini. Введите ключ (AIza...) в боковой панели. Бесплатно на https://aistudio.google.com",
    "openrouter": "Не указан API-ключ OpenRouter. Введите ключ (sk-or-...) в боковой панели на https://openrouter.ai",
}


# --------------------------------------------------------------------------- окружение
def get_secret_or_env(name: str, default: str = "") -> str:
    """Значение из os.environ или streamlit.secrets; никогда не бросает исключений."""
    val = os.getenv(name)
    if val:
        return val
    try:
        import streamlit as st

        sec = getattr(st, "secrets", None)
        if sec is not None and name in sec:
            return str(sec[name])
    except Exception:  # нет streamlit, нет secrets.toml и т.п.
        pass
    return default


def _local_endpoint(endpoint: str | None = None) -> str:
    return endpoint or get_secret_or_env("VLLM_ENDPOINT") or get_secret_or_env("OLLAMA_ENDPOINT") or DEFAULT_LOCAL_ENDPOINT


def _server_base(endpoint: str) -> str:
    """Базовый адрес сервера без суффикса /v1 (для служебных API Ollama)."""
    base = endpoint.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def is_local_ollama_available(endpoint: str | None = None) -> bool:
    """Быстро проверяет, доступен ли локальный сервер Ollama / vLLM."""
    try:
        with httpx.Client(timeout=1.2) as client:
            return client.get(_server_base(_local_endpoint(endpoint)) + "/api/version").status_code == 200
    except Exception:
        return False


# --------------------------------------------------------------------------- транспорт
def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> httpx.Response:
    """Единственная точка HTTP-вызова (подменяется в тестах)."""
    with httpx.Client(timeout=timeout) as client:
        return client.post(url, json=payload, headers=headers)


def _messages(system: str, prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def _openai_chat(
    url: str, *, model: str, system: str, prompt: str, api_key: str | None, timeout: float,
    temperature: float = 0.2, max_tokens: int = 2048, extra_headers: dict[str, str] | None = None,
    log_name: str = "LLM",
) -> str | None:
    """Вызов OpenAI-совместимого /chat/completions. Возвращает текст ответа или None."""
    headers = {"Content-Type": "application/json", **(extra_headers or {})}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": _messages(system, prompt),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        res = _post_json(url, payload, headers, timeout)
    except Exception as e:
        log.warning("%s invocation failed: %s", log_name, e)
        return None
    if res.status_code != 200:
        log.warning("%s API error (%s): %s", log_name, res.status_code, res.text[:200])
        return None
    try:
        choices = res.json().get("choices") or []
        ans = (choices[0].get("message", {}).get("content") or "").strip() if choices else ""
    except Exception as e:
        log.warning("%s returned malformed response: %s", log_name, e)
        return None
    return ans or None


# --------------------------------------------------------------------------- провайдеры
def _call_cloud(p: CloudProvider, system: str, prompt: str, api_key: str, model_name: str | None):
    model = model_name or os.getenv(p.model_env) or p.default_model
    ans = _openai_chat(p.url, model=model, system=system, prompt=prompt, api_key=api_key,
                       timeout=p.timeout, extra_headers=p.extra_headers, log_name=p.label)
    return ans, f"{p.label} ({model})"


def _pick_local_model(endpoint: str, requested: str | None) -> str:
    """Выбирает модель локального Ollama: запрошенную, если она установлена, иначе предпочтительную."""
    target = requested or get_secret_or_env("TBANK_MODEL") or DEFAULT_TBANK_MODEL
    if "localhost" not in endpoint and "127.0.0.1" not in endpoint:
        return target
    try:
        with httpx.Client(timeout=1.0) as c:
            r = c.get(_server_base(endpoint) + "/api/tags")
        installed = [m.get("name") for m in r.json().get("models", [])] if r.status_code == 200 else []
    except Exception:
        return target
    if not installed or target in installed:
        return target
    return next((m for m in LOCAL_MODEL_PREFERENCE if m in installed), installed[0])


def _call_local(system: str, prompt: str, api_key: str | None, endpoint_url: str | None, model_name: str | None):
    endpoint = _local_endpoint(endpoint_url).rstrip("/")
    if not endpoint.endswith("/v1") and ("11434" in endpoint or "8000" in endpoint):
        endpoint += "/v1"
    model = _pick_local_model(endpoint, model_name)
    is_large = any(x in model.lower() for x in _LARGE_MODEL_MARKERS)
    ans = _openai_chat(
        f"{endpoint}/chat/completions", model=model, system=system, prompt=prompt, api_key=api_key,
        timeout=240.0 if is_large else float(os.getenv("LLM_TIMEOUT", "90")),
        temperature=0.3, max_tokens=1024 if is_large else 1536, log_name=f"Local LLM ({endpoint})",
    )
    return ans, f"Т-Банк / Ollama ({model})"


def _call_gigachat(system: str, prompt: str, credentials: str, model_name: str | None):
    model = model_name or DEFAULT_GIGACHAT_MODEL
    label = f"Сбер GigaChat ({model})"
    try:
        from gigachat import GigaChat
    except ImportError:
        log.warning("GigaChat: пакет `gigachat` не установлен (pip install gigachat)")
        return None, label

    # Сертификат НУЦ Минцифры: укажите GIGACHAT_CA_BUNDLE. Без него проверка TLS отключается
    # только при явном GIGACHAT_VERIFY_SSL=false.
    ca_bundle = get_secret_or_env("GIGACHAT_CA_BUNDLE")
    verify = get_secret_or_env("GIGACHAT_VERIFY_SSL", "true").lower() not in ("0", "false", "no")
    if not verify:
        log.warning("GigaChat: проверка TLS-сертификата отключена (GIGACHAT_VERIFY_SSL=false)")
    kwargs: dict[str, Any] = {
        "credentials": credentials,
        "scope": get_secret_or_env("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        "model": model,
        "verify_ssl_certs": verify,
    }
    if ca_bundle:
        kwargs["ca_bundle_file"] = ca_bundle
    try:
        with GigaChat(**kwargs) as giga:
            response = giga.chat(f"{system}\n\nКонтекст и вопрос:\n{prompt}")
        if response.choices and response.choices[0].message:
            return (response.choices[0].message.content or "").strip() or None, label
    except Exception as e:
        log.warning("GigaChat failed: %s", e)
    return None, label


def _call_free_cloud(system: str, prompt: str):
    """Pollinations anonymous tier. Запрос уходит на публичный сервер — только по явному выбору."""
    label = "Free Cloud AI (Zero-Config)"
    payload = {"messages": _messages(system, prompt), "model": "openai-fast", "temperature": 0.3}
    try:
        res = _post_json("https://text.pollinations.ai/", payload, {"Content-Type": "application/json"}, 22.0)
    except Exception as e:
        log.debug("Free Cloud AI failed: %s", e)
        return None, label
    text = res.text.strip()
    if res.status_code == 200 and len(text) > 30 and "limit reached" not in text.lower():
        return text, label
    return None, label


def _generate(prompt: str, system: str, provider: str, api_key: str | None,
              endpoint_url: str | None, model_name: str | None) -> tuple[str | None, str | None]:
    # 1. Явно выбранный провайдер: только он, без перехода к другим сервисам
    if provider in CLOUD_PROVIDERS:
        p = CLOUD_PROVIDERS[provider]
        key = api_key or get_secret_or_env(p.key_env)
        if not key:
            log.warning("%s: API-ключ не задан", p.label)
            return None, None
        return _call_cloud(p, system, prompt, key, model_name)
    if provider in PROVIDERS_REQUIRING_LOCAL_SERVER:
        return _call_local(system, prompt, api_key, endpoint_url, model_name)
    if provider == "gigachat":
        cred = api_key or get_secret_or_env("GIGACHAT_CREDENTIALS")
        if not cred:
            log.warning("GigaChat: не заданы GIGACHAT_CREDENTIALS")
            return None, None
        return _call_gigachat(system, prompt, cred, model_name)
    if provider == "free_cloud":
        return _call_free_cloud(system, prompt)
    if provider != "auto":
        log.warning("Неизвестный LLM-провайдер: %s", provider)
        return None, None

    # 2. auto: каскад по провайдерам, для которых настроены СОБСТВЕННЫЕ ключи окружения.
    #    Явный api_key в auto-режиме игнорируется: неизвестно, какому сервису он принадлежит.
    if api_key:
        log.warning("auto: переданный api_key проигнорирован — укажите провайдера явно")
    for p in CLOUD_PROVIDERS.values():
        key = get_secret_or_env(p.key_env)
        if key:
            ans, label = _call_cloud(p, system, prompt, key, None)
            if ans:
                return ans, label
    if is_local_ollama_available(endpoint_url):
        ans, label = _call_local(system, prompt, None, endpoint_url, None)
        if ans:
            return ans, label
    cred = get_secret_or_env("GIGACHAT_CREDENTIALS")
    if cred:
        ans, label = _call_gigachat(system, prompt, cred, None)
        if ans:
            return ans, label
    return _call_free_cloud(system, prompt)


def generate_with_llm(
    prompt: str,
    system_prompt: str | None = None,
    provider: str = "auto",  # groq | gemini | openrouter | tbank_local | openai_compat | gigachat | free_cloud | auto
    api_key: str | None = None,
    endpoint_url: str | None = None,
    model_name: str | None = None,
) -> tuple[str | None, str | None]:
    """Генерирует ответ выбранным провайдером (или каскадом в режиме auto).

    Возвращает (текст ответа или None, метка использованной модели или None).
    """
    return _generate(prompt, system_prompt or DEFAULT_SYSTEM_PROMPT, provider or "auto",
                     api_key, endpoint_url, model_name)


def test_llm_connection(
    provider: str,
    api_key: str | None = None,
    endpoint_url: str | None = None,
    model_name: str | None = None,
) -> tuple[bool, str]:
    """Проверяет доступность выбранного провайдера LLM с понятной диагностикой ошибок."""
    if provider in CLOUD_PROVIDERS and not (api_key or get_secret_or_env(CLOUD_PROVIDERS[provider].key_env)):
        return False, _MISSING_KEY_HINTS[provider]
    if provider in PROVIDERS_REQUIRING_LOCAL_SERVER:
        ep = _local_endpoint(endpoint_url)
        if not is_local_ollama_available(ep):
            return False, f"Сервер Ollama недоступен по адресу {ep}. Запустите его командой: ollama serve (или bash scripts/setup_local_llm.sh)"

    try:
        reply, _ = generate_with_llm(
            prompt="Тест связи. Ответь одним словом: 'Готов'.",
            system_prompt="Отвечай кратко одним словом.",
            provider=provider,
            api_key=api_key,
            endpoint_url=endpoint_url,
            model_name=model_name,
        )
    except Exception as e:
        return False, f"Ошибка подключения ({provider}): {e}"
    if reply:
        preview = reply.replace("\n", " ")
        if len(preview) > 60:
            preview = preview[:60] + "..."
        return True, f"Соединение успешно! Ответ модели: {preview}"
    if provider == "free_cloud":
        return False, "Публичный сервер Free Cloud сейчас перегружен. Для надежной работы укажите бесплатный ключ Groq (console.groq.com) в панели слева."
    return False, "Модель вернула пустой ответ, ошибку или превысила таймаут (подробности в логе)."


# pytest собирает функции test_* из импортирующих модулей; это не тест
test_llm_connection.__test__ = False  # type: ignore[attr-defined]
