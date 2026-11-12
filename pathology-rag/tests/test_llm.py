"""Выбор LLM-провайдера: изоляция ключей, отсутствие скрытых переходов в другие сервисы."""
import httpx
import pytest

from pathrag.agent import llm


class _Recorder:
    """Подмена llm._post_json: запоминает вызовы и отвечает заданным статусом."""

    def __init__(self, status: int = 200, content: str = "Готов"):
        self.calls: list[dict] = []
        self.status = status
        self.content = content

    def __call__(self, url, payload, headers, timeout):
        self.calls.append({"url": url, "headers": headers, "payload": payload})
        body = {"choices": [{"message": {"content": self.content}}]}
        return httpx.Response(self.status, json=body, request=httpx.Request("POST", url))


@pytest.fixture
def rec(monkeypatch):
    r = _Recorder()
    monkeypatch.setattr(llm, "_post_json", r)
    for env in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "GIGACHAT_CREDENTIALS"):
        monkeypatch.delenv(env, raising=False)
    return r


def test_explicit_provider_uses_only_its_url_and_key(rec):
    ans, label = llm.generate_with_llm("q", provider="groq", api_key="gsk_user")
    assert ans == "Готов" and label.startswith("Groq")
    assert len(rec.calls) == 1
    assert rec.calls[0]["url"].startswith("https://api.groq.com/")
    assert rec.calls[0]["headers"]["Authorization"] == "Bearer gsk_user"


def test_auto_mode_never_forwards_user_key_to_other_providers(rec, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini_env_key")
    llm.generate_with_llm("q", provider="auto", api_key="gsk_user")
    sent_keys = {c["headers"].get("Authorization") for c in rec.calls}
    assert "Bearer gsk_user" not in sent_keys
    assert rec.calls[0]["headers"]["Authorization"] == "Bearer gemini_env_key"


def test_explicit_provider_without_key_makes_no_request(rec):
    assert llm.generate_with_llm("q", provider="openrouter") == (None, None)
    assert rec.calls == []


@pytest.mark.parametrize("provider", ["groq", "tbank_local"])
def test_failed_provider_does_not_fall_back_to_public_cloud(rec, provider):
    rec.status = 500
    ans, _ = llm.generate_with_llm("q", provider=provider, api_key="k")
    assert ans is None
    assert all("pollinations" not in c["url"] for c in rec.calls)


def test_local_network_error_does_not_fall_back_to_public_cloud(monkeypatch):
    urls = []

    def boom(url, payload, headers, timeout):
        urls.append(url)
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(llm, "_post_json", boom)
    assert llm.generate_with_llm("q", provider="tbank_local")[0] is None
    assert urls and all("pollinations" not in u for u in urls)


def test_malformed_response_returns_none(monkeypatch):
    monkeypatch.setattr(
        llm, "_post_json",
        lambda url, *a: httpx.Response(200, text="not json", request=httpx.Request("POST", url)),
    )
    assert llm.generate_with_llm("q", provider="groq", api_key="k")[0] is None


def test_connection_check_reports_missing_key(rec):
    ok, msg = llm.test_llm_connection("gemini")
    assert not ok and "Gemini" in msg
    assert rec.calls == []


def test_get_secret_or_env_prefers_environment(monkeypatch):
    monkeypatch.setenv("PATHRAG_TEST_SECRET", "x")
    assert llm.get_secret_or_env("PATHRAG_TEST_SECRET") == "x"
    assert llm.get_secret_or_env("PATHRAG_TEST_MISSING", "d") == "d"


def test_free_cloud_accepts_short_answer(monkeypatch):
    # Раньше ответ короче 30 символов («Готов») считался сбоем и кнопка писала «сервер перегружен»
    monkeypatch.setattr(
        llm, "_post_json",
        lambda url, *a: httpx.Response(200, text="Готов", request=httpx.Request("POST", url)),
    )
    ok, msg = llm.test_llm_connection("free_cloud")
    assert ok and "Готов" in msg


def test_free_cloud_reports_real_failure_reason(monkeypatch):
    def slow(url, payload, headers, timeout):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(llm, "_post_json", slow)
    ok, msg = llm.test_llm_connection("free_cloud")
    assert not ok and "не ответил" in msg

    monkeypatch.setattr(
        llm, "_post_json",
        lambda url, *a: httpx.Response(429, text="Too Many Requests", request=httpx.Request("POST", url)),
    )
    ok, msg = llm.test_llm_connection("free_cloud")
    assert not ok and "лимит" in msg
