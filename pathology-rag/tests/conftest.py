"""Общие фикстуры: по умолчанию тесты изолированы от сети.

Внешний поиск (PubMed, arXiv, DuckDuckGo) возвращает пустые списки, HTTP-вызовы LLM запрещены,
Qdrant считается недоступным (работает только локальный BM25).
Тесты, которым действительно нужна сеть / локальный сервер, помечаются @pytest.mark.network.
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "network: тест обращается к реальным внешним сервисам")


@pytest.fixture(autouse=True)
def _no_network(request, monkeypatch):
    if request.node.get_closest_marker("network"):
        return
    from pathrag import index
    from pathrag.agent import llm, nodes

    # Живой Qdrant на машине разработчика не должен менять результаты тестов (и грузить bge-m3)
    monkeypatch.setattr(index, "qdrant_collection_ready", lambda *a, **k: False)

    for name in ("search_pubmed", "search_arxiv", "search_web"):
        monkeypatch.setattr(nodes, name, lambda *a, **k: [])

    def _blocked(*args, **kwargs):
        raise AssertionError("HTTP-вызов LLM в тесте без @pytest.mark.network")

    monkeypatch.setattr(llm, "_post_json", _blocked)
    monkeypatch.setattr(llm, "is_local_ollama_available", lambda *a, **k: False)
