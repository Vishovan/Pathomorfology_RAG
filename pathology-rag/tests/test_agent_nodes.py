"""Узлы графа и ретриверы: релевантность картинок, передача состояния, fallback без LLM."""
from pathrag.agent import ask_agent, llm
from pathrag.agent.nodes import grade_documents_node
from pathrag.agent.router import route_intent
from pathrag.retrieval.hybrid_retriever import _rrf_merge
from pathrag.retrieval.image_retriever import search_pathology_images


def test_no_irrelevant_images_for_unrelated_query():
    # Раньше при отсутствии совпадений возвращались первые записи базы (HER2 IHC)
    assert search_pathology_images("Морфология туберкулёзной гранулёмы", k=2) == []


def test_short_generic_stem_alone_is_not_enough():
    # «рак» совпадает с «Рак молочной железы», но вопрос про желудок — снимки РМЖ не нужны
    assert search_pathology_images("рак желудка", k=2) == []


def test_relevant_image_is_found():
    imgs = search_pathology_images("серое опеченение пневмония", k=2)
    assert imgs and "пневмония" in imgs[0]["title"].lower()


def test_pathology_answer_does_not_attach_her2_images_to_unrelated_topic():
    res = ask_agent("Опишите морфологию казеозного некроза")
    assert all("her2" not in (img.get("title") or "").lower() for img in res.get("retrieved_images", []))


def test_legal_boundary_is_kept_in_state():
    res = ask_agent("Какие ГОСТы нужны для ТЗ?")
    assert res["intent"] == "legal_gost"
    assert "pathology_rag_strategy" in res.get("legal_boundary", {})


def test_autonomous_provider_never_calls_llm(monkeypatch):
    def fail(*a, **k):
        raise AssertionError("LLM не должна вызываться")

    monkeypatch.setattr(llm, "_generate", fail)
    res = ask_agent("Критерии HER2 2+", llm_provider="autonomous", use_llm=True)
    assert res["is_llm_generated"] is False
    assert res["llm_used"]


def test_llm_answer_is_used_when_available(monkeypatch):
    from pathrag.agent import nodes

    monkeypatch.setattr(nodes, "generate_with_llm", lambda *a, **k: ("Ответ модели", "Mock (m)"))
    res = ask_agent("Критерии HER2 2+", llm_provider="groq", llm_api_key="k")
    assert res["final_answer"] == "Ответ модели"
    assert res["llm_used"] == "Mock (m)" and res["is_llm_generated"] is True


def test_grader_ignores_stopwords():
    docs = [{"text": "Казеозный некроз характерен для туберкулёза."}]
    assert grade_documents_node({"query": "Что такое казеозный некроз?", "retrieved_docs": docs})["is_relevant"]
    other = [{"text": "Совсем другой текст про атеросклероз аорты."}]
    assert not grade_documents_node({"query": "Что такое казеозный некроз?", "retrieved_docs": other})["is_relevant"]


def test_router_matches_word_starts_only():
    # «молочн» внутри «кисломолочных» — не про рак молочной железы
    assert route_intent("Влияние кисломолочных продуктов на гастрит") == "pathology_qa"
    assert route_intent("Привет") == "conversational"
    assert route_intent("Найди свежие статьи про HER2-low") == "literature_search"


def test_rrf_merge_deduplicates_and_interleaves():
    a = [{"text": "alpha"}, {"text": "beta"}]
    b = [{"text": "beta"}, {"text": "gamma"}]
    merged = [d["text"] for d in _rrf_merge(a, b, top_k=3)]
    assert merged[0] == "beta" and set(merged) == {"alpha", "beta", "gamma"}
