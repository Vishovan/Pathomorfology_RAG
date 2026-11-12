"""Узлы (Nodes) графа LangGraph."""
import logging
from typing import Any

from ..retrieval.external_search import search_arxiv, search_pubmed, search_web
from ..retrieval.hybrid_retriever import retrieve
from ..retrieval.image_retriever import search_pathology_images
from ..retrieval.legal_kb import get_medical_device_boundary_info, search_gost_standards
from ..text import tokens
from . import prompts
from .llm import generate_with_llm
from .prompts import LEGAL_DISCLAIMER, MED_DISCLAIMER  # noqa: F401  (реэкспорт для обратной совместимости)
from .router import route_intent
from .state import AgentState

log = logging.getLogger("pathrag.agent.nodes")

AUTONOMOUS_PROVIDER = "autonomous"


# ============================================================================ поиск
def route_node(state: AgentState) -> dict[str, Any]:
    """Узел маршрутизации запроса."""
    intent = route_intent(state.get("query", ""), explicit_mode=state.get("mode"))
    log.info("Query routed to: %s", intent)
    return {"intent": intent}


def retrieve_pathology_node(state: AgentState) -> dict[str, Any]:
    """Узел поиска в базе знаний по патоморфологии (+ релевантные микрофотографии)."""
    query = state.get("query", "")
    filters = {"organ": "Молочная железа"} if state.get("intent") == "breast_cancer_ihc" else None
    return {
        "retrieved_docs": retrieve(query, filters=filters, top_k=4),
        "retrieved_images": search_pathology_images(query, k=2),
    }


def grade_documents_node(state: AgentState) -> dict[str, Any]:
    """Узел оценки релевантности найденных документов (Corrective RAG).

    Доля значимых основ слов запроса, встречающихся в двух лучших документах.
    """
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {"is_relevant": False, "needs_external_search": True}

    query_stems = set(tokens(state.get("query", "")))
    top_text = " ".join(d.get("text") or "" for d in docs[:2]).lower().replace("ё", "е")
    matches = sum(1 for s in query_stems if s in top_text)
    is_relevant = matches >= max(1, len(query_stems) // 3)
    return {"is_relevant": is_relevant, "needs_external_search": not is_relevant}


def image_search_node(state: AgentState) -> dict[str, Any]:
    """Узел поиска микропрепаратов и WSI-слайдов."""
    return {"retrieved_images": search_pathology_images(state.get("query", ""), k=4)}


def legal_consultant_node(state: AgentState) -> dict[str, Any]:
    """Узел юридической консультации по ГОСТам и регуляторике медицинского ПО."""
    return {
        "legal_results": search_gost_standards(state.get("query", "")),
        "legal_boundary": get_medical_device_boundary_info(),
    }


def literature_search_node(state: AgentState) -> dict[str, Any]:
    """Узел поиска во внешних научных источниках (PubMed, arXiv, Web)."""
    query = state.get("query", "")
    return {
        "literature_results": search_pubmed(query, max_results=3) + search_arxiv(query, max_results=2),
        "web_results": search_web(query, max_results=2),
    }


# ============================================================================ генерация
def _llm_enabled(state: AgentState) -> bool:
    provider = state.get("llm_provider")
    if provider == AUTONOMOUS_PROVIDER:
        return False
    flag = state.get("use_llm")
    return bool(provider) if flag is None else flag


def _ask_llm(state: AgentState, prompt: str, system: str) -> tuple[str | None, str | None]:
    """Вызов LLM с параметрами из состояния; (None, None), если LLM выключена."""
    if not _llm_enabled(state):
        return None, None
    return generate_with_llm(
        prompt,
        system_prompt=system,
        provider=state.get("llm_provider") or "auto",
        api_key=state.get("llm_api_key"),
        endpoint_url=state.get("llm_endpoint"),
        model_name=state.get("llm_model"),
    )


def _with_fallback(state: AgentState, prompt: str, system: str, fallback_text: str, fallback_label: str) -> dict[str, Any]:
    answer, label = _ask_llm(state, prompt, system)
    if answer:
        return {"final_answer": answer, "llm_used": label, "is_llm_generated": True}
    return {"final_answer": fallback_text, "llm_used": fallback_label, "is_llm_generated": False}


def _answer_legal(state: AgentState) -> dict[str, Any]:
    query = state.get("query", "")
    standards = state.get("legal_results", [])
    boundary = state.get("legal_boundary") or get_medical_device_boundary_info()
    citations = [
        {"source": s.get("gost", "ГОСТ"), "title": s.get("title", ""), "reference": "Каталог Росстандарта"}
        for s in standards[:5]
    ]
    result = _with_fallback(
        state,
        prompts.legal_prompt(query, standards[:6], boundary),
        prompts.LEGAL_SYSTEM,
        prompts.legal_fallback(query, standards[:5], boundary),
        "Автономный нормативный синтез",
    )
    return {**result, "citations": citations, "disclaimer": LEGAL_DISCLAIMER}


def _answer_images(state: AgentState) -> dict[str, Any]:
    images = state.get("retrieved_images", [])
    citations = [
        {
            "source": img.get("source_name") or img.get("source_id", ""),
            "title": img.get("title", "Микропрепарат"),
            "reference": f"Увеличение {img.get('magnification', '')}, окраска {img.get('stain', '')}",
        }
        for img in images
    ]
    return {
        "final_answer": prompts.image_results_text(state.get("query", ""), images),
        "citations": citations,
        "disclaimer": MED_DISCLAIMER,
        "llm_used": None,
        "is_llm_generated": False,
    }


def _answer_conversational(state: AgentState) -> dict[str, Any]:
    result = _with_fallback(
        state,
        prompts.conversational_prompt(state.get("query", "")),
        prompts.CONVERSATIONAL_SYSTEM,
        prompts.CONVERSATIONAL_FALLBACK,
        "Системное приветствие",
    )
    return {**result, "citations": [], "disclaimer": prompts.CONVERSATIONAL_DISCLAIMER}


def _answer_pathology(state: AgentState) -> dict[str, Any]:
    query = state.get("query", "")
    docs = state.get("retrieved_docs", [])[:4]
    lit = state.get("literature_results", [])[:3]
    web = state.get("web_results", [])[:2]

    citations = [
        {"source": d.get("source", ""), "title": d.get("title", ""), "reference": d.get("page_reference", "")}
        for d in docs
    ]
    prompt = (
        prompts.clinical_prompt_from_docs(query, docs) if docs
        else prompts.clinical_prompt_from_external(query, lit, web)
    )
    result = _with_fallback(
        state,
        prompt,
        prompts.CLINICAL_SYSTEM,
        prompts.clinical_fallback(query, docs[:3], lit, web),
        "Автономный RAG-экстрактор (офлайн)",
    )
    return {**result, "citations": citations, "disclaimer": MED_DISCLAIMER}


_ANSWER_BY_INTENT = {
    "legal_gost": _answer_legal,
    "image_search": _answer_images,
    "conversational": _answer_conversational,
}


def generate_answer_node(state: AgentState) -> dict[str, Any]:
    """Узел синтеза итогового ответа: LLM по найденным фрагментам или детерминированный fallback."""
    handler = _ANSWER_BY_INTENT.get(state.get("intent", "pathology_qa"), _answer_pathology)
    return handler(state)
