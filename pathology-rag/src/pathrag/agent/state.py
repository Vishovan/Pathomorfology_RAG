"""Схема состояния LangGraph для мультимодального патоморфологического агента."""
from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # Входные данные
    query: str
    mode: str | None  # явный выбор режима: 'pathology', 'images', 'legal', 'literature', 'auto'
    llm_provider: str | None  # 'groq', 'gemini', 'openrouter', 'tbank_local', 'gigachat', 'free_cloud', 'openai_compat', 'auto', 'autonomous'
    llm_api_key: str | None
    llm_endpoint: str | None
    llm_model: str | None
    use_llm: bool | None

    # Маршрутизация
    intent: str  # pathology_qa | breast_cancer_ihc | image_search | legal_gost | literature_search | conversational

    # Промежуточные результаты
    retrieved_docs: list[dict[str, Any]]
    retrieved_images: list[dict[str, Any]]
    legal_results: list[dict[str, Any]]
    legal_boundary: dict[str, Any]
    literature_results: list[dict[str, Any]]
    web_results: list[dict[str, Any]]

    # Оценка релевантности (Corrective RAG)
    is_relevant: bool
    needs_external_search: bool

    # Итоговый ответ и метаданные
    final_answer: str
    citations: list[dict[str, str]]
    disclaimer: str
    llm_used: str | None
    is_llm_generated: bool | None
