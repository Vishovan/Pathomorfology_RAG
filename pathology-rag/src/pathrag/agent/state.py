"""Схема состояния LangGraph для мультимодального патоморфологического агента."""
from typing import Annotated, Any, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # Входные данные
    query: str
    mode: str | None  # явный выбор режима: 'pathology', 'images', 'legal', 'literature', 'auto'
    chat_history: list[dict[str, str]]
    llm_provider: str | None  # 'gigachat', 'groq', 'free_cloud', 'qwen_local', 'openai_compat', 'auto'
    llm_api_key: str | None
    llm_endpoint: str | None
    llm_model: str | None
    use_llm: bool | None

    # Маршрутизация
    intent: str  # pathology_qa | breast_cancer_ihc | image_search | legal_gost | literature_search

    # Промежуточные результаты
    retrieved_docs: list[dict[str, Any]]
    retrieved_images: list[dict[str, Any]]
    legal_results: list[dict[str, Any]]
    literature_results: list[dict[str, Any]]
    web_results: list[dict[str, Any]]

    # Оценка качества и переформулировка
    is_relevant: bool
    needs_external_search: bool
    rewritten_query: str | None

    # Итоговый ответ
    final_answer: str
    citations: list[dict[str, str]]
    disclaimer: str
