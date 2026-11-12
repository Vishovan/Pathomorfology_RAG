"""Построение и компиляция графа LangGraph для Pathology RAG."""
from langgraph.graph import END, StateGraph

from .nodes import (
    generate_answer_node,
    grade_documents_node,
    image_search_node,
    legal_consultant_node,
    literature_search_node,
    retrieve_pathology_node,
    route_node,
)
from .state import AgentState


def router_decision(state: AgentState) -> str:
    """Выбор следующего шага после маршрутизатора."""
    intent = state.get("intent", "pathology_qa")
    if intent in ("breast_cancer_ihc", "pathology_qa"):
        return "retrieve_pathology"
    elif intent == "image_search":
        return "image_search"
    elif intent == "legal_gost":
        return "legal_consultant"
    elif intent == "literature_search":
        return "literature_search"
    return "retrieve_pathology"


def grader_decision(state: AgentState) -> str:
    """Выбор следующего шага после проверки релевантности документов."""
    if state.get("needs_external_search"):
        return "literature_search"
    return "generate_answer"


def build_pathology_graph():
    """Создает и компилирует граф LangGraph."""
    workflow = StateGraph(AgentState)

    # Добавление узлов
    workflow.add_node("route", route_node)
    workflow.add_node("retrieve_pathology", retrieve_pathology_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("image_search", image_search_node)
    workflow.add_node("legal_consultant", legal_consultant_node)
    workflow.add_node("literature_search", literature_search_node)
    workflow.add_node("generate_answer", generate_answer_node)

    # Установка точки входа
    workflow.set_entry_point("route")

    # Условные переходы от роутера
    workflow.add_conditional_edges(
        "route",
        router_decision,
        {
            "retrieve_pathology": "retrieve_pathology",
            "image_search": "image_search",
            "legal_consultant": "legal_consultant",
            "literature_search": "literature_search",
        },
    )

    # Переход от поиска патологии к грейдеру
    workflow.add_edge("retrieve_pathology", "grade_documents")

    # Условный переход от грейдера: сразу ответ или внешний поиск
    workflow.add_conditional_edges(
        "grade_documents",
        grader_decision,
        {
            "generate_answer": "generate_answer",
            "literature_search": "literature_search",
        },
    )

    # Переходы к генератору ответа
    workflow.add_edge("literature_search", "generate_answer")
    workflow.add_edge("image_search", "generate_answer")
    workflow.add_edge("legal_consultant", "generate_answer")

    # Выход из графа
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


# Создаем скомпилированный экземпляр приложения графа
pathology_agent = build_pathology_graph()


def ask_agent(
    query: str,
    mode: str | None = None,
    llm_provider: str | None = None,
    llm_api_key: str | None = None,
    use_llm: bool | None = None,
) -> dict:
    """Удобная функция запуска агента для одного запроса."""
    initial_state = {
        "query": query,
        "mode": mode,
        "chat_history": [],
        "llm_provider": llm_provider,
        "llm_api_key": llm_api_key,
        "use_llm": use_llm,
    }
    result = pathology_agent.invoke(initial_state)
    return result
