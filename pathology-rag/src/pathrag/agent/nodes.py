"""Узлы (Nodes) графа LangGraph."""
import logging
from typing import Any

from ..retrieval.external_search import search_arxiv, search_pubmed, search_web
from ..retrieval.hybrid_retriever import retrieve
from ..retrieval.image_retriever import search_pathology_images
from ..retrieval.legal_kb import get_medical_device_boundary_info, search_gost_standards
from .router import route_intent
from .state import AgentState

log = logging.getLogger("pathrag.agent.nodes")

MED_DISCLAIMER = "⚠️ Справочно-образовательная система для патоморфологов. Данные носят исключительно академический и информационный характер, не являются медицинским изделием и не предназначены для постановки диагноза конкретным пациентам."
LEGAL_DISCLAIMER = "⚖️ Юридическая и нормативная справка. Подготовлена в соответствии с действующими национальными стандартами ГОСТ и нормативными актами РФ."


def route_node(state: AgentState) -> dict[str, Any]:
    """Узел маршрутизации запроса."""
    query = state.get("query", "")
    mode = state.get("mode")
    intent = route_intent(query, explicit_mode=mode)
    log.info("Query routed to: %s", intent)
    return {"intent": intent}


def retrieve_pathology_node(state: AgentState) -> dict[str, Any]:
    """Узел поиска в базе знаний по патоморфологии."""
    query = state.get("query", "")
    intent = state.get("intent", "pathology_qa")

    filters = {}
    if intent == "breast_cancer_ihc":
        filters["organ"] = "Молочная железа"

    docs = retrieve(query, filters=filters, top_k=4)

    # Параллельно находим релевантные микрофотографии к нозологии
    images = search_pathology_images(query, k=2)

    return {
        "retrieved_docs": docs,
        "retrieved_images": images,
    }


def grade_documents_node(state: AgentState) -> dict[str, Any]:
    """Узел оценки релевантности найденных документов (Corrective RAG)."""
    docs = state.get("retrieved_docs", [])
    query = state.get("query", "").lower()

    if not docs:
        return {"is_relevant": False, "needs_external_search": True}

    # Простая проверка покрытия ключевых терминов
    query_terms = [w.strip(".,;:?!()") for w in query.split() if len(w) >= 3]
    top_doc_text = " ".join([d.get("text", "").lower() for d in docs[:2]])

    matches = sum(1 for term in query_terms if term in top_doc_text)
    is_relevant = matches >= max(1, len(query_terms) // 3)

    return {
        "is_relevant": is_relevant,
        "needs_external_search": not is_relevant,
    }


def image_search_node(state: AgentState) -> dict[str, Any]:
    """Узел поиска микропрепаратов и WSI-слайдов."""
    query = state.get("query", "")
    images = search_pathology_images(query, k=4)
    return {"retrieved_images": images}


def legal_consultant_node(state: AgentState) -> dict[str, Any]:
    """Узел юридической консультации по ГОСТам и регуляторике медицинского ПО."""
    query = state.get("query", "")
    standards = search_gost_standards(query)
    boundary = get_medical_device_boundary_info()
    return {
        "legal_results": standards,
        "legal_boundary": boundary,
    }


def literature_search_node(state: AgentState) -> dict[str, Any]:
    """Узел поиска во внешних научных источниках (PubMed, arXiv, Web)."""
    query = state.get("query", "")
    pubmed_articles = search_pubmed(query, max_results=3)
    arxiv_papers = search_arxiv(query, max_results=2)
    web_articles = search_web(query, max_results=2)

    return {
        "literature_results": pubmed_articles + arxiv_papers,
        "web_results": web_articles,
    }


def generate_answer_node(state: AgentState) -> dict[str, Any]:
    """Узел синтеза итогового структурированного ответа с цитатами."""
    intent = state.get("intent", "pathology_qa")
    query = state.get("query", "")
    citations = []

    # 1. Формирование юридического ответа
    if intent == "legal_gost":
        legal_res = state.get("legal_results", [])
        boundary = state.get("legal_boundary", get_medical_device_boundary_info())

        lines = [
            "### ⚖️ Консультация по ГОСТам и регуляторике ПО в сфере здравоохранения\n",
            f"**Запрос:** {query}\n",
        ]

        if "медиздели" in query.lower() or "samd" in query.lower() or "роспатент" in query.lower() or "минцифры" in query.lower() or "росздравнадзор" in query.lower():
            lines.append("#### 1. Разграничение трех самостоятельных процедур регистрации:\n")
            lines.append("| Направление | Государственный орган | Итоговый результат |")
            lines.append("| :--- | :--- | :--- |")
            lines.append("| **Регистрация программы для ЭВМ** | Роспатент (ФИПС) | Свидетельство о госрегистрации ЭВМ, защита кода и авторства |")
            lines.append("| **Включение в Реестр отечественного ПО** | Минцифры России | Освобождение от НДС, допуск к госзакупкам по 44-ФЗ/223-ФЗ |")
            lines.append("| **Регистрация в качестве медизделия (SaMD)** | Росздравнадзор | Регистрационное удостоверение (РУ) на медицинское изделие |\n")
            lines.append("> [!IMPORTANT]\n> **Принципиальное отличие:** регистрация в Роспатенте не заменяет реестр Минцифры, а включение в реестр Минцифры НЕ означает допуск системы как медицинского изделия.\n")

            lines.append("#### 2. Квалификация Pathology RAG: как избежать статуса медизделия?\n")
            lines.append(f"**Стратегия для ТЗ:** {boundary.get('pathology_rag_strategy', '')}\n")
            lines.append("- **Критерии НЕ отнесения к SaMD:** система не анализирует персональные снимки пациентов, не формулирует клинический диагноз и предназначена для академическо-справочных целей врачей и студентов.")

        if legal_res:
            lines.append("\n#### 3. Применимые стандарты ГОСТ:\n")
            for item in legal_res[:5]:
                gost_name = item.get("gost", "ГОСТ")
                title = item.get("title", "")
                summary = item.get("summary", "")
                practical = item.get("practical_value", "")
                lines.append(f"##### 📌 {gost_name}: {title}")
                lines.append(f"- **Назначение:** {summary}")
                lines.append(f"- **Практическое применение в проекте:** {practical}\n")
                citations.append({"source": gost_name, "title": title, "reference": "Каталог Росстандарта"})

        answer_text = "\n".join(lines)
        disclaimer = LEGAL_DISCLAIMER

    # 2. Формирование ответа по поиску изображений
    elif intent == "image_search":
        images = state.get("retrieved_images", [])
        lines = [
            "### 🖼️ Результаты поиска микропрепаратов и WSI-слайдов\n",
            f"**Запрос:** {query}\n",
        ]

        if not images:
            lines.append("К сожалению, по данному запросу не найдено подходящих микропрепаратов в базе данных.")
        else:
            for i, img in enumerate(images, 1):
                title = img.get("title", "Микропрепарат")
                diag = img.get("diagnosis", "")
                organ = img.get("organ", "")
                stain = img.get("stain", "")
                mag = img.get("magnification", "")
                cap = img.get("caption", "")
                src_name = img.get("source_name") or img.get("source_id", "")
                wsi_url = img.get("wsi_url") or img.get("page_url", "")

                lines.append(f"#### #{i} {title}")
                lines.append(f"- **Орган / ткань:** {organ}")
                lines.append(f"- **Окраска:** `{stain}` | **Увеличение:** `{mag}`")
                lines.append(f"- **Диагноз:** {diag}")
                lines.append(f"- **Морфологическое описание:** {cap}")
                lines.append(f"- **Источник:** {src_name}")
                if wsi_url:
                    lines.append(f"- 🔬 **Интерактивный Whole Slide Image:** [Открыть слайд в цифровом вьюере]({wsi_url})")
                lines.append("")
                citations.append({"source": src_name, "title": title, "reference": f"Увеличение {mag}, окраска {stain}"})

        answer_text = "\n".join(lines)
        disclaimer = MED_DISCLAIMER

    # 3. Формирование ответа по патоморфологии (РМЖ / ИГХ / общая патология)
    else:
        docs = state.get("retrieved_docs", [])
        images = state.get("retrieved_images", [])
        lit = state.get("literature_results", [])
        web = state.get("web_results", [])

        lines = [
            "### 🔬 Патоморфологический справочник\n",
        ]

        if docs:
            for doc in docs[:3]:
                title = doc.get("title", "")
                text = doc.get("text", "")
                src = doc.get("source", "")
                page = doc.get("page_reference", "")

                lines.append(f"#### 📖 {title}\n")
                lines.append(f"{text}\n")
                lines.append(f"> *Источник цитирования: [{src}, {page}]*\n")
                citations.append({"source": src, "title": title, "reference": page})
        else:
            lines.append("В локальной базе знаний точных фрагментов не обнаружено. Выполнен поиск по открытым научным источникам.\n")

        # Добавление связанных микрофотографий
        if images:
            lines.append("\n#### 📸 Иллюстративный материал (Микропрепараты):\n")
            for img in images[:2]:
                title = img.get("title", "")
                stain = img.get("stain", "")
                mag = img.get("magnification", "")
                cap = img.get("caption", "")
                wsi_url = img.get("wsi_url", "")
                src = img.get("source_name", "")
                lines.append(f"- **{title}** (`{stain}`, `{mag}`): {cap}")
                if wsi_url:
                    lines.append(f"  👉 [Смотреть WSI-слайд]({wsi_url}) ({src})")
            lines.append("")

        # Добавление внешней литературы при необходимости
        if lit or web:
            lines.append("\n#### 📚 Дополнительные научные публикации и ссылки:")
            for item in lit[:3]:
                title = item.get("title", "")
                url = item.get("url", "")
                stype = item.get("source_type", "")
                lines.append(f"- [{title}]({url}) — *{stype}*")
            for item in web[:2]:
                title = item.get("title", "")
                url = item.get("url", "")
                lines.append(f"- [{title}]({url}) — *Веб-источник*")

        answer_text = "\n".join(lines)
        disclaimer = MED_DISCLAIMER

    return {
        "final_answer": answer_text,
        "citations": citations,
        "disclaimer": disclaimer,
    }
