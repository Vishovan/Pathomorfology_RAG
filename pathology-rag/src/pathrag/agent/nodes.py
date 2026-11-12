"""Узлы (Nodes) графа LangGraph."""
import logging
from typing import Any

from ..retrieval.external_search import search_arxiv, search_pubmed, search_web
from ..retrieval.hybrid_retriever import retrieve
from ..retrieval.image_retriever import search_pathology_images
from ..retrieval.legal_kb import get_medical_device_boundary_info, search_gost_standards
from .llm import generate_with_llm
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
    """Узел синтеза итогового структурированного ответа с цитатами через LLM или экспертную базу."""
    intent = state.get("intent", "pathology_qa")
    query = state.get("query", "")
    citations = []

    llm_provider = state.get("llm_provider") or "auto"
    llm_key = state.get("llm_api_key")
    llm_endpoint = state.get("llm_endpoint")
    llm_model = state.get("llm_model")

    use_llm_flag = state.get("use_llm")
    if use_llm_flag is None:
        use_llm_flag = bool(state.get("llm_provider") and state.get("llm_provider") != "autonomous")
    if state.get("llm_provider") == "autonomous":
        use_llm_flag = False

    # 1. Формирование юридического ответа
    if intent == "legal_gost":
        legal_res = state.get("legal_results", [])
        boundary = state.get("legal_boundary", get_medical_device_boundary_info())

        for item in legal_res[:5]:
            citations.append({"source": item.get("gost", "ГОСТ"), "title": item.get("title", ""), "reference": "Каталог Росстандарта"})

        # Попытка генерации через LLM
        legal_context = "\n".join([
            f"Стандарт/Процедура: {s.get('gost')} - {s.get('title')}\nСуть: {s.get('summary')}\nПрактическое применение: {s.get('practical_value')}"
            for s in legal_res[:6]
        ])
        legal_prompt = (
            f"Вопрос пользователя:\n{query}\n\n"
            f"Нормативная база (ГОСТы, процедуры Роспатента, Минцифры, Росздравнадзора):\n{legal_context}\n\n"
            f"Стратегия квалификации Non-SaMD:\n{boundary.get('pathology_rag_strategy', '')}\n\n"
            f"Сформируйте подробный, юридически безупречный ответ по российским стандартам ГОСТ и процедурам регистрации. "
            f"Объясните различия между Роспатентом, Минцифры и Росздравнадзором (медизделием), и как правильно составить ТЗ."
        )

        llm_response = None
        if use_llm_flag:
            llm_response = generate_with_llm(
                legal_prompt,
                system_prompt="Вы — ведущий юрисконсульт по стандартам ГОСТ и регуляторике медицинского программного обеспечения в Российской Федерации.",
                provider=llm_provider,
                api_key=llm_key,
                endpoint_url=llm_endpoint,
                model_name=llm_model,
            )

        if llm_response:
            answer_text = llm_response
        else:
            # Детерминированный структурированный синтез
            lines = [
                "### ⚖️ Консультация по ГОСТам и регуляторике ПО в сфере здравоохранения\n",
                f"**Запрос:** {query}\n",
            ]

            if any(w in query.lower() for w in ["медиздели", "samd", "роспатент", "минцифры", "росздравнадзор", "процедур"]):
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
                    lines.append(f"##### 📌 {item.get('gost')}: {item.get('title')}")
                    lines.append(f"- **Назначение:** {item.get('summary')}")
                    lines.append(f"- **Практическое применение в проекте:** {item.get('practical_value')}\n")

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
            lines.append("По данному запросу не найдено подходящих микропрепаратов в базе данных.")
        else:
            for i, img in enumerate(images, 1):
                title = img.get("title", "Микропрепарат")
                diag = img.get("diagnosis", "")
                organ = img.get("organ", "")
                stain = img.get("stain", "")
                mag = img.get("magnification", "")
                cap = img.get("caption", "")
                src_name = img.get("source_name") or img.get("source_id", "")
                wsi_url = img.get("wsi_url") or img.get("image_url", "")

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

        # Формируем список цитат
        for doc in docs[:4]:
            citations.append({"source": doc.get("source", ""), "title": doc.get("title", ""), "reference": doc.get("page_reference", "")})

        # Попытка вызова LLM
        context_parts = []
        for d in docs[:4]:
            context_parts.append(f"Документ: {d.get('title')}\nИсточник: {d.get('source')} ({d.get('page_reference')})\nСодержание:\n{d.get('text')}")
        clinical_context = "\n\n---\n\n".join(context_parts)

        clinical_prompt = (
            f"Вопрос пользователя:\n{query}\n\n"
            f"Клинический и патоморфологический контекст из верифицированных руководств:\n{clinical_context}\n\n"
            f"Сформулируйте развернутый, профессиональный ответ врача-патологоанатома. "
            f"Выделите ключевые диагностические критерии, процентные пороги, шкалы оценки и правила дифференциальной диагностики. "
            f"Обязательно сошлитесь на клинические рекомендации и стандарты из контекста."
        )

        llm_response = None
        if use_llm_flag:
            llm_response = generate_with_llm(
                clinical_prompt,
                system_prompt="Вы — ведущий врач-патологоанатом, эксперт по цифровой патоморфологии и иммуногистохимии.",
                provider=llm_provider,
                api_key=llm_key,
                endpoint_url=llm_endpoint,
                model_name=llm_model,
            )

        if llm_response:
            answer_text = llm_response
        else:
            # Высокоструктурированный экспертный клинический синтез
            lines = [
                "### 🔬 Клинико-морфологическое экспертное заключение\n",
                f"**Клинический запрос:** {query}\n",
            ]
            if docs:
                lines.append("#### 1. Диагностические критерии и пороговые значения\n")
                for doc in docs[:3]:
                    title = doc.get("title", "")
                    text = doc.get("text", "")
                    src = doc.get("source", "")
                    ref = doc.get("page_reference", "")
                    lines.append(f"**{title}**:")
                    lines.append(f"{text}\n")
                    lines.append(f"> *Источник: {src} ({ref})*\n")

                lines.append("#### 2. Рекомендации по дифференциальной диагностике и рефлекс-тестированию")
                if "her2" in query.lower():
                    lines.append("- При получении неопределенного статуса **HER2 2+** обязательным этапом является подтверждающая **FISH/SISH гибридизация** in situ.")
                    lines.append("- Опухоли со статусом **HER2 1+** или **HER2 2+ / FISH-** классифицируются как **HER2-low** с возможностью назначения современных конъюгатов (T-DXd).")
                elif "ki-67" in query.lower() or "люминальн" in query.lower():
                    lines.append("- Подсчет индекса **Ki-67** выполняется стандартизированным методом в «горячих точках» (hot spots) с оценкой не менее 500-1000 опухолевых клеток.")
                    lines.append("- Значение **< 14-20%** свидетельствует о Люминальном А подтипе, значение **≥ 20%** указывает на Люминальный В подтип с высокой пролиферативной активностью.")
                else:
                    lines.append("- Окончательная верификация требует сопоставления гистологической картины (окраска гематоксилином и эозином) с панелью специфических ИГХ-маркеров и клинико-инструментальными данными.")
            else:
                lines.append("В локальной базе знаний точных фрагментов не обнаружено. Выполнен поиск по открытым научным источникам.\n")

            if lit or web:
                lines.append("\n#### 3. Научные публикации и внешние источники:")
                for item in lit[:3]:
                    lines.append(f"- [{item.get('title')}]({item.get('url')}) — *{item.get('source_type')}*")
                for item in web[:2]:
                    lines.append(f"- [{item.get('title')}]({item.get('url')}) — *Веб-источник*")

            answer_text = "\n".join(lines)

        disclaimer = MED_DISCLAIMER

    return {
        "final_answer": answer_text,
        "citations": citations,
        "disclaimer": disclaimer,
    }
