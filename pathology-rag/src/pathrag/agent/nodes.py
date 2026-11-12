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
    """Узел синтеза итогового структурированного ответа из чанков через LLM или верифицированный ретривер."""
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

    llm_used_label = None
    is_llm_generated = False

    # 1. Формирование юридического ответа (ГОСТы, регуляторика ПО, SaMD)
    if intent == "legal_gost":
        legal_res = state.get("legal_results", [])
        boundary = state.get("legal_boundary", get_medical_device_boundary_info())

        for item in legal_res[:5]:
            citations.append({
                "source": item.get("gost", "ГОСТ"),
                "title": item.get("title", ""),
                "reference": "Каталог Росстандарта"
            })

        legal_context = "\n\n".join([
            f"[Стандарт {i}] {s.get('gost')}: {s.get('title')}\n"
            f"Суть: {s.get('summary')}\n"
            f"Практическое применение в ТЗ: {s.get('practical_value')}"
            for i, s in enumerate(legal_res[:6], 1)
        ])

        legal_prompt = (
            f"Вопрос пользователя:\n{query}\n\n"
            f"=== НОРМАТИВНЫЙ КОНТЕКСТ И СТАНДАРТЫ ГОСТ ===\n"
            f"{legal_context}\n"
            f"Стратегия квалификации Non-SaMD:\n{boundary.get('pathology_rag_strategy', '')}\n"
            f"=============================================\n\n"
            f"ИНСТРУКЦИЯ ДЛЯ ЮРИСКОНСУЛЬТА:\n"
            f"1. Сформулируйте развернутый, юридически выверенный ответ по российскому законодательству и стандартам ГОСТ.\n"
            f"2. Синтезируйте данные из приведенных выше стандартов. Четко разграничьте три самостоятельные процедуры:\n"
            f"   - Регистрация программы для ЭВМ в Роспатенте (ФИПС);\n"
            f"   - Включение в Единый реестр отечественного ПО Минцифры России;\n"
            f"   - Регистрация медицинского изделия (SaMD) в Росздравнадзоре.\n"
            f"3. Разъясните, как сформулировать ТЗ по ГОСТ 34.602-2020 и назначение системы, чтобы сохранить статус Non-SaMD (справочно-образовательная система).\n"
            f"4. Ссылайтесь на применимые стандарты ГОСТ с номерами [1], [2]..."
        )

        llm_response = None
        if use_llm_flag:
            res_meta = generate_with_llm(
                legal_prompt,
                system_prompt="Вы — ведущий юрисконсульт по стандартам ГОСТ и регуляторике медицинского программного обеспечения в Российской Федерации.",
                provider=llm_provider,
                api_key=llm_key,
                endpoint_url=llm_endpoint,
                model_name=llm_model,
                return_meta=True,
            )
            if isinstance(res_meta, tuple):
                llm_response, llm_used_label = res_meta
            else:
                llm_response = res_meta

        if llm_response:
            answer_text = llm_response
            is_llm_generated = True
        else:
            # Алгоритмический юридический синтез
            llm_used_label = "Автономный нормативный синтез"
            lines = [
                "### ⚖️ Консультация по ГОСТам и регуляторике ПО в сфере здравоохранения\n",
                f"**Запрос:** {query}\n",
            ]
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
                for i, item in enumerate(legal_res[:5], 1):
                    lines.append(f"##### 📌 [{i}] {item.get('gost')}: {item.get('title')}")
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

    # 3. Разговорный режим (приветствие, знакомство, возможности)
    elif intent == "conversational":
        conv_prompt = (
            f"Пользователь написал: \"{query}\".\n\n"
            f"ИНСТРУКЦИЯ:\n"
            f"1. Ответьте вежливо и профессионально от лица специализированной интеллектуальной системы поддержки принятия врачебных решений по патоморфологии (Pathology RAG Copilot).\n"
            f"2. Кратко перечислите ключевые клинические направления, по которым вы помогаете:\n"
            f"   - 🔬 Верификация критериев ИГХ при раке молочной железы (шкала HER2 0, 1+, 2+, 3+, FISH/SISH, Ki-67 hot spots, Allred);\n"
            f"   - 📖 Диагностика патологических процессов общей патоморфологии (некрозы, воспаление, дистрофии);\n"
            f"   - 🖼️ Поиск и анализ цифровых гистологических сканов (Whole Slide Images);\n"
            f"   - ⚖️ Режим юриста: разработка ТЗ по ГОСТ 34.602, разграничение Роспатента, Минцифры и Росздравнадзора (Non-SaMD).\n"
            f"3. Предложите задать клинический вопрос или воспользоваться быстрыми сценариями."
        )
        llm_response = None
        if use_llm_flag:
            res_meta = generate_with_llm(
                conv_prompt,
                system_prompt="Вы — вежливый клинический AI-ассистент врача-патологоанатома (Pathology RAG Copilot).",
                provider=llm_provider,
                api_key=llm_key,
                endpoint_url=llm_endpoint,
                model_name=llm_model,
                return_meta=True,
            )
            if isinstance(res_meta, tuple):
                llm_response, llm_used_label = res_meta
            else:
                llm_response = res_meta

        if llm_response:
            answer_text = llm_response
            is_llm_generated = True
        else:
            llm_used_label = "Системное приветствие"
            answer_text = (
                "Здравствуйте! Я — интеллектуальный клинический ассистент врача-патологоанатома (**Pathology RAG Copilot**).\n\n"
                "**Ключевые направления помощи:**\n"
                "- 🔬 **Рак молочной железы (РМЖ):** интерпретация статуса HER2 (0, 1+, 2+ equivocal, 3+), подтверждающее ISH/FISH тестирование, индекс пролиферации Ki-67 и фенотип HER2-low;\n"
                "- 📖 **Общая патоморфология:** стадии крупозной пневмонии (серое/красное опеченение), казеозный некроз, жировой гепатоз;\n"
                "- 🖼️ **Микропрепараты и Whole Slide Images:** просмотр гистологических срезов высокого разрешения с прямым доступом в цифровой вьюер;\n"
                "- ⚖️ **Режим юриста:** ГОСТ 34.602, ГОСТ Р 56939, разграничение процедур Роспатента, Минцифры и Росздравнадзора (Non-SaMD).\n\n"
                "Задайте вопрос в поле ввода ниже или нажмите на один из готовых сценариев."
            )
        disclaimer = "Справочно-образовательная система для патоморфологов."

    # 4. Формирование ответа по патоморфологии (РМЖ / ИГХ / общая патология)
    else:
        docs = state.get("retrieved_docs", [])
        images = state.get("retrieved_images", [])
        lit = state.get("literature_results", [])
        web = state.get("web_results", [])

        # Формируем список цитат
        for doc in docs[:4]:
            citations.append({
                "source": doc.get("source", ""),
                "title": doc.get("title", ""),
                "reference": doc.get("page_reference", "")
            })

        # Формируем структурированный контекст из чанков с явными метками источников
        if docs:
            context_parts = []
            for i, d in enumerate(docs[:4], 1):
                src_label = f"{d.get('source')} ({d.get('page_reference')})" if d.get('page_reference') else d.get('source', '')
                context_parts.append(
                    f"[ЧАНК {i}]\n"
                    f"Название документа: {d.get('title')}\n"
                    f"Источник: {src_label}\n"
                    f"Текст:\n{d.get('text')}"
                )
            clinical_context = "\n\n---\n\n".join(context_parts)

            clinical_prompt = (
                f"Вопрос пользователя:\n{query}\n\n"
                f"=== ФРАГМЕНТЫ БАЗЫ ЗНАНИЙ (ЧАНКИ ИЗ ВЕРИФИЦИРОВАННЫХ РУКОВОДСТВ) ===\n"
                f"{clinical_context}\n"
                f"====================================================================\n\n"
                f"ИНСТРУКЦИЯ ДЛЯ ВРАЧА-ПАТОЛОГОАНАТОМА:\n"
                f"1. Сформулируйте профессиональный, логичный и клинически выверенный ответ на вопрос пользователя.\n"
                f"2. ВАЖНО: Ваш ответ должен быть СИНТЕЗИРОВАН ИЗ ПРЕДОСТАВЛЕННЫХ ЧАНКОВ. Сопоставьте информацию из разных фрагментов: выделите ключевые диагностические критерии, процентные пороги (0, 1+, 2+, 3+ или отсечки Ki-67), правила дифференциальной диагностики и алгоритмы подтверждающего тестирования (FISH, SISH).\n"
                f"3. Не копируйте текст чанков целиком — проведите глубокий экспертный синтез и обобщение.\n"
                f"4. Обязательно сошлитесь в тексте на номера источников [1], [2], подтверждающие каждое диагностическое утверждение.\n"
                f"5. Оформите ответ в Markdown со структурированными подразделами, списками и выделением ключевых терминов жирным шрифтом."
            )
        else:
            # Чанков в локальной базе нет — используем литературу / веб или клинические знания LLM
            ext_context_parts = []
            for it in lit[:3]:
                ext_context_parts.append(f"Публикация ({it.get('source_type', 'PubMed')}): {it.get('title')}\n{it.get('summary', '')}")
            for it in web[:2]:
                ext_context_parts.append(f"Веб-источник: {it.get('title')}\n{it.get('snippet', '')}")
            ext_str = "\n\n".join(ext_context_parts)
            ext_banner = f"=== МАТЕРИАЛЫ ВНЕШНЕГО НАУЧНОГО ПОИСКА ===\n{ext_str}\n==========================================\n\n" if ext_str else ""

            clinical_prompt = (
                f"Клинический вопрос врача:\n{query}\n\n"
                f"{ext_banner}"
                f"ИНСТРУКЦИЯ ДЛЯ ВРАЧА-ПАТОЛОГОАНАТОМА:\n"
                f"1. Дайте точный, научно обоснованный и структурированный ответ на вопрос по патологической анатомии и гистологии.\n"
                f"2. Подробно опишите патогенез, макро- и микроскопическую картину, дифференциальную диагностику и применяемые окраски.\n"
                f"3. Оформите ответ в Markdown с четкими подразделами, маркированными списками и акцентами на ключевых терминах."
            )

        llm_response = None
        if use_llm_flag:
            res_meta = generate_with_llm(
                clinical_prompt,
                system_prompt="Вы — ведущий врач-патологоанатом, эксперт по цифровой патоморфологии и иммуногистохимии. Вы формируете точные клинические заключения, синтезируя данные из чанков доказательной базы знаний.",
                provider=llm_provider,
                api_key=llm_key,
                endpoint_url=llm_endpoint,
                model_name=llm_model,
                return_meta=True,
            )
            if isinstance(res_meta, tuple):
                llm_response, llm_used_label = res_meta
            else:
                llm_response = res_meta

        if llm_response:
            answer_text = llm_response
            is_llm_generated = True
        else:
            # Динамический экспертный RAG-синтез при отсутствии связи с LLM
            llm_used_label = "Автономный RAG-экстрактор (офлайн)"
            lines = [
                "### 🔬 Клинико-морфологическое заключение (Автономный RAG-синтез)\n",
                f"> ℹ️ *Внешний инференс LLM сейчас недоступен (не указан ключ или превышен лимит). Ответ сформирован прямым алгоритмическим извлечением релевантных данных из базы знаний.*\n",
                f"**Клинический запрос:** {query}\n",
            ]
            if docs:
                lines.append("#### 1. Диагностические критерии и выдержки из верифицированных источников:\n")
                for i, doc in enumerate(docs[:3], 1):
                    title = doc.get("title", "")
                    text = doc.get("text", "")
                    src = doc.get("source", "")
                    ref = doc.get("page_reference", "")
                    ref_str = f" · {ref}" if ref else ""
                    lines.append(f"##### 📌 [{i}] {title} ({src}{ref_str})")
                    lines.append(f"{text}\n")

                lines.append("#### 2. Экспертные рекомендации по тактике верификации:")
                q_low = query.lower()
                if "her2" in q_low:
                    lines.append("- При получении неопределенного статуса **HER2 2+ (equivocal)** обязательным этапом является подтверждающая **FISH / SISH гибридизация** in situ для оценки амплификации гена ERBB2.")
                    lines.append("- Опухоли с ИГХ 1+ или ИГХ 2+ при отсутствии амплификации по FISH классифицируются как **HER2-low**, что открывает возможность таргетной терапии новыми конъюгатами (T-DXd).")
                elif "ki-67" in q_low or "люминальн" in q_low:
                    lines.append("- Индекс пролиферации **Ki-67** оценивается стандартизированным методом в зонах максимальной активности («hot spots») с подсчетом не менее 500-1000 клеток.")
                    lines.append("- Пороговое значение **< 14-20%** разграничивает Люминальный А и Люминальный В подтипы рака молочной железы с различным прогнозом и чувствительностью к химиотерапии.")
                else:
                    lines.append("- Для окончательной верификации диагноза сопоставьте световую микроскопию (окраска H&E) с панелью специфических ИГХ-маркеров и данными анамнеза.")
            else:
                lines.append("В локальной базе знаний точных фрагментов не обнаружено. Выполнен поиск по открытым научным источникам.\n")

            if lit or web:
                lines.append("\n#### 3. Внешние научные публикации и ссылки:")
                for item in lit[:3]:
                    lines.append(f"- [{item.get('title')}]({item.get('url')}) — *{item.get('source_type')}*")
                for item in web[:2]:
                    lines.append(f"- [{item.get('title')}]({item.get('url')}) — *Веб-источник*")

            lines.append("\n---\n💡 **Рекомендация:** Для генерации развернутого ответа нейросетью Llama 3.3 70B укажите бесплатный API-ключ **Groq** в левой панели (получить бесплатно за 30 секунд на [console.groq.com](https://console.groq.com)).")
            answer_text = "\n".join(lines)

        disclaimer = MED_DISCLAIMER

    return {
        "final_answer": answer_text,
        "citations": citations,
        "disclaimer": disclaimer,
        "llm_used": llm_used_label,
        "is_llm_generated": is_llm_generated,
    }
