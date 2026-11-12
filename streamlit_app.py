"""Streamlit Web Application для Pathology RAG (деплой на share.streamlit.io).

Мультимодальный ассистент патоморфолога & консультант по стандартам ГОСТ:
- Рак молочной железы (РМЖ): HER2 0/1+/2+/3+, ISH-алгоритм, Ki-67, Allred ER/PR, Nottingham grade;
- Общая патоморфология и нозологии;
- Поиск микропрепаратов с глубокими ссылками на Whole Slide Images (WSI);
- Режим юриста: Роспатент, Минцифры, Росздравнадзор (Non-SaMD) и ГОСТы (34.602, 59793, 56939-2024 и др.);
- Внешний поиск: PubMed E-utilities, arXiv, bioRxiv, Web.
"""
import os
import sys
from pathlib import Path

# Добавляем пути к модулям (для запуска как из корня, так и из подпапки)
CURRENT_DIR = Path(__file__).resolve().parent
if (CURRENT_DIR / "pathology-rag" / "src").exists():
    sys.path.insert(0, str(CURRENT_DIR / "pathology-rag" / "src"))
    sys.path.insert(0, str(CURRENT_DIR / "pathology-rag"))
elif (CURRENT_DIR / "src").exists():
    sys.path.insert(0, str(CURRENT_DIR / "src"))
    sys.path.insert(0, str(CURRENT_DIR))

import streamlit as st

# Импорт графа агента
try:
    from pathrag.agent import ask_agent
    from pathrag.retrieval.legal_kb import load_legal_data
except ImportError as e:
    st.error(f"Ошибка импорта модулей Pathology RAG: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 1. Настройка страницы
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Pathology RAG | Ассистент патоморфолога и юриста",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Пользовательские CSS стили
st.markdown(
    """
    <style>
    .main-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 1.2rem;
    }
    .badge-status {
        background-color: #ecfdf5;
        color: #065f46;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid #a7f3d0;
    }
    .img-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .wsi-btn {
        display: inline-block;
        background: #2563eb;
        color: white !important;
        padding: 6px 14px;
        border-radius: 8px;
        text-decoration: none;
        font-size: 0.82rem;
        font-weight: 500;
        margin-top: 8px;
    }
    .wsi-btn:hover {
        background: #1d4ed8;
    }
    .disclaimer-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 10px 14px;
        font-size: 0.8rem;
        color: #92400e;
        border-radius: 4px;
        margin-top: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. Боковая панель (Sidebar)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔬 Pathology RAG")
    st.markdown('<span class="badge-status">● LangGraph + LlamaIndex Active</span>', unsafe_allow_html=True)
    st.caption("Мультимодальная справочная система по патоморфологии и стандартам ГОСТ")
    st.divider()

    st.markdown("#### 🎯 Режим работы")
    mode_options = {
        "auto": "🤖 Авто-выбор (LangGraph Router)",
        "breast_cancer": "🔬 РМЖ: ИГХ (HER2 & Ki-67)",
        "pathology": "📖 Общая патоморфология",
        "images": "🖼️ Микропрепараты & WSI",
        "legal": "⚖️ Режим юриста (ГОСТы и SaMD)",
        "literature": "🌐 Внешний поиск (PubMed / arXiv)",
    }
    selected_mode_key = st.radio(
        "Выберите тематический фокус:",
        options=list(mode_options.keys()),
        format_func=lambda x: mode_options[x],
        index=0,
    )

    st.divider()

    st.markdown("#### ⚙️ LLM для коммерциализации")
    llm_choice = st.selectbox(
        "Модель генерации:",
        options=[
            "Базовая экспертная база (Бесплатно, без API-ключей)",
            "Сбер GigaChat (152-ФЗ, коммерческая лицензия в РФ)",
            "Qwen 2.5 (Apache 2.0 open-weights / vLLM / Ollama)",
            "OpenAI-совместимый эндпоинт",
        ],
        index=0,
    )

    api_key_input = None
    if "GigaChat" in llm_choice:
        # Проверяем секреты Streamlit
        default_key = st.secrets.get("GIGACHAT_CREDENTIALS", "") if hasattr(st, "secrets") else ""
        api_key_input = st.text_input("GigaChat Client Secret / Token:", value=default_key, type="password")
        st.caption("Соответствует 152-ФЗ. Сервера в РФ, коммерческое использование разрешено.")
    elif "Qwen" in llm_choice or "OpenAI" in llm_choice:
        default_endpoint = os.getenv("VLLM_ENDPOINT", "http://localhost:11434/v1")
        endpoint_url = st.text_input("API Base URL:", value=default_endpoint)
        api_key_input = st.text_input("API Key (при наличии):", type="password")
        st.caption("Лицензия Apache 2.0: полностью свободна для коммерческих продуктов.")

    st.divider()

    with st.expander("📜 Проектная документация по ГОСТ"):
        st.markdown("""
        - [ТЗ по ГОСТ 34.602-2020](https://github.com/Vishovan/Pathomorfology_RAG/blob/main/pathology-rag/docs/gost/TZ_GOST_34_602_2020.md)
        - [Ведомость документов ГОСТ 34.201](https://github.com/Vishovan/Pathomorfology_RAG/blob/main/pathology-rag/docs/gost/Vedomost_Dokumentov_GOST_34_201_2020.md)
        - [Описание архитектуры ГОСТ Р 59795](https://github.com/Vishovan/Pathomorfology_RAG/blob/main/pathology-rag/docs/gost/Architecture_and_Data_GOST_59795_2021.md)
        - [Обоснование Non-SaMD статуса](https://github.com/Vishovan/Pathomorfology_RAG/blob/main/pathology-rag/docs/gost/Intended_Use_and_Regulatory_Limitations.md)
        """)

    st.markdown(
        """
        <div class="disclaimer-box">
        <strong>Статус Non-SaMD:</strong><br>
        Система разработана как справочно-образовательная. Не обрабатывает персональные данные пациентов и не является медицинским изделием.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# 3. Основная область (Main Area)
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">Pathology RAG Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Интеллектуальная система ответов по патоморфологии (РМЖ ИГХ, нозологии, WSI) и нормативной базе ГОСТ</div>',
    unsafe_allow_html=True,
)

# Быстрые кнопки-примеры
st.markdown("##### 💡 Быстрые примеры запросов:")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔬 HER2 2+ и ISH/FISH (РМЖ)", use_container_width=True):
        st.session_state["preset_query"] = "Какие критерии оценки HER2 2+ при раке молочной железы и когда требуется подтверждающее тестирование FISH?"
with col2:
    if st.button("🔬 Ki-67 и подтипы РМЖ", use_container_width=True):
        st.session_state["preset_query"] = "Как подсчитывается Ki-67 и какие пороговые значения разделяют люминальный А и В подтипы?"
with col3:
    if st.button("🖼️ WSI: серое опеченение", use_container_width=True):
        st.session_state["preset_query"] = "Покажи микропрепарат серого опеченения при крупозной пневмонии с описанием и ссылкой на WSI"
with col4:
    if st.button("⚖️ Юрист: ГОСТы и медизделие", use_container_width=True):
        st.session_state["preset_query"] = "Какие процедуры пройти (Роспатент, Минцифры, Росздравнадзор) и какие ГОСТы нужны для разработки ТЗ?"

# Инициализация истории сообщений
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """Приветствую! Я мультимодальный RAG-ассистент по **патоморфологии** и **нормативной базе ГОСТ**.

Я специализируюсь на:
1. **Раке молочной железы (РМЖ):** критерии оценки HER2/neu (0, 1+, 2+ equivocal, 3+), рефлекс-тестирование FISH/SISH, индекс пролиферации Ki-67 (hot spots, люминальный А/В), Allred ER/PR, Ноттингемская градация, дифдиагноз долькового рака (потеря E-cadherin);
2. **Общей патоморфологии:** пневмонии, дистрофии, атеросклероз, туберкулез, колоректальный рак;
3. **Микропрепаратах и Whole Slide Images (WSI):** показ гистологических снимков с глубокими ссылками на цифровые вьюеры;
4. **Режиме юриста:** разграничение Роспатента, Минцифры и Росздравнадзора (Non-SaMD), разработка ТЗ по ГОСТ 34.602-2020, безопасная разработка ГОСТ Р 56939-2024.

Задайте любой интересующий вас вопрос!""",
            "intent": "system_welcome",
            "images": [],
            "citations": [],
            "disclaimer": "Справочно-образовательная система. Не является медицинским изделием.",
        }
    ]

# Отображение истории сообщений
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍⚕️" if msg["role"] == "user" else "🔬"):
        if msg.get("intent") and msg["role"] == "assistant" and msg.get("intent") != "system_welcome":
            st.caption(f"Маршрут графа LangGraph: `{msg['intent']}`")

        st.markdown(msg["content"])

        # Вывод карточек изображений (если есть)
        images = msg.get("images", [])
        if images:
            st.markdown("#### 📸 Микропрепараты и Whole Slide Images:")
            img_cols = st.columns(min(len(images), 2))
            for i, img in enumerate(images[:4]):
                with img_cols[i % 2]:
                    st.markdown(
                        f"""
                        <div class="img-card">
                            <strong>{img.get('title', 'Препарат')}</strong><br>
                            <small>📍 <strong>Орган:</strong> {img.get('organ', '-')} | <strong>Окраска:</strong> {img.get('stain', 'H&E')} | <strong>Увеличение:</strong> {img.get('magnification', '-')}</small>
                            <p style="font-size:0.82rem; color:#475569; margin: 6px 0;">{img.get('caption', '')}</p>
                            <small style="color:#64748b;">Источник: {img.get('source_name', 'Атлас')}</small><br>
                            {f'<a href="{img.get("wsi_url")}" target="_blank" class="wsi-btn">🔬 Открыть Whole Slide Image (WSI)</a>' if img.get("wsi_url") else ''}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Вывод источников и цитат
        citations = msg.get("citations", [])
        if citations:
            with st.expander(f"📚 Источники и цитирование ({len(citations)})"):
                for c in citations:
                    st.markdown(f"- **{c.get('source')}**: {c.get('title')} (*{c.get('reference')}*)")

        # Вывод дисклеймера
        if msg.get("disclaimer"):
            st.markdown(
                f'<div class="disclaimer-box">{msg["disclaimer"]}</div>',
                unsafe_allow_html=True,
            )

# -----------------------------------------------------------------------------
# 4. Обработка ввода пользователя
# -----------------------------------------------------------------------------
prompt = st.chat_input("Задайте вопрос по патоморфологии, микропрепаратам или ГОСТам...")

# Если была нажата кнопка с быстрым примером
if "preset_query" in st.session_state and st.session_state["preset_query"]:
    prompt = st.session_state["preset_query"]
    st.session_state["preset_query"] = None

if prompt:
    # 1. Добавляем сообщение пользователя
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍⚕️"):
        st.markdown(prompt)

    # 2. Обработка через граф LangGraph
    with st.chat_message("assistant", avatar="🔬"):
        with st.spinner("Оркестрация в графе LangGraph (роутер -> ретривер -> грейдер -> генерация)..."):
            active_mode = None if selected_mode_key == "auto" else selected_mode_key

            # Определение LLM параметров
            llm_provider = None
            if "GigaChat" in llm_choice:
                llm_provider = "gigachat"
            elif "Qwen" in llm_choice:
                llm_provider = "qwen_local"
            elif "OpenAI" in llm_choice:
                llm_provider = "openai_compat"

            result = ask_agent(
                prompt,
                mode=active_mode,
                llm_provider=llm_provider,
                llm_api_key=api_key_input,
                use_llm=bool(llm_provider),
            )

            intent = result.get("intent", "pathology_qa")
            answer = result.get("final_answer", "")
            citations = result.get("citations", [])
            disclaimer = result.get("disclaimer", "")
            images = result.get("retrieved_images", [])

            st.caption(f"Маршрут графа LangGraph: `{intent}`")
            st.markdown(answer)

            if images:
                st.markdown("#### 📸 Микропрепараты и Whole Slide Images:")
                img_cols = st.columns(min(len(images), 2))
                for i, img in enumerate(images[:4]):
                    with img_cols[i % 2]:
                        st.markdown(
                            f"""
                            <div class="img-card">
                                <strong>{img.get('title', 'Препарат')}</strong><br>
                                <small>📍 <strong>Орган:</strong> {img.get('organ', '-')} | <strong>Окраска:</strong> {img.get('stain', 'H&E')} | <strong>Увеличение:</strong> {img.get('magnification', '-')}</small>
                                <p style="font-size:0.82rem; color:#475569; margin: 6px 0;">{img.get('caption', '')}</p>
                                <small style="color:#64748b;">Источник: {img.get('source_name', 'Атлас')}</small><br>
                                {f'<a href="{img.get("wsi_url")}" target="_blank" class="wsi-btn">🔬 Открыть Whole Slide Image (WSI)</a>' if img.get("wsi_url") else ''}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            if citations:
                with st.expander(f"📚 Источники и цитирование ({len(citations)})"):
                    for c in citations:
                        st.markdown(f"- **{c.get('source')}**: {c.get('title')} (*{c.get('reference')}*)")

            if disclaimer:
                st.markdown(
                    f'<div class="disclaimer-box">{disclaimer}</div>',
                    unsafe_allow_html=True,
                )

            # Сохранение ответа в историю
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "intent": intent,
                "images": images,
                "citations": citations,
                "disclaimer": disclaimer,
            })
