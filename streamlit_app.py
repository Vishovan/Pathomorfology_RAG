"""Streamlit Web Application для Pathology RAG (деплой на share.streamlit.io).

Ультрасовременный минималистичный интерфейс в темной теме (Dark Obsidian Theme):
- Темная стилистика Linear / Vercel (глубокий графит, мягкий неоновый акцент, frosted cards);
- Прямой вывод микропрепаратов через st.image (без битых редиректов на GitHub);
- Активная загрузка коммерческих и открытых LLM:
    * Сбер GigaChat Pro (152-ФЗ, официальная лицензия для РФ);
    * Groq Cloud (сверхбыстрый бесплатный инференс Qwen 2.5 / Llama 3.3);
    * OpenRouter / DeepSeek / vLLM / Ollama;
    * Free Cloud AI (Pollinations Qwen 2.5 без ключей);
    * Автономный клинический анализатор;
- Специализированный RAG по РМЖ (HER2 0/1+/2+/3+, FISH, Ki-67, Allred, Nottingham);
- Режим юриста: Роспатент, Минцифры, Росздравнадзор (Non-SaMD) и 12 стандартов ГОСТ.
"""
import os
import sys
from pathlib import Path

# Добавляем пути к модулям системы
CURRENT_DIR = Path(__file__).resolve().parent
if (CURRENT_DIR / "pathology-rag" / "src").exists():
    sys.path.insert(0, str(CURRENT_DIR / "pathology-rag" / "src"))
    sys.path.insert(0, str(CURRENT_DIR / "pathology-rag"))
elif (CURRENT_DIR / "src").exists():
    sys.path.insert(0, str(CURRENT_DIR / "src"))
    sys.path.insert(0, str(CURRENT_DIR))

import streamlit as st

# Импорт графа агента и LLM утилит
try:
    from pathrag.agent import ask_agent
    from pathrag.agent.llm import test_llm_connection
except ImportError as e:
    st.error(f"Ошибка импорта модулей Pathology RAG: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 1. Настройка страницы и ультрасовременный темный стиль (Obsidian Minimalist)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Pathology RAG | Clinical & Regulatory Copilot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --bg-dark: #0A0D14;
        --card-bg: #111622;
        --sidebar-bg: #0D111A;
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-hover: rgba(59, 130, 246, 0.4);
        --accent-blue: #3B82F6;
        --accent-blue-glow: rgba(59, 130, 246, 0.25);
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
    }

    html, body, [class*="css"], .stApp {
        background-color: var(--bg-dark) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    /* Убираем лишние верхние и нижние отступы Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1140px;
    }

    /* Боковая панель в темном стиле */
    section[data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }

    /* Верхний блок заголовка */
    .hero-header {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        border-bottom: 1px solid var(--border-subtle);
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
        gap: 0.75rem;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #FFFFFF 0%, #CBD5E1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-subtitle {
        margin: 4px 0 0 0;
        color: var(--text-secondary);
        font-size: 0.9rem;
    }

    /* Бейджи с неоновым свечением */
    .hero-badges {
        display: flex;
        gap: 8px;
        align-items: center;
        flex-wrap: wrap;
    }
    .badge-chip {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.73rem;
        font-weight: 500;
        letter-spacing: 0.02em;
    }
    .badge-emerald {
        background: rgba(16, 185, 129, 0.1);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .badge-blue {
        background: rgba(59, 130, 246, 0.1);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.25);
    }
    .badge-purple {
        background: rgba(168, 85, 247, 0.1);
        color: #C084FC;
        border: 1px solid rgba(168, 85, 247, 0.25);
    }

    /* Кнопки быстрых сценариев */
    .stButton > button {
        border-radius: 10px;
        font-size: 0.83rem;
        font-weight: 500;
        border: 1px solid var(--border-subtle);
        background: var(--card-bg);
        color: #E2E8F0;
        padding: 10px 14px;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        width: 100%;
        text-align: left;
    }
    .stButton > button:hover {
        border-color: var(--accent-blue);
        color: #FFFFFF;
        background: #172033;
        box-shadow: 0 0 16px var(--accent-blue-glow);
        transform: translateY(-1px);
    }

    /* Карточки гистологических снимков в темном оформлении */
    .micro-card-dark {
        background: var(--card-bg);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 14px 16px;
        margin-top: 10px;
        margin-bottom: 14px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    .card-meta-bar {
        display: flex;
        gap: 6px;
        margin: 8px 0 10px 0;
        flex-wrap: wrap;
    }
    .meta-tag-dark {
        background: rgba(255, 255, 255, 0.05);
        color: #94A3B8;
        border: 1px solid rgba(255, 255, 255, 0.08);
        font-size: 0.72rem;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 500;
    }
    .wsi-action-btn-dark {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: #FFFFFF !important;
        text-decoration: none;
        padding: 7px 14px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 10px;
        box-shadow: 0 2px 10px rgba(37, 99, 235, 0.35);
        transition: all 0.15s ease-in-out;
    }
    .wsi-action-btn-dark:hover {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.5);
        transform: translateY(-1px);
        text-decoration: none;
    }

    /* Дисклеймер Non-SaMD в темной теме */
    .regulatory-notice-dark {
        background: rgba(245, 158, 11, 0.06);
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-left: 3px solid #F59E0B;
        padding: 10px 14px;
        border-radius: 8px;
        font-size: 0.78rem;
        color: #FCD34D;
        line-height: 1.45;
        margin-top: 14px;
    }

    /* Стилизация сообщений чата */
    div[data-testid="stChatMessage"] {
        background-color: transparent !important;
        padding: 0.8rem 0;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. Боковая панель: Конфигурация модели и режима работы
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔬 **Pathology RAG**")
    st.caption("Клинический ассистент и консультант по ГОСТам")
    st.markdown('<span class="badge-chip badge-emerald">● LangGraph Multi-Agent</span>', unsafe_allow_html=True)
    st.divider()

    st.markdown("#### ⚙️ **Модель генерации (LLM)**")

    llm_options_map = {
        "gigachat": "🟢 Сбер GigaChat Pro (152-ФЗ, РФ)",
        "groq": "⚡ Groq Cloud (Qwen 2.5 / Llama 3.3)",
        "free_cloud": "🌐 Free Cloud AI (Pollinations Qwen 2.5)",
        "openai_compat": "🖥️ Корпоративный vLLM / Ollama / OpenAI",
        "autonomous": "🔬 Автономный экспертный RAG-анализ",
    }

    selected_llm_key = st.selectbox(
        "Выберите провайдера интеллекта:",
        options=list(llm_options_map.keys()),
        format_func=lambda k: llm_options_map[k],
        index=0,
    )

    api_key_val = None
    custom_endpoint = None
    custom_model_name = None

    if selected_llm_key == "gigachat":
        default_secret = ""
        if hasattr(st, "secrets") and "GIGACHAT_CREDENTIALS" in st.secrets:
            default_secret = st.secrets["GIGACHAT_CREDENTIALS"]
        elif os.getenv("GIGACHAT_CREDENTIALS"):
            default_secret = os.getenv("GIGACHAT_CREDENTIALS", "")

        api_key_val = st.text_input(
            "GigaChat Client Secret / Токен:",
            value=default_secret,
            type="password",
            placeholder="Вставьте токен Сбера...",
            help="152-ФЗ, серверы в РФ. Для коммерциализации и медицинских систем в РФ.",
        )
        if not api_key_val:
            st.info("💡 Токен можно ввести здесь или сохранить в Streamlit Secrets (`GIGACHAT_CREDENTIALS`).")

    elif selected_llm_key == "groq":
        default_secret = ""
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            default_secret = st.secrets["GROQ_API_KEY"]
        elif os.getenv("GROQ_API_KEY"):
            default_secret = os.getenv("GROQ_API_KEY", "")

        api_key_val = st.text_input(
            "Groq API Key (Бесплатно на console.groq.com):",
            value=default_secret,
            type="password",
            placeholder="gsk_...",
        )
        custom_model_name = st.selectbox(
            "Модель на Groq:",
            options=["llama-3.3-70b-versatile", "qwen-2.5-32b", "mixtral-8x7b-32768"],
            index=0,
        )

    elif selected_llm_key == "free_cloud":
        st.success("✨ Работает онлайн без API-ключей (Qwen 2.5 via Cloud Serverless).")

    elif selected_llm_key == "openai_compat":
        custom_endpoint = st.text_input("Base URL:", value="http://localhost:11434/v1")
        api_key_val = st.text_input("API Key / Token (если есть):", type="password")
        custom_model_name = st.text_input("Model ID:", value="qwen2.5:14b-instruct")

    elif selected_llm_key == "autonomous":
        st.info("📌 Работает на встроенной верифицированной базе знаний клинических руководств без внешних запросов.")

    # Кнопка проверки связи с моделью в реальном времени
    if selected_llm_key != "autonomous":
        if st.button("🧪 Проверить связь с LLM", use_container_width=True):
            with st.spinner("Тестирование соединения с моделью..."):
                ok, msg = test_llm_connection(
                    provider=selected_llm_key,
                    api_key=api_key_val,
                    endpoint_url=custom_endpoint,
                    model_name=custom_model_name,
                )
                if ok:
                    st.success(f"✅ {msg}")
                else:
                    st.warning(f"⚠️ {msg}")

    st.divider()

    st.markdown("#### 🎯 **Режим запроса**")
    mode_options = {
        "auto": "🤖 Авто-маршрутизация LangGraph",
        "breast_cancer": "🔬 РМЖ: ИГХ (HER2 & Ki-67)",
        "pathology": "📖 Общая патоморфология",
        "images": "🖼️ Микропрепараты & WSI",
        "legal": "⚖️ Юрист (ГОСТы, Роспатент, SaMD)",
        "literature": "🌐 Научный поиск (PubMed/arXiv)",
    }
    chosen_mode = st.radio(
        "Направление анализа:",
        options=list(mode_options.keys()),
        format_func=lambda k: mode_options[k],
        index=0,
    )

    st.divider()

    with st.expander("📜 Нормативные документы ГОСТ"):
        st.markdown("""
        - [ТЗ по ГОСТ 34.602-2020](https://github.com/Vishovan/Pathomorfology_RAG/blob/main/pathology-rag/docs/gost/TZ_GOST_34_602_2020.md)
        - [Ведомость документов ГОСТ 34.201](https://github.com/Vishovan/Pathomorfology_RAG/blob/main/pathology-rag/docs/gost/Vedomost_Dokumentov_GOST_34_201_2020.md)
        - [Архитектура ПО ГОСТ Р 59795](https://github.com/Vishovan/Pathomorfology_RAG/blob/main/pathology-rag/docs/gost/Architecture_and_Data_GOST_59795_2021.md)
        - [Обоснование Non-SaMD статуса](https://github.com/Vishovan/Pathomorfology_RAG/blob/main/pathology-rag/docs/gost/Intended_Use_and_Regulatory_Limitations.md)
        """)

    st.markdown(
        """
        <div class="regulatory-notice-dark">
        <strong>Статус Non-SaMD:</strong><br>
        Система разработана для академических и справочно-образовательных целей и не классифицируется как медицинское изделие.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# 3. Основная рабочая область
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-header">
        <div>
            <h1 class="hero-title">Pathology RAG</h1>
            <p class="hero-subtitle">Мультимодальный ассистент по патоморфологии (РМЖ ИГХ, WSI) и стандартам ГОСТ</p>
        </div>
        <div class="hero-badges">
            <span class="badge-chip badge-purple">ГОСТ 34.602-2020</span>
            <span class="badge-chip badge-blue">ASCO/CAP & RUSSCO</span>
            <span class="badge-chip badge-emerald">Non-SaMD</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Быстрые клинические и юридические сценарии в 1 клик
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔬 HER2 2+ & FISH", help="Критерии оценки и алгоритм рефлекс-тестирования"):
        st.session_state["active_preset"] = "Какие критерии оценки HER2 2+ при раке молочной железы и когда требуется подтверждающее тестирование FISH?"
with col2:
    if st.button("📊 Ki-67 & Люминальные типы", help="Подсчет пролиферации и разделение Люминального А и В"):
        st.session_state["active_preset"] = "Как подсчитывается Ki-67 и какие пороговые значения разделяют люминальный А и В подтипы?"
with col3:
    if st.button("🖼️ Слайд: серое опеченение", help="Микропрепарат крупозной пневмонии высокого разрешения"):
        st.session_state["active_preset"] = "Покажи микропрепарат серого опеченения при крупозной пневмонии"
with col4:
    if st.button("⚖️ Режим юриста: ГОСТ и SaMD", help="Процедуры Роспатента, Минцифры и Росздравнадзора"):
        st.session_state["active_preset"] = "Какие процедуры пройти (Роспатент, Минцифры, Росздравнадзор) и какие ГОСТы нужны для разработки ТЗ?"


def resolve_img_source(img_dict: dict) -> str | None:
    """Безопасно разрешает локальный путь или прямой веб-URL без редиректов на гит."""
    loc = img_dict.get("local_path")
    if loc:
        p1 = CURRENT_DIR / loc
        if p1.exists():
            return str(p1)
        p2 = CURRENT_DIR / "pathology-rag" / loc
        if p2.exists():
            return str(p2)
    return img_dict.get("image_url")


# История сообщений чата
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """Приветствую! Я мультимодальный RAG-ассистент по **патоморфологии** и **нормативной базе ГОСТ**.

Готов ответить на вопросы по:
- **Раку молочной железы (РМЖ):** шкала HER2/neu (0, 1+, 2+ equivocal, 3+), фенотип HER2-low, подтверждающее тестирование FISH/SISH, индекс пролиферации Ki-67 (hot spots, отсечки Люминального А vs В), гормональные рецепторы ER/PR по Allred, дифференциальная диагностика долькового рака (E-cadherin);
- **Общей патоморфологии:** стадии крупозной пневмонии (серое/красное опеченение), жировой гепатоз, атеросклеротическая бляшка;
- **Микропрепаратам и Whole Slide Images:** просмотр гистологических срезов с прямой ссылкой в цифровой вьюер;
- **Режиму юриста:** разграничение процедур Роспатента, Минцифры и Росздравнадзора (Non-SaMD), разработка ТЗ по ГОСТ 34.602-2020 и безопасная разработка по ГОСТ Р 56939-2024.

Выберите готовый сценарий выше или задайте вопрос ниже.""",
            "intent": "system_welcome",
            "images": [],
            "citations": [],
            "disclaimer": "Справочно-образовательная система. Не является медицинским изделием.",
        }
    ]

# Отрисовка ленты диалога
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍⚕️" if msg["role"] == "user" else "🔬"):
        if msg.get("intent") and msg["role"] == "assistant" and msg.get("intent") != "system_welcome":
            st.markdown(f'<span class="badge-chip badge-blue">Маршрут: {msg["intent"]}</span>', unsafe_allow_html=True)

        st.markdown(msg["content"])

        # Отрисовка микропрепаратов
        images = msg.get("images", [])
        if images:
            st.markdown("##### 📸 Микропрепараты и гистологические срезы:")
            cols = st.columns(min(len(images), 2))
            for i, img in enumerate(images[:4]):
                with cols[i % 2]:
                    src = resolve_img_source(img)
                    if src:
                        try:
                            st.image(src, caption=img.get("title", "Микропрепарат"), use_container_width=True)
                        except Exception:
                            if img.get("image_url"):
                                st.image(img.get("image_url"), caption=img.get("title", "Микропрепарат"), use_container_width=True)

                    wsi_link = img.get("wsi_url") or img.get("image_url") or ""
                    st.markdown(
                        f"""
                        <div class="micro-card-dark">
                            <strong style="color: #F8FAFC;">{img.get('title', 'Препарат')}</strong>
                            <div class="card-meta-bar">
                                <span class="meta-tag-dark">📍 {img.get('organ', '-')}</span>
                                <span class="meta-tag-dark">🔬 {img.get('stain', 'H&E')}</span>
                                <span class="meta-tag-dark">🔍 {img.get('magnification', '-')}</span>
                            </div>
                            <p style="font-size:0.8rem; color:#94A3B8; margin:4px 0 8px 0; line-height:1.4;">{img.get('caption', '')}</p>
                            <small style="color:#64748B;">Источник: {img.get('source_name', 'Атлас')}</small><br>
                            {f'<a href="{wsi_link}" target="_blank" rel="noopener noreferrer" class="wsi-action-btn-dark">🔬 Открыть WSI в полном разрешении ↗</a>' if wsi_link else ''}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Вывод источников
        citations = msg.get("citations", [])
        if citations:
            with st.expander(f"📚 Верифицированные источники и цитаты ({len(citations)})"):
                for c in citations:
                    st.markdown(f"- **{c.get('source')}**: {c.get('title')} (*{c.get('reference')}*)")

        # Вывод регуляторного дисклеймера
        if msg.get("disclaimer"):
            st.markdown(f'<div class="regulatory-notice-dark">{msg["disclaimer"]}</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. Обработка запросов пользователя
# -----------------------------------------------------------------------------
prompt_input = st.chat_input("Задайте вопрос по патоморфологии, иммуногистохимии, WSI или стандартам ГОСТ...")

# Подстановка предустановленного вопроса по клику
if "active_preset" in st.session_state and st.session_state["active_preset"]:
    prompt_input = st.session_state["active_preset"]
    st.session_state["active_preset"] = None

if prompt_input:
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user", avatar="🧑‍⚕️"):
        st.markdown(prompt_input)

    with st.chat_message("assistant", avatar="🔬"):
        with st.spinner("Оркестрация в LangGraph (маршрутизация -> ретривер -> верификация -> генерация)..."):
            active_m = None if chosen_mode == "auto" else chosen_mode
            use_llm_flag = (selected_llm_key != "autonomous")

            try:
                result = ask_agent(
                    prompt_input,
                    mode=active_m,
                    llm_provider=selected_llm_key if use_llm_flag else None,
                    llm_api_key=api_key_val,
                    llm_endpoint=custom_endpoint,
                    llm_model=custom_model_name,
                    use_llm=use_llm_flag,
                )
            except Exception as err:
                st.error(f"Ошибка при обработке запроса: {err}")
                result = {
                    "intent": "error",
                    "final_answer": f"Произошла ошибка при обращении к модели: {err}. Попробуйте переключиться на 'Автономный экспертный RAG-анализ' или проверить токен.",
                    "citations": [],
                    "disclaimer": "Справочно-образовательная система.",
                    "retrieved_images": [],
                }

            intent = result.get("intent", "pathology_qa")
            answer = result.get("final_answer", "")
            citations = result.get("citations", [])
            disclaimer = result.get("disclaimer", "")
            images = result.get("retrieved_images", [])

            st.markdown(f'<span class="badge-chip badge-blue">Маршрут: {intent}</span>', unsafe_allow_html=True)
            st.markdown(answer)

            if images:
                st.markdown("##### 📸 Микропрепараты и гистологические срезы:")
                cols = st.columns(min(len(images), 2))
                for i, img in enumerate(images[:4]):
                    with cols[i % 2]:
                        src = resolve_img_source(img)
                        if src:
                            try:
                                st.image(src, caption=img.get("title", "Микропрепарат"), use_container_width=True)
                            except Exception:
                                if img.get("image_url"):
                                    st.image(img.get("image_url"), caption=img.get("title", "Микропрепарат"), use_container_width=True)

                        wsi_link = img.get("wsi_url") or img.get("image_url") or ""
                        st.markdown(
                            f"""
                            <div class="micro-card-dark">
                                <strong style="color: #F8FAFC;">{img.get('title', 'Препарат')}</strong>
                                <div class="card-meta-bar">
                                    <span class="meta-tag-dark">📍 {img.get('organ', '-')}</span>
                                    <span class="meta-tag-dark">🔬 {img.get('stain', 'H&E')}</span>
                                    <span class="meta-tag-dark">🔍 {img.get('magnification', '-')}</span>
                                </div>
                                <p style="font-size:0.8rem; color:#94A3B8; margin:4px 0 8px 0; line-height:1.4;">{img.get('caption', '')}</p>
                                <small style="color:#64748B;">Источник: {img.get('source_name', 'Атлас')}</small><br>
                                {f'<a href="{wsi_link}" target="_blank" rel="noopener noreferrer" class="wsi-action-btn-dark">🔬 Открыть WSI в полном разрешении ↗</a>' if wsi_link else ''}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            if citations:
                with st.expander(f"📚 Верифицированные источники и цитаты ({len(citations)})"):
                    for c in citations:
                        st.markdown(f"- **{c.get('source')}**: {c.get('title')} (*{c.get('reference')}*)")

            if disclaimer:
                st.markdown(f'<div class="regulatory-notice-dark">{disclaimer}</div>', unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "intent": intent,
                "images": images,
                "citations": citations,
                "disclaimer": disclaimer,
            })
