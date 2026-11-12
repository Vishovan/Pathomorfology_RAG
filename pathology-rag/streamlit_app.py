"""Streamlit Web Application для Pathology RAG (деплой на share.streamlit.io).

Ультрасовременный минималистичный интерфейс:
- Прямой вывод микропрепаратов через st.image (без битых редиректов на GitHub);
- Поддержка коммерческих LLM: Сбер GigaChat (152-ФЗ), Qwen 2.5 (Apache 2.0) и автономная база;
- Специализированный RAG по РМЖ (HER2 0/1+/2+/3+, FISH, Ki-67, Allred, Nottingham);
- Режим юриста: Роспатент, Минцифры, Росздравнадзор (Non-SaMD) и ГОСТы (34.602, 56939-2024 и др.).
"""
import os
import sys
from pathlib import Path

# Добавляем пути к модулям
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
except ImportError as e:
    st.error(f"Ошибка импорта модулей Pathology RAG: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 1. Настройка страницы и минималистичный ультрасовременный стиль
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Pathology RAG | Ассистент патоморфолога и юриста",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Скрытие дефолтного хедера Streamlit для минимализма */
    header[data-testid="stHeader"] {
        background-color: transparent;
    }
    
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #0f172a;
        margin-bottom: 0.15rem;
    }
    
    .hero-subtitle {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    
    .badge-tag {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-blue {
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #dbeafe;
    }
    .badge-emerald {
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
    }
    
    /* Карточки быстрых действий */
    .stButton > button {
        border-radius: 10px;
        font-size: 0.82rem;
        font-weight: 500;
        border: 1px solid #e2e8f0;
        background: #ffffff;
        color: #334155;
        padding: 8px 12px;
        transition: all 0.15s ease-in-out;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .stButton > button:hover {
        border-color: #3b82f6;
        color: #2563eb;
        background: #f8fafc;
        transform: translateY(-1px);
        box-shadow: 0 3px 6px rgba(0,0,0,0.05);
    }
    
    /* Карточка микропрепарата */
    .micro-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px;
        margin-top: 10px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    
    .card-meta {
        display: flex;
        gap: 6px;
        margin: 6px 0 8px 0;
        flex-wrap: wrap;
    }
    
    .meta-pill {
        background: #f1f5f9;
        color: #475569;
        font-size: 0.72rem;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 500;
    }
    
    .wsi-link-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #2563eb;
        color: #ffffff !important;
        text-decoration: none;
        padding: 7px 14px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 8px;
        transition: background 0.15s;
    }
    .wsi-link-btn:hover {
        background: #1d4ed8;
    }
    
    /* Минималистичный дисклеймер */
    .disclaimer-banner {
        background: #fffbeb;
        border-left: 3px solid #f59e0b;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.78rem;
        color: #78350f;
        margin-top: 12px;
        line-height: 1.4;
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
    st.markdown('<span class="badge-tag badge-emerald">● LangGraph + LlamaIndex</span>', unsafe_allow_html=True)
    st.caption("Мультимодальный ассистент патоморфолога & консультант по ГОСТам")
    st.divider()

    st.markdown("#### 🎯 Режим консультации")
    modes = {
        "auto": "🤖 Авто-выбор (LangGraph Router)",
        "breast_cancer": "🔬 РМЖ: ИГХ (HER2 & Ki-67)",
        "pathology": "📖 Общая патоморфология",
        "images": "🖼️ Микропрепараты & WSI",
        "legal": "⚖️ Режим юриста (ГОСТы и SaMD)",
        "literature": "🌐 Научный поиск (PubMed / arXiv)",
    }
    selected_mode = st.radio(
        "Выберите направление:",
        options=list(modes.keys()),
        format_func=lambda x: modes[x],
        index=0,
    )

    st.divider()

    st.markdown("#### ⚙️ Коммерциализация & LLM")
    llm_option = st.selectbox(
        "Модель генерации:",
        options=[
            "Автономная база (Бесплатно, без API-ключей)",
            "Сбер GigaChat (152-ФЗ, коммерческая лицензия в РФ)",
            "Qwen 2.5 (Apache 2.0 open-weights / vLLM / Ollama)",
            "OpenAI-совместимый эндпоинт",
        ],
        index=0,
    )

    api_key_val = None
    if "GigaChat" in llm_option:
        default_secret = st.secrets.get("GIGACHAT_CREDENTIALS", "") if hasattr(st, "secrets") else ""
        api_key_val = st.text_input("GigaChat Client Secret / Токен:", value=default_secret, type="password")
        st.caption("Соответствует 152-ФЗ (серверы в РФ). Разрешено для коммерческого ПО в РФ.")
    elif "Qwen" in llm_option or "OpenAI" in llm_option:
        default_endpoint = os.getenv("VLLM_ENDPOINT", "http://localhost:11434/v1")
        st.text_input("API Base URL:", value=default_endpoint)
        api_key_val = st.text_input("API Token (если требуется):", type="password")
        st.caption("Лицензия Apache 2.0: полностью свободна для коммерческих внедрений.")

    st.divider()

    with st.expander("📜 Документация по ГОСТам"):
        st.markdown("""
        - [ТЗ по ГОСТ 34.602-2020](https://github.com/Vishovan/Pathomorfology_RAG/blob/main/pathology-rag/docs/gost/TZ_GOST_34_602_2020.md)
        - [Ведомость документов ГОСТ 34.201](https://github.com/Vishovan/Pathomorfology_RAG/blob/main/pathology-rag/docs/gost/Vedomost_Dokumentov_GOST_34_201_2020.md)
        - [Описание архитектуры ГОСТ Р 59795](https://github.com/Vishovan/Pathomorfology_RAG/blob/main/pathology-rag/docs/gost/Architecture_and_Data_GOST_59795_2021.md)
        - [Обоснование Non-SaMD статуса](https://github.com/Vishovan/Pathomorfology_RAG/blob/main/pathology-rag/docs/gost/Intended_Use_and_Regulatory_Limitations.md)
        """)

    st.markdown(
        """
        <div class="disclaimer-banner">
        <strong>Статус Non-SaMD:</strong><br>
        Система разработана для справочно-образовательных целей и не классифицируется как медицинское изделие.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# 3. Основная область (Main Area)
# -----------------------------------------------------------------------------
st.markdown('<div class="hero-title">Pathology RAG Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Интеллектуальная система по патоморфологии (РМЖ ИГХ, WSI) и нормативной базе ГОСТ</div>',
    unsafe_allow_html=True,
)

# Быстрые кнопки запросов в один клик
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔬 HER2 2+ и ISH/FISH", use_container_width=True):
        st.session_state["query_preset"] = "Какие критерии оценки HER2 2+ при раке молочной железы и когда требуется подтверждающее тестирование FISH?"
with col2:
    if st.button("🔬 Ki-67 и подтипы РМЖ", use_container_width=True):
        st.session_state["query_preset"] = "Как подсчитывается Ki-67 и какие пороговые значения разделяют люминальный А и В подтипы?"
with col3:
    if st.button("🖼️ WSI: серое опеченение", use_container_width=True):
        st.session_state["query_preset"] = "Покажи микропрепарат серого опеченения при крупозной пневмонии"
with col4:
    if st.button("⚖️ Юрист: ГОСТ 34.602 и SaMD", use_container_width=True):
        st.session_state["query_preset"] = "Какие процедуры пройти (Роспатент, Минцифры, Росздравнадзор) и какие ГОСТы нужны для разработки ТЗ?"


def resolve_image_path(img_dict: dict) -> str | None:
    """Определяет доступный путь или URL к изображению."""
    local_rel = img_dict.get("local_path")
    if local_rel:
        p1 = CURRENT_DIR / local_rel
        if p1.exists():
            return str(p1)
        p2 = CURRENT_DIR / "pathology-rag" / local_rel
        if p2.exists():
            return str(p2)
    # Иначе используем проверенный web URL
    return img_dict.get("image_url")


# История сообщений
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": """Приветствую! Я мультимодальный RAG-ассистент по **патоморфологии** и **нормативной базе ГОСТ**.

Я готов ответить на вопросы по:
- **Раку молочной железы (РМЖ):** критерии оценки HER2/neu (0, 1+, 2+ equivocal, 3+), рефлекс-тестирование FISH/SISH, индекс пролиферации Ki-67 (hot spots, люминальный А/В), Allred ER/PR, Ноттингемская градация, дифдиагноз долькового рака (утрата E-cadherin);
- **Общей патоморфологии:** пневмонии, дистрофии, атеросклероз, туберкулез, аденокарциномы;
- **Микропрепаратам и Whole Slide Images (WSI):** показ гистологических снимков высокого разрешения с кнопкой перехода в цифровой вьюер;
- **Режиму юриста:** разграничение Роспатента, Минцифры и Росздравнадзора (Non-SaMD), разработка ТЗ по ГОСТ 34.602-2020, безопасная разработка ГОСТ Р 56939-2024.

Задайте вопрос в поле внизу или выберите готовый пример выше!""",
            "intent": "system_welcome",
            "images": [],
            "citations": [],
            "disclaimer": "Справочно-образовательная система. Не является медицинским изделием.",
        }
    ]

# Отрисовка сообщений
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="🧑‍⚕️" if msg["role"] == "user" else "🔬"):
        if msg.get("intent") and msg["role"] == "assistant" and msg.get("intent") != "system_welcome":
            st.markdown(f'<span class="badge-tag badge-blue">Маршрут: {msg["intent"]}</span>', unsafe_allow_html=True)

        st.markdown(msg["content"])

        # Отрисовка карточек микропрепаратов
        images = msg.get("images", [])
        if images:
            st.markdown("#### 📸 Микропрепараты и гистологические срезы:")
            cols = st.columns(min(len(images), 2))
            for i, img in enumerate(images[:4]):
                with cols[i % 2]:
                    # 1. Показ самого изображения
                    img_src = resolve_image_path(img)
                    if img_src:
                        try:
                            st.image(img_src, caption=img.get("title", "Микропрепарат"), use_container_width=True)
                        except Exception:
                            # Fallback на image_url
                            if img.get("image_url"):
                                st.image(img.get("image_url"), caption=img.get("title", "Микропрепарат"), use_container_width=True)

                    # 2. Карточка метаданных и прямая кнопка перехода
                    wsi_link = img.get("wsi_url") or img.get("image_url") or ""
                    st.markdown(
                        f"""
                        <div class="micro-card">
                            <strong>{img.get('title', 'Препарат')}</strong>
                            <div class="card-meta">
                                <span class="meta-pill">📍 {img.get('organ', '-')}</span>
                                <span class="meta-pill">🔬 {img.get('stain', 'H&E')}</span>
                                <span class="meta-pill">🔍 {img.get('magnification', '-')}</span>
                            </div>
                            <p style="font-size:0.8rem; color:#475569; margin:4px 0 8px 0;">{img.get('caption', '')}</p>
                            <small style="color:#94a3b8;">Источник: {img.get('source_name', 'Атлас')}</small><br>
                            {f'<a href="{wsi_link}" target="_blank" class="wsi-link-btn">🔬 Открыть полноразмерный снимок / WSI</a>' if wsi_link else ''}
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
                f'<div class="disclaimer-banner">{msg["disclaimer"]}</div>',
                unsafe_allow_html=True,
            )

# -----------------------------------------------------------------------------
# 4. Обработка ввода
# -----------------------------------------------------------------------------
user_query = st.chat_input("Задайте вопрос по патоморфологии, микропрепаратам или ГОСТам...")

# Обработка нажатия на кнопку-пример
if "query_preset" in st.session_state and st.session_state["query_preset"]:
    user_query = st.session_state["query_preset"]
    st.session_state["query_preset"] = None

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🧑‍⚕️"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🔬"):
        with st.spinner("Оркестрация в графе LangGraph (роутер -> ретривер -> грейдер -> генерация)..."):
            active_m = None if selected_mode == "auto" else selected_mode

            provider_code = None
            if "GigaChat" in llm_option:
                provider_code = "gigachat"
            elif "Qwen" in llm_option:
                provider_code = "qwen_local"
            elif "OpenAI" in llm_option:
                provider_code = "openai_compat"

            res = ask_agent(
                user_query,
                mode=active_m,
                llm_provider=provider_code,
                llm_api_key=api_key_val,
                use_llm=bool(provider_code),
            )

            intent = res.get("intent", "pathology_qa")
            answer = res.get("final_answer", "")
            citations = res.get("citations", [])
            disclaimer = res.get("disclaimer", "")
            images = res.get("retrieved_images", [])

            st.markdown(f'<span class="badge-tag badge-blue">Маршрут: {intent}</span>', unsafe_allow_html=True)
            st.markdown(answer)

            if images:
                st.markdown("#### 📸 Микропрепараты и гистологические срезы:")
                cols = st.columns(min(len(images), 2))
                for i, img in enumerate(images[:4]):
                    with cols[i % 2]:
                        img_src = resolve_image_path(img)
                        if img_src:
                            try:
                                st.image(img_src, caption=img.get("title", "Микропрепарат"), use_container_width=True)
                            except Exception:
                                if img.get("image_url"):
                                    st.image(img.get("image_url"), caption=img.get("title", "Микропрепарат"), use_container_width=True)

                        wsi_link = img.get("wsi_url") or img.get("image_url") or ""
                        st.markdown(
                            f"""
                            <div class="micro-card">
                                <strong>{img.get('title', 'Препарат')}</strong>
                                <div class="card-meta">
                                    <span class="meta-pill">📍 {img.get('organ', '-')}</span>
                                    <span class="meta-pill">🔬 {img.get('stain', 'H&E')}</span>
                                    <span class="meta-pill">🔍 {img.get('magnification', '-')}</span>
                                </div>
                                <p style="font-size:0.8rem; color:#475569; margin:4px 0 8px 0;">{img.get('caption', '')}</p>
                                <small style="color:#94a3b8;">Источник: {img.get('source_name', 'Атлас')}</small><br>
                                {f'<a href="{wsi_link}" target="_blank" class="wsi-link-btn">🔬 Открыть полноразмерный снимок / WSI</a>' if wsi_link else ''}
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
                    f'<div class="disclaimer-banner">{disclaimer}</div>',
                    unsafe_allow_html=True,
                )

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "intent": intent,
                "images": images,
                "citations": citations,
                "disclaimer": disclaimer,
            })
