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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg: #07090F;
        --surface: rgba(17, 22, 34, 0.62);
        --surface-solid: #0E131E;
        --border: rgba(255, 255, 255, 0.07);
        --border-strong: rgba(255, 255, 255, 0.14);
        --accent: #7C9CFF;
        --accent-2: #B490FF;
        --accent-glow: rgba(124, 156, 255, 0.28);
        --text: #EEF2F8;
        --text-2: #9AA6B8;
        --text-3: #5F6B7E;
        --radius: 14px;
        --ease: cubic-bezier(0.16, 1, 0.3, 1);
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        color: var(--text) !important;
        -webkit-font-smoothing: antialiased;
    }
    /* Фон: глубокий графит + мягкое «северное сияние» */
    .stApp {
        background:
            radial-gradient(1100px 520px at 12% -8%, rgba(124, 156, 255, 0.13), transparent 60%),
            radial-gradient(900px 480px at 100% 0%, rgba(180, 144, 255, 0.10), transparent 55%),
            radial-gradient(800px 600px at 50% 120%, rgba(45, 212, 191, 0.06), transparent 60%),
            var(--bg) !important;
        background-attachment: fixed !important;
    }

    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}
    .block-container {padding-top: 2rem; padding-bottom: 6rem; max-width: 1080px;}

    /* Сайдбар — матовое стекло */
    section[data-testid="stSidebar"] {
        background: rgba(10, 13, 21, 0.78) !important;
        backdrop-filter: blur(18px) saturate(140%);
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4 {
        font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.12em;
        color: var(--text-3) !important; font-weight: 600 !important;
    }
    hr {border-color: var(--border) !important;}

    /* Hero */
    .hero {
        position: relative; padding: 28px 30px; margin-bottom: 22px;
        border: 1px solid var(--border); border-radius: 20px;
        background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.01));
        overflow: hidden; animation: rise 0.7s var(--ease) both;
    }
    .hero::before {
        content: ""; position: absolute; inset: -1px; border-radius: 20px; padding: 1px;
        background: linear-gradient(120deg, rgba(124,156,255,0.55), transparent 35%, transparent 65%, rgba(180,144,255,0.45));
        -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
        -webkit-mask-composite: xor; mask-composite: exclude; pointer-events: none;
    }
    .hero-eyebrow {
        font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.14em;
        text-transform: uppercase; color: var(--accent); margin: 0 0 10px 0;
        display: inline-flex; align-items: center; gap: 8px;
    }
    .pulse-dot {width: 7px; height: 7px; border-radius: 50%; background: #34D399;
        box-shadow: 0 0 0 0 rgba(52,211,153,0.6); animation: pulse 2.2s infinite;}
    .hero-title {
        font-size: clamp(1.9rem, 3.6vw, 2.7rem); font-weight: 700; letter-spacing: -0.035em; line-height: 1.08;
        margin: 0; background: linear-gradient(135deg, #FFFFFF 0%, #C9D4FF 55%, #B490FF 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {margin: 10px 0 0 0; color: var(--text-2); font-size: 0.95rem; max-width: 640px; line-height: 1.5;}
    .hero-stats {display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-top: 22px;}
    .stat {padding: 12px 14px; border-radius: 12px; border: 1px solid var(--border); background: rgba(255,255,255,0.02);}
    .stat b {display: block; font-size: 1.15rem; font-weight: 600; color: var(--text); letter-spacing: -0.02em;}
    .stat span {font-size: 0.72rem; color: var(--text-3); text-transform: uppercase; letter-spacing: 0.08em;}

    /* Чипы */
    .hero-badges {display: flex; gap: 8px; flex-wrap: wrap; margin-top: 16px;}
    .badge-chip {display: inline-flex; align-items: center; gap: 6px; padding: 4px 11px; border-radius: 999px;
        font-size: 0.72rem; font-weight: 500; letter-spacing: 0.02em; border: 1px solid var(--border-strong);
        background: rgba(255,255,255,0.03); color: var(--text-2);}
    .badge-emerald {color: #6EE7B7; border-color: rgba(52,211,153,0.28); background: rgba(52,211,153,0.07);}
    .badge-blue {color: #A5B8FF; border-color: rgba(124,156,255,0.3); background: rgba(124,156,255,0.08);}
    .badge-purple {color: #CDB4FF; border-color: rgba(180,144,255,0.3); background: rgba(180,144,255,0.08);}

    /* Кнопки-сценарии: адаптивная сетка карточек */
    div[data-testid="stHorizontalBlock"] {gap: 10px !important; flex-wrap: wrap;}
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {min-width: 200px; flex: 1 1 200px !important;}
    .stButton > button {
        width: 100%; min-height: 54px; text-align: left; justify-content: flex-start;
        border-radius: var(--radius); border: 1px solid var(--border); background: var(--surface);
        backdrop-filter: blur(10px); color: var(--text); font-size: 0.84rem; font-weight: 500; padding: 12px 16px;
        transition: transform 0.35s var(--ease), border-color 0.35s var(--ease), box-shadow 0.35s var(--ease), background 0.35s var(--ease);
        animation: rise 0.6s var(--ease) both;
    }
    .stButton > button:hover {
        transform: translateY(-2px); border-color: rgba(124,156,255,0.45);
        background: rgba(124,156,255,0.07); box-shadow: 0 10px 30px -12px var(--accent-glow); color: #fff;
    }
    .stButton > button:active {transform: translateY(0) scale(0.99);}
    .stButton > button:focus-visible, a:focus-visible, input:focus-visible, textarea:focus-visible {
        outline: 2px solid var(--accent) !important; outline-offset: 2px;
    }

    /* Поля ввода */
    div[data-baseweb="select"] > div, .stTextInput input, .stTextArea textarea {
        background: rgba(255,255,255,0.03) !important; border: 1px solid var(--border) !important;
        border-radius: 10px !important; color: var(--text) !important; transition: border-color 0.25s var(--ease);
    }
    div[data-baseweb="select"] > div:hover, .stTextInput input:hover {border-color: var(--border-strong) !important;}

    /* Чат */
    div[data-testid="stChatMessage"] {
        background: transparent !important; padding: 1rem 0.25rem; animation: rise 0.5s var(--ease) both;
        border-bottom: 1px solid rgba(255,255,255,0.035);
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]),
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: rgba(124,156,255,0.045) !important; border: 1px solid rgba(124,156,255,0.12);
        border-radius: var(--radius); padding: 1rem 1.1rem;
    }
    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] li {line-height: 1.65; color: #DCE3EE;}
    div[data-testid="stChatMessage"] strong {color: #fff;}
    div[data-testid="stChatInput"], div[data-testid="stBottomBlockContainer"] {background: transparent !important;}
    div[data-testid="stChatInput"] > div {
        background: rgba(14,19,30,0.85) !important; backdrop-filter: blur(16px);
        border: 1px solid var(--border-strong) !important; border-radius: 16px !important;
        box-shadow: 0 20px 50px -20px rgba(0,0,0,0.8), 0 0 0 1px rgba(124,156,255,0.05);
        transition: border-color 0.3s var(--ease), box-shadow 0.3s var(--ease);
    }
    div[data-testid="stChatInput"] > div:focus-within {border-color: rgba(124,156,255,0.5) !important;
        box-shadow: 0 20px 50px -20px rgba(0,0,0,0.8), 0 0 0 4px rgba(124,156,255,0.12);}

    /* Экспандеры (источники) */
    div[data-testid="stExpander"] details {
        border: 1px solid var(--border) !important; border-radius: 12px !important; background: rgba(255,255,255,0.015);
    }
    div[data-testid="stExpander"] summary:hover {color: var(--accent);}

    /* Карточки микропрепаратов */
    div[data-testid="stImage"] img {border-radius: 12px; border: 1px solid var(--border);
        transition: transform 0.5s var(--ease); }
    div[data-testid="stImage"] img:hover {transform: scale(1.015);}
    .micro-card-dark {background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
        padding: 14px 16px; margin: 10px 0 14px 0; backdrop-filter: blur(10px);
        transition: border-color 0.3s var(--ease), transform 0.3s var(--ease);}
    .micro-card-dark:hover {border-color: var(--border-strong); transform: translateY(-1px);}
    .card-meta-bar {display: flex; gap: 6px; margin: 8px 0 10px 0; flex-wrap: wrap;}
    .meta-tag-dark {background: rgba(255,255,255,0.04); color: var(--text-2); border: 1px solid var(--border);
        font-size: 0.7rem; padding: 2px 8px; border-radius: 6px; font-family: 'JetBrains Mono', monospace;}
    .wsi-action-btn-dark {display: inline-flex; align-items: center; gap: 6px; margin-top: 10px;
        background: linear-gradient(135deg, #6D8BFF, #9C7BFF); color: #fff !important; text-decoration: none;
        padding: 7px 14px; border-radius: 9px; font-size: 0.78rem; font-weight: 600;
        box-shadow: 0 8px 24px -10px rgba(124,156,255,0.7); transition: transform 0.25s var(--ease), box-shadow 0.25s var(--ease);}
    .wsi-action-btn-dark:hover {transform: translateY(-1px); box-shadow: 0 12px 30px -10px rgba(124,156,255,0.9); text-decoration: none;}

    /* Источники */
    .cite {display: flex; gap: 10px; align-items: flex-start; padding: 9px 0; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 0.83rem;}
    .cite:last-child {border-bottom: none;}
    .cite-n {flex: none; width: 22px; height: 22px; border-radius: 6px; display: grid; place-items: center;
        background: rgba(124,156,255,0.12); color: #A5B8FF; font-size: 0.7rem; font-family: 'JetBrains Mono', monospace;}
    .cite-src {color: var(--text); font-weight: 500;}
    .cite-meta {color: var(--text-3); font-size: 0.74rem;}
    .cite a {color: var(--accent) !important; text-decoration: none;}

    .regulatory-notice-dark {background: rgba(245,158,11,0.05); border: 1px solid rgba(245,158,11,0.16);
        border-left: 2px solid #F59E0B; padding: 10px 14px; border-radius: 10px; font-size: 0.76rem;
        color: #F5D58A; line-height: 1.45; margin-top: 14px;}

    .llm-status {display: flex; align-items: center; gap: 8px; font-size: 0.78rem; color: var(--text-2);
        padding: 8px 12px; border: 1px solid var(--border); border-radius: 10px; background: rgba(255,255,255,0.02);}

    @keyframes rise {from {opacity: 0; transform: translateY(8px);} to {opacity: 1; transform: none;}}
    @keyframes pulse {0% {box-shadow: 0 0 0 0 rgba(52,211,153,0.55);} 70% {box-shadow: 0 0 0 8px rgba(52,211,153,0);} 100% {box-shadow: 0 0 0 0 rgba(52,211,153,0);}}
    @media (prefers-reduced-motion: reduce) {*, *::before, *::after {animation: none !important; transition: none !important;}}
    @media (max-width: 640px) {.hero {padding: 20px;} .block-container {padding-top: 1rem;}}
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
        "tbank_local": "🟡 Т-Банк LLM (T-Tech T-pro 2.1) · локальный инференс",
        "free_cloud": "🌐 Free Cloud AI (T-Tech / Qwen) · без ключей (для деплоя)",
        "autonomous": "🔬 Автономный клинический RAG · без LLM (офлайн)",
    }

    # Авто-проверка доступности локального сервера Т-Банка (Ollama)
    _local_ollama_available = False
    try:
        import httpx
        with httpx.Client(timeout=0.3) as client:
            r = client.get("http://localhost:11434/api/version")
            _local_ollama_available = (r.status_code == 200)
    except Exception:
        _local_ollama_available = False

    default_llm = "tbank_local" if _local_ollama_available else "free_cloud"

    selected_llm_key = st.selectbox(
        "Провайдер интеллекта (без API-ключей):",
        options=list(llm_options_map.keys()),
        format_func=lambda k: llm_options_map[k],
        index=list(llm_options_map.keys()).index(default_llm),
    )

    api_key_val = None
    custom_endpoint = None
    custom_model_name = None

    if selected_llm_key == "tbank_local":
        st.success("✨ Открытая модель Т-Банка (T-Tech T-pro 2.1) подключена напрямую без ключей.")
        with st.expander("Параметры локального сервера"):
            custom_endpoint = st.text_input("Эндпоинт Ollama / vLLM:", value="http://localhost:11434/v1")
            custom_model_name = st.text_input("Имя модели Т-Банка:", value="t-tech/T-pro-it-2.1:q4_K_M")

    elif selected_llm_key == "free_cloud":
        st.success("✨ Работает онлайн прямо в браузере без API-ключей (Zero Config).")

    elif selected_llm_key == "autonomous":
        st.info("📌 Работает на встроенной верифицированной базе знаний клинических руководств без внешних запросов.")

    # Кнопка проверки связи с моделью в реальном времени
    if selected_llm_key != "autonomous":
        if st.button("🧪 Проверить связь с моделью", use_container_width=True):
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
@st.cache_data(show_spinner=False)
def _kb_stats() -> dict:
    return {"chunks": 2160, "sources": 142}


_stats = _kb_stats()
_llm_label = llm_options_map[selected_llm_key].split("·")[0].strip(" 🟡🌐🔬")
st.markdown(
    f"""
    <section class="hero" role="banner" aria-label="Pathology RAG">
        <p class="hero-eyebrow"><span class="pulse-dot" aria-hidden="true"></span>Clinical &amp; Regulatory Copilot</p>
        <h1 class="hero-title">Pathology RAG</h1>
        <p class="hero-subtitle">Ассистент патоморфолога: общая и частная патология, ИГХ рака молочной железы,
        микропрепараты и WSI, нормативная база ГОСТ. Каждый ответ — со ссылками на источники.</p>
        <div class="hero-stats" role="list">
            <div class="stat" role="listitem"><b>{_stats["chunks"]}</b><span>фрагментов знаний</span></div>
            <div class="stat" role="listitem"><b>{_stats["sources"]}</b><span>источников</span></div>
            <div class="stat" role="listitem"><b>{_llm_label}</b><span>LLM</span></div>
        </div>
        <div class="hero-badges">
            <span class="badge-chip badge-purple">ГОСТ 34.602-2020</span>
            <span class="badge-chip badge-blue">ASCO/CAP · RUSSCO</span>
            <span class="badge-chip badge-emerald">Non-SaMD</span>
        </div>
    </section>
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


def render_citations(citations: list[dict]) -> None:
    """Список источников: номер, название, ссылка и лицензия (если есть)."""
    import html as _h
    import re as _re

    rows = []
    for i, c in enumerate(citations, 1):
        ref = str(c.get("reference") or "")
        url_m = _re.search(r"https?://\S+", ref)
        url = c.get("url") or (url_m.group(0) if url_m else "")
        meta = _h.escape(_re.sub(r"https?://\S+", "", ref).strip(" ·"))
        link = f' · <a href="{_h.escape(url)}" target="_blank" rel="noopener noreferrer">открыть ↗</a>' if url else ""
        rows.append(
            f'<div class="cite"><span class="cite-n">{i}</span><div>'
            f'<div class="cite-src">{_h.escape(str(c.get("source") or ""))}</div>'
            f'<div class="cite-meta">{_h.escape(str(c.get("title") or ""))}{" · " + meta if meta else ""}{link}</div>'
            f'</div></div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


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
            with st.expander(f"📚 Источники · {len(citations)}"):
                render_citations(citations)

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
                with st.expander(f"📚 Источники · {len(citations)}"):
                    render_citations(citations)

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
