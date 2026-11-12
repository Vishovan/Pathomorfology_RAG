"""Streamlit Web Application для Pathology RAG (деплой на share.streamlit.io).

Светлый «клинический» интерфейс над LangGraph-агентом (pathrag.agent):
- выбор LLM-провайдера: Groq, Gemini, OpenRouter, Ollama / vLLM (Т-Банк), GigaChat, Free Cloud,
  либо автономный режим без LLM;
- специализированный RAG по РМЖ (HER2 0/1+/2+/3+, FISH, Ki-67, Allred, Nottingham) и общей патоморфологии;
- микропрепараты и ссылки на WSI;
- режим юриста: Роспатент, Минцифры, Росздравнадзор (Non-SaMD) и стандарты ГОСТ.
"""
import html
import re
import sys
from pathlib import Path
from typing import Any

# Streamlit Cloud не устанавливает пакет, поэтому добавляем src/ в путь импорта.
# Корневой /streamlit_app.py репозитория — тонкая точка входа, запускающая этот файл.
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR / "src"))

import streamlit as st

# Импорт графа агента и LLM утилит
try:
    from pathrag.agent import ask_agent
    from pathrag.agent.llm import (
        get_secret_or_env,
        is_local_ollama_available,
        test_llm_connection,
    )
except ImportError as e:
    st.error(f"Ошибка импорта модулей Pathology RAG: {e}")
    st.stop()

def _btn_stretch(label: str, **kwargs) -> bool:
    try:
        return st.button(label, width="stretch", **kwargs)
    except TypeError:
        return st.button(label, use_container_width=True, **kwargs)


def _img_stretch(src: Any, caption: str | None = None) -> None:
    try:
        st.image(src, caption=caption, width="stretch")
    except TypeError:
        st.image(src, caption=caption, use_container_width=True)


# -----------------------------------------------------------------------------
# 1. Настройка страницы и бело-голубой медицинский стиль (Clinical Minimalist)
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-page: #F8FAFC;
        --surface: #FFFFFF;
        --surface-soft: #F0F7FF;
        --surface-subtle: #F8FAFC;
        --border: #E2E8F0;
        --border-strong: #CBD5E1;
        --primary: #0284C7;
        --primary-dark: #0369A1;
        --primary-light: #E0F2FE;
        --primary-border: #BAE6FD;
        --accent-cyan: #0EA5E9;
        --accent-teal: #0D9488;
        --text-main: #0F172A;
        --text-muted: #334155;
        --text-subtle: #64748B;
        --radius: 14px;
        --radius-lg: 18px;
        --shadow-sm: 0 1px 3px rgba(15, 23, 42, 0.04);
        --shadow-md: 0 4px 18px -2px rgba(2, 132, 199, 0.08), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        --shadow-hover: 0 10px 25px -3px rgba(2, 132, 199, 0.15), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
        --ease: cubic-bezier(0.16, 1, 0.3, 1);
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        color: var(--text-main) !important;
        -webkit-font-smoothing: antialiased;
    }

    /* Фон: медицинский чистый белый с мягким небесно-голубым свечением */
    .stApp {
        background:
            radial-gradient(1100px 500px at 10% -5%, rgba(14, 165, 233, 0.08), transparent 60%),
            radial-gradient(900px 500px at 95% 5%, rgba(2, 132, 199, 0.06), transparent 55%),
            radial-gradient(1000px 700px at 50% 105%, rgba(13, 148, 136, 0.04), transparent 60%),
            var(--bg-page) !important;
        background-attachment: fixed !important;
    }

    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}
    .block-container {padding-top: 1.8rem; padding-bottom: 5rem; max-width: 1100px;}

    /* Сайдбар — чистый белый клинический стиль */
    section[data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid var(--border) !important;
        box-shadow: 2px 0 16px rgba(15, 23, 42, 0.03);
    }
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4 {
        font-size: 0.74rem !important; text-transform: uppercase; letter-spacing: 0.1em;
        color: var(--primary-dark) !important; font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label {
        color: var(--text-muted) !important;
    }
    hr {border-color: var(--border) !important;}

    /* Hero Header */
    .hero-medical {
        position: relative; padding: 28px 32px; margin-bottom: 24px;
        border: 1px solid var(--border); border-radius: var(--radius-lg);
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
        box-shadow: var(--shadow-md); overflow: hidden;
        animation: rise 0.6s var(--ease) both;
    }
    .hero-medical::before {
        content: ""; position: absolute; inset: -1px; border-radius: var(--radius-lg); padding: 1px;
        background: linear-gradient(120deg, rgba(2, 132, 199, 0.45), transparent 40%, transparent 60%, rgba(14, 165, 233, 0.4));
        -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
        -webkit-mask-composite: xor; mask-composite: exclude; pointer-events: none;
    }
    .hero-top-row {
        display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;
    }
    .hero-eyebrow {
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.12em;
        text-transform: uppercase; color: var(--primary); font-weight: 600;
        display: inline-flex; align-items: center; gap: 8px;
    }
    .pulse-dot-cyan {
        width: 8px; height: 8px; border-radius: 50%; background: #0284C7;
        box-shadow: 0 0 0 0 rgba(2, 132, 199, 0.6); animation: pulse 2s infinite;
    }
    .hero-institution {
        font-size: 0.75rem; color: var(--text-subtle); font-weight: 500;
        background: #F1F5F9; padding: 3px 10px; border-radius: 6px;
    }
    .hero-title {
        font-size: clamp(2rem, 3.8vw, 2.7rem); font-weight: 700; letter-spacing: -0.03em; line-height: 1.1;
        margin: 6px 0 10px 0; color: #0C4A6E;
    }
    .hero-subtitle {
        margin: 0; color: var(--text-muted); font-size: 0.95rem; max-width: 680px; line-height: 1.55;
    }
    .hero-stats {
        display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-top: 22px;
    }
    .stat-box {
        padding: 12px 16px; border-radius: 12px; border: 1px solid var(--border);
        background: #FFFFFF; box-shadow: var(--shadow-sm);
        transition: transform 0.25s var(--ease), box-shadow 0.25s var(--ease);
    }
    .stat-box:hover {
        transform: translateY(-2px); box-shadow: var(--shadow-md); border-color: var(--primary-border);
    }
    .stat-box b {display: block; font-size: 1.2rem; font-weight: 700; color: #0F172A; letter-spacing: -0.02em;}
    .stat-box span {font-size: 0.72rem; color: var(--text-subtle); text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;}
    .stat-llm {border-left: 3px solid var(--primary);}

    /* Чипы */
    .hero-chips {display: flex; gap: 8px; flex-wrap: wrap; margin-top: 18px;}
    .badge-med {
        display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 999px;
        font-size: 0.73rem; font-weight: 600; letter-spacing: 0.02em; border: 1px solid;
    }
    .badge-cyan {color: #0369A1; border-color: #BAE6FD; background: #F0F9FF;}
    .badge-blue {color: #1D4ED8; border-color: #BFDBFE; background: #EFF6FF;}
    .badge-teal {color: #0F766E; border-color: #99F6E4; background: #F0FDFA;}
    .badge-slate {color: #475569; border-color: #E2E8F0; background: #F8FAFC;}

    /* Чипы в ответах чата */
    .badge-chip {
        display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em; border: 1px solid;
    }
    .badge-blue {color: #0369A1; border-color: #BAE6FD; background: #F0F9FF;}
    .badge-emerald {color: #047857; border-color: #A7F3D0; background: #ECFDF5;}
    .badge-purple {color: #6D28D9; border-color: #DDD6FE; background: #F5F3FF;}

    /* Кнопки-сценарии */
    div[data-testid="stHorizontalBlock"] {gap: 12px !important; flex-wrap: wrap;}
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {min-width: 200px; flex: 1 1 200px !important;}
    .stButton > button {
        width: 100%; min-height: 54px; text-align: left; justify-content: flex-start;
        border-radius: var(--radius); border: 1px solid var(--border); background: #FFFFFF;
        color: var(--text-main); font-size: 0.85rem; font-weight: 500; padding: 12px 16px;
        box-shadow: var(--shadow-sm);
        transition: transform 0.25s var(--ease), border-color 0.25s var(--ease), box-shadow 0.25s var(--ease), background 0.25s var(--ease);
    }
    .stButton > button:hover {
        transform: translateY(-2px); border-color: #38BDF8;
        background: #F0F9FF; box-shadow: var(--shadow-hover); color: var(--primary-dark);
    }
    .stButton > button:active {transform: translateY(0) scale(0.99);}
    .stButton > button:focus-visible, a:focus-visible, input:focus-visible, textarea:focus-visible {
        outline: 2px solid var(--primary) !important; outline-offset: 2px;
    }

    /* Поля ввода и селекторы */
    div[data-baseweb="select"] > div, .stTextInput input, .stTextArea textarea {
        background: #FFFFFF !important; border: 1px solid var(--border) !important;
        border-radius: 10px !important; color: var(--text-main) !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03); transition: border-color 0.2s var(--ease), box-shadow 0.2s var(--ease);
    }
    div[data-baseweb="select"] > div:hover, .stTextInput input:hover {border-color: var(--border-strong) !important;}
    div[data-baseweb="select"] > div:focus-within, .stTextInput input:focus-within {
        border-color: var(--primary) !important; box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.15) !important;
    }

    /* Чат */
    div[data-testid="stChatMessage"] {
        background: transparent !important; padding: 0.9rem 0.2rem;
        border-bottom: 1px solid rgba(226, 232, 240, 0.6);
        animation: rise 0.4s var(--ease) both;
    }
    /* Сообщение пользователя: мягкий светло-голубой медицинский блок */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]),
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: #EFF6FF !important; border: 1px solid #BFDBFE;
        border-radius: var(--radius); padding: 1.1rem 1.25rem;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) p,
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {
        color: #1E3A8A !important; font-weight: 500; line-height: 1.6;
    }
    /* Сообщение ассистента: чисто белая медицинская карточка */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]),
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background: #FFFFFF !important; border: 1px solid #E2E8F0;
        border-radius: var(--radius); padding: 1.1rem 1.25rem;
        box-shadow: 0 2px 12px rgba(15, 23, 42, 0.03);
    }
    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] li {line-height: 1.68; color: var(--text-muted);}
    div[data-testid="stChatMessage"] strong {color: var(--text-main);}
    div[data-testid="stChatMessage"] h3, div[data-testid="stChatMessage"] h4, div[data-testid="stChatMessage"] h5 {
        color: #0C4A6E !important; margin-top: 1rem; margin-bottom: 0.5rem;
    }

    /* Поле ввода чата */
    div[data-testid="stChatInput"], div[data-testid="stBottomBlockContainer"] {background: transparent !important;}
    div[data-testid="stChatInput"] > div {
        background: #FFFFFF !important;
        border: 1px solid var(--border-strong) !important; border-radius: 16px !important;
        box-shadow: 0 12px 36px -12px rgba(2, 132, 199, 0.15), 0 2px 6px rgba(0, 0, 0, 0.04);
        transition: border-color 0.25s var(--ease), box-shadow 0.25s var(--ease);
    }
    div[data-testid="stChatInput"] > div:focus-within {
        border-color: var(--primary) !important;
        box-shadow: 0 12px 40px -10px rgba(2, 132, 199, 0.25), 0 0 0 4px rgba(2, 132, 199, 0.12);
    }

    /* Экспандеры источников */
    div[data-testid="stExpander"] details {
        border: 1px solid var(--border) !important; border-radius: 12px !important; background: #FFFFFF;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    div[data-testid="stExpander"] summary {color: #0369A1; font-weight: 600;}
    div[data-testid="stExpander"] summary:hover {color: var(--primary);}

    /* Карточки микропрепаратов */
    div[data-testid="stImage"] img {border-radius: 12px; border: 1px solid var(--border);
        transition: transform 0.4s var(--ease); box-shadow: 0 2px 8px rgba(0,0,0,0.04);}
    div[data-testid="stImage"] img:hover {transform: scale(1.015);}
    .micro-card-med {
        background: #FFFFFF; border: 1px solid var(--border); border-radius: var(--radius);
        padding: 14px 16px; margin: 10px 0 14px 0; box-shadow: var(--shadow-sm);
        transition: border-color 0.25s var(--ease), transform 0.25s var(--ease), box-shadow 0.25s var(--ease);
    }
    .micro-card-med:hover {
        border-color: #38BDF8; transform: translateY(-2px); box-shadow: var(--shadow-md);
    }
    .card-meta-bar {display: flex; gap: 6px; margin: 8px 0 10px 0; flex-wrap: wrap;}
    .meta-tag-med {
        background: #F0F9FF; color: #0369A1; border: 1px solid #BAE6FD;
        font-size: 0.72rem; padding: 2px 9px; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-weight: 500;
    }
    .wsi-action-btn-med {
        display: inline-flex; align-items: center; gap: 6px; margin-top: 10px;
        background: linear-gradient(135deg, #0284C7, #0EA5E9); color: #FFFFFF !important; text-decoration: none;
        padding: 8px 16px; border-radius: 10px; font-size: 0.8rem; font-weight: 600;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35); transition: transform 0.2s var(--ease), box-shadow 0.2s var(--ease);
    }
    .wsi-action-btn-med:hover {
        transform: translateY(-1px); box-shadow: 0 6px 20px rgba(2, 132, 199, 0.45); text-decoration: none;
    }

    /* Источники цитирования */
    .cite {display: flex; gap: 12px; align-items: flex-start; padding: 10px 0; border-bottom: 1px solid #F1F5F9; font-size: 0.85rem;}
    .cite:last-child {border-bottom: none;}
    .cite-n {
        flex: none; width: 24px; height: 24px; border-radius: 6px; display: grid; place-items: center;
        background: #E0F2FE; color: #0284C7; font-size: 0.74rem; font-family: 'JetBrains Mono', monospace; font-weight: 700;
    }
    .cite-src {color: #0F172A; font-weight: 600;}
    .cite-meta {color: #64748B; font-size: 0.76rem;}
    .cite a {color: #0284C7 !important; text-decoration: none; font-weight: 500;}
    .cite a:hover {text-decoration: underline;}

    /* Регуляторное уведомление (Non-SaMD) */
    .regulatory-notice-med {
        background: #F0F9FF; border: 1px solid #BAE6FD; border-left: 4px solid #0284C7;
        padding: 12px 16px; border-radius: 10px; font-size: 0.78rem; color: #0369A1; line-height: 1.5; margin-top: 14px;
    }

    @keyframes rise {from {opacity: 0; transform: translateY(6px);} to {opacity: 1; transform: none;}}
    @keyframes pulse {0% {box-shadow: 0 0 0 0 rgba(2, 132, 199, 0.5);} 70% {box-shadow: 0 0 0 8px rgba(2, 132, 199, 0);} 100% {box-shadow: 0 0 0 0 rgba(2, 132, 199, 0);}}
    @media (prefers-reduced-motion: reduce) {*, *::before, *::after {animation: none !important; transition: none !important;}}
    @media (max-width: 640px) {.hero-medical {padding: 20px;} .block-container {padding-top: 1rem;}}
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
        "groq": "⚡ Groq Cloud · Llama 3.3 70B (Бесплатно, сверхбыстро)",
        "tbank_local": "🟡 Т-Банк LLM / Ollama (T-Tech T-pro 2.1 / Llama 3.1) · локально",
        "free_cloud": "🌐 Free Cloud AI (Zero Config) · без ключей (для деплоя)",
        "gemini": "🤖 Google Gemini (Flash 2.0 / 1.5) · бесплатно",
        "openrouter": "🌐 OpenRouter (Бесплатные модели :free)",
        "gigachat": "🟢 Сбер GigaChat Pro (152-ФЗ)",
        "autonomous": "🔬 Автономный RAG-синтез · без LLM (офлайн)",
    }

    _local_ollama_available = is_local_ollama_available()
    groq_secret = get_secret_or_env("GROQ_API_KEY")
    gemini_secret = get_secret_or_env("GEMINI_API_KEY")
    openrouter_secret = get_secret_or_env("OPENROUTER_API_KEY")
    gigachat_secret = get_secret_or_env("GIGACHAT_CREDENTIALS")

    # Умный выбор модели по умолчанию
    if groq_secret:
        default_llm = "groq"
    elif _local_ollama_available:
        default_llm = "tbank_local"
    elif gemini_secret:
        default_llm = "gemini"
    elif openrouter_secret:
        default_llm = "openrouter"
    elif gigachat_secret:
        default_llm = "gigachat"
    else:
        default_llm = "free_cloud"

    selected_llm_key = st.selectbox(
        "Провайдер генерации ответа из чанков:",
        options=list(llm_options_map.keys()),
        format_func=lambda k: llm_options_map[k],
        index=list(llm_options_map.keys()).index(default_llm),
    )

    api_key_val = None
    custom_endpoint = "http://localhost:11434/v1"
    custom_model_name = "llama3.1:8b"

    if selected_llm_key == "groq":
        st.success("⚡ Сверхбыстрый бесплатный LPU-инференс (Llama 3.3 70B / Qwen 2.5 32B).")
        api_key_val = st.text_input(
            "Groq API Key:",
            value=groq_secret,
            type="password",
            placeholder="gsk_...",
            help="Получите бесплатный ключ за 30 секунд без банковской карты на https://console.groq.com",
        )
        custom_model_name = st.selectbox(
            "Модель на Groq:",
            options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen-2.5-32b"],
            index=0,
        )
        if not api_key_val:
            st.info("💡 Ключ можно ввести выше или сохранить в Streamlit Cloud Secrets (`GROQ_API_KEY`). Ключ 100% бесплатный на [console.groq.com](https://console.groq.com).")

    elif selected_llm_key == "tbank_local":
        st.success("🟡 Инференс через локальный сервер Ollama / vLLM.")
        with st.expander("Параметры локального сервера", expanded=not _local_ollama_available):
            custom_endpoint = st.text_input("Эндпоинт Ollama / vLLM:", value="http://localhost:11434/v1")
            custom_model_name = st.selectbox(
                "Модель Ollama:",
                options=["llama3.1:8b", "t-tech/T-pro-it-2.1:q4_K_M", "qwen2.5-coder:14b", "llama3.1", "qwen2.5"],
                index=0,
            )

    elif selected_llm_key == "free_cloud":
        st.success("✨ Работает онлайн прямо в браузере без API-ключей (Zero Config).")
        st.caption("Использует открытый облачный инференс для формирования ответа из чанков.")

    elif selected_llm_key == "gemini":
        st.success("🤖 Бесплатный доступ к моделям Google Gemini.")
        api_key_val = st.text_input(
            "Gemini API Key:",
            value=gemini_secret,
            type="password",
            placeholder="AIza...",
            help="Бесплатный ключ на https://aistudio.google.com",
        )
        custom_model_name = st.selectbox(
            "Модель Gemini:",
            options=["gemini-2.0-flash", "gemini-1.5-flash"],
            index=0,
        )

    elif selected_llm_key == "openrouter":
        st.success("🌐 Пул бесплатных открытых моделей через OpenRouter.")
        api_key_val = st.text_input(
            "OpenRouter API Key:",
            value=openrouter_secret,
            type="password",
            placeholder="sk-or-...",
            help="Бесплатный ключ на https://openrouter.ai",
        )
        custom_model_name = st.selectbox(
            "Модель OpenRouter:",
            options=[
                "meta-llama/llama-3.3-70b-instruct:free",
                "google/gemini-2.0-flash-exp:free",
                "deepseek/deepseek-r1:free",
                "qwen/qwen-2.5-72b-instruct:free",
            ],
            index=0,
        )

    elif selected_llm_key == "gigachat":
        st.success("🟢 Сбер GigaChat Pro (152-ФЗ, серверы в РФ).")
        api_key_val = st.text_input(
            "GigaChat Client Secret / Токен:",
            value=gigachat_secret,
            type="password",
            placeholder="Вставьте токен Сбера...",
        )
        custom_model_name = "GigaChat-Pro"
        if not get_secret_or_env("GIGACHAT_CA_BUNDLE") and get_secret_or_env("GIGACHAT_VERIFY_SSL", "true").lower() != "false":
            st.caption(
                "🔐 GigaChat использует сертификат НУЦ Минцифры. Если проверка связи не проходит, "
                "укажите `GIGACHAT_CA_BUNDLE` (путь к корневому сертификату) в secrets."
            )

    elif selected_llm_key == "autonomous":
        st.info("📌 Работает на встроенной верифицированной базе знаний клинических руководств без внешних запросов к LLM.")

    # Кнопка проверки связи с моделью в реальном времени
    if selected_llm_key != "autonomous":
        if _btn_stretch("🧪 Проверить связь с моделью"):
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
        <div class="regulatory-notice-med">
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
    """Фактический размер локальной базы знаний (фрагменты и уникальные источники)."""
    try:
        from pathrag.retrieval.hybrid_retriever import _corpus

        corpus = _corpus()
        return {"chunks": len(corpus), "sources": len({d.get("url") or d.get("source") for d in corpus})}
    except Exception:
        return {"chunks": "—", "sources": "—"}


_stats = _kb_stats()
_llm_label = llm_options_map[selected_llm_key].split("·")[0].strip(" ⚡🟡🌐🤖🟢🔬")
st.markdown(
    f"""
    <section class="hero-medical" role="banner" aria-label="Pathology RAG">
        <div class="hero-top-row">
            <span class="hero-eyebrow"><span class="pulse-dot-cyan" aria-hidden="true"></span>Clinical &amp; Regulatory Copilot</span>
            <span class="hero-institution">Digital Pathology AI · Evidence-Based System</span>
        </div>
        <h1 class="hero-title">Pathology RAG</h1>
        <p class="hero-subtitle">Интеллектуальный ассистент врача-патологоанатома: доказательный синтез по протоколам иммуногистохимии РМЖ (HER2, Ki-67, ER/PR), цифровой атлас микропрепаратов WSI и нормативная база ГОСТ для медицинского ПО.</p>
        <div class="hero-stats" role="list">
            <div class="stat-box" role="listitem"><b>{_stats["chunks"]}</b><span>фрагментов знаний</span></div>
            <div class="stat-box" role="listitem"><b>{_stats["sources"]}</b><span>клинических источников</span></div>
            <div class="stat-box stat-llm" role="listitem"><b>{_llm_label}</b><span>активная модель</span></div>
        </div>
        <div class="hero-chips">
            <span class="badge-med badge-cyan">🔬 ASCO/CAP · RUSSCO</span>
            <span class="badge-med badge-blue">📜 ГОСТ 34.602-2020</span>
            <span class="badge-med badge-teal">⚖️ Статус Non-SaMD</span>
            <span class="badge-med badge-slate">🛡️ 152-ФЗ Совместимость</span>
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
    """Локальный путь к микрофотографии (если файл есть в репозитории) или прямой веб-URL."""
    loc = img_dict.get("local_path")
    if loc and (CURRENT_DIR / loc).exists():
        return str(CURRENT_DIR / loc)
    return img_dict.get("image_url")


def _esc(value: Any, default: str = "") -> str:
    """Экранирование значений, попадающих в HTML (подписи могут прийти из краулинга чужих сайтов)."""
    return html.escape(str(value if value not in (None, "") else default))


def _safe_url(url: Any) -> str:
    """Только http(s)-ссылки: javascript:/data: и прочие схемы отбрасываются."""
    u = str(url or "").strip()
    return html.escape(u) if re.match(r"^https?://", u, re.IGNORECASE) else ""


def render_citations(citations: list[dict]) -> None:
    """Список источников: номер, название, ссылка и лицензия (если есть)."""
    rows = []
    for i, c in enumerate(citations, 1):
        ref = str(c.get("reference") or "")
        url_m = re.search(r"https?://\S+", ref)
        url = _safe_url(c.get("url") or (url_m.group(0) if url_m else ""))
        meta = _esc(re.sub(r"https?://\S+", "", ref).strip(" ·"))
        link = f' · <a href="{url}" target="_blank" rel="noopener noreferrer">открыть ↗</a>' if url else ""
        rows.append(
            f'<div class="cite"><span class="cite-n">{i}</span><div>'
            f'<div class="cite-src">{_esc(c.get("source"))}</div>'
            f'<div class="cite-meta">{_esc(c.get("title"))}{" · " + meta if meta else ""}{link}</div>'
            f'</div></div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def _micro_card_html(img: dict) -> str:
    wsi_link = _safe_url(img.get("wsi_url") or img.get("image_url"))
    wsi_btn = (
        f'<a href="{wsi_link}" target="_blank" rel="noopener noreferrer" class="wsi-action-btn-med">'
        f'🔬 Открыть WSI в полном разрешении ↗</a>' if wsi_link else ""
    )
    return f"""
        <div class="micro-card-med">
            <strong style="color: #0F172A; font-size: 0.95rem;">{_esc(img.get('title'), 'Препарат')}</strong>
            <div class="card-meta-bar">
                <span class="meta-tag-med">📍 {_esc(img.get('organ'), '-')}</span>
                <span class="meta-tag-med">🔬 {_esc(img.get('stain'), 'H&E')}</span>
                <span class="meta-tag-med">🔍 {_esc(img.get('magnification'), '-')}</span>
            </div>
            <p style="font-size:0.83rem; color:#475569; margin:6px 0 10px 0; line-height:1.45;">{_esc(img.get('caption'))}</p>
            <small style="color:#64748B;">Источник: {_esc(img.get('source_name'), 'Атлас')}</small><br>
            {wsi_btn}
        </div>
    """


def render_images(images: list[dict]) -> None:
    if not images:
        return
    st.markdown("##### 📸 Микропрепараты и гистологические срезы:")
    cols = st.columns(min(len(images), 2))
    for i, img in enumerate(images[:4]):
        with cols[i % 2]:
            caption = img.get("title", "Микропрепарат")
            src = resolve_img_source(img)
            if src:
                try:
                    _img_stretch(src, caption=caption)
                except Exception:
                    if img.get("image_url") and img.get("image_url") != src:
                        _img_stretch(img["image_url"], caption=caption)
            st.markdown(_micro_card_html(img), unsafe_allow_html=True)


def render_message(msg: dict) -> None:
    """Отрисовка одного сообщения ленты (используется и для истории, и для нового ответа)."""
    if msg["role"] == "assistant" and msg.get("intent") and msg["intent"] != "system_welcome":
        chips = [f'<span class="badge-chip badge-blue">Маршрут: {_esc(msg["intent"])}</span>']
        if msg.get("llm_used"):
            chips.append(f'<span class="badge-chip badge-emerald">🧠 {_esc(msg["llm_used"])}</span>')
        st.markdown(" ".join(chips), unsafe_allow_html=True)

    st.markdown(msg["content"])
    render_images(msg.get("images") or [])

    citations = msg.get("citations") or []
    if citations:
        with st.expander(f"📚 Источники · {len(citations)}"):
            render_citations(citations)

    if msg.get("disclaimer"):
        st.markdown(f'<div class="regulatory-notice-med">{_esc(msg["disclaimer"])}</div>', unsafe_allow_html=True)


def _avatar(role: str) -> str:
    return "🧑‍⚕️" if role == "user" else "🔬"


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
    with st.chat_message(msg["role"], avatar=_avatar(msg["role"])):
        render_message(msg)


# -----------------------------------------------------------------------------
# 4. Обработка запросов пользователя
# -----------------------------------------------------------------------------
prompt_input = st.chat_input("Задайте вопрос по патоморфологии, иммуногистохимии, WSI или стандартам ГОСТ...")

# Подстановка предустановленного вопроса по клику (введённый вручную текст важнее)
preset = st.session_state.pop("active_preset", None)
if not prompt_input and preset:
    prompt_input = preset

if prompt_input:
    user_msg = {"role": "user", "content": prompt_input}
    st.session_state.messages.append(user_msg)
    with st.chat_message("user", avatar=_avatar("user")):
        render_message(user_msg)

    with st.chat_message("assistant", avatar=_avatar("assistant")):
        with st.spinner("Оркестрация в LangGraph (маршрутизация -> ретривер -> верификация -> генерация)..."):
            use_llm_flag = selected_llm_key != "autonomous"
            try:
                result = ask_agent(
                    prompt_input,
                    mode=None if chosen_mode == "auto" else chosen_mode,
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
                    "final_answer": f"Произошла ошибка при обращении к модели: {err}. Попробуйте переключиться на «Автономный RAG-синтез» или проверить токен.",
                    "disclaimer": "Справочно-образовательная система.",
                }

        assistant_msg = {
            "role": "assistant",
            "content": result.get("final_answer", ""),
            "intent": result.get("intent", "pathology_qa"),
            "images": result.get("retrieved_images", []),
            "citations": result.get("citations", []),
            "disclaimer": result.get("disclaimer", ""),
            "llm_used": result.get("llm_used"),
        }
        render_message(assistant_msg)
        st.session_state.messages.append(assistant_msg)
