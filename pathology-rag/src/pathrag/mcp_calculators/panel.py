import json

import streamlit as st

from .client import invoke
from .network import invoke_network

NAMES = {"nottingham": "Ноттингемская шкала · Nottingham Grade", "ki67": "Индекс Ki-67",
         "h_score": "Гистохимическая оценка · H-score", "list_calculators": "Справочник методик"}
NAMES.update({"gleason": "Рак простаты · Gleason", "fnclcc": "Саркомы · FNCLCC",
              "allred": "Рак молочной железы · Allred", "search_images": "Поиск микропрепаратов",
              "search_corpus": "Поиск по корпусу", "pubmed_search_articles": "Поиск статей · PubMed"})
LABELS = {"query": "Поисковый запрос", "primary": "Первичный паттерн", "secondary": "Второй паттерн",
          "differentiation": "Дифференцировка", "necrosis": "Некроз", "intensity": "Интенсивность (баллы)",
          "proportion": "Доля окрашенных клеток (баллы)","tubules": "Образование трубочек", "nuclei": "Ядерный полиморфизм", "mitoses": "Митозы",
          "positive": "Положительные клетки", "total": "Всего оценено клеток",
          "negative": "Без окрашивания, %", "weak": "Слабое окрашивание, %",
          "moderate": "Умеренное окрашивание, %", "strong": "Сильное окрашивание, %"}


def inspector(trace):
    with st.expander("Инспектор JSON-RPC"):
        for direction, packet in trace:
            st.caption(direction)
            st.json(packet)


@st.cache_data(ttl=900, max_entries=128, show_spinner=False)
def cached_pubmed(method, params):
    result, trace, error = invoke_network("PubMed", method, params)
    if error:
        raise PubmedError(error, trace)
    return result, trace, None


class PubmedError(Exception):
    pass


def dispatch(source, method, params):
    if source == "stdio":
        return invoke(method, params)
    if source == "PubMed":
        try:
            return cached_pubmed(method, params)
        except PubmedError as exc:
            return {}, exc.args[1], exc.args[0]
    return invoke_network(source, method, params, autostart=True)


def render():
    st.title("Калькуляторы и источники")
    source = st.radio("Источник MCP", ["stdio", "SSE", "PubMed"], horizontal=True)
    catalog_key = "mcp_tools_" + source
    if catalog_key not in st.session_state:
        with st.spinner("Подключение к серверу"):
            catalog, trace, error = dispatch(source, "tools/list", {})
        if error:
            st.error(error)
            inspector(trace)
            st.button("Повторить подключение")
            return
        st.session_state[catalog_key] = catalog["tools"]
    tools = {item["name"]: item for item in st.session_state[catalog_key]}
    if source == "PubMed":
        tools = {k: v for k, v in tools.items() if k == "pubmed_search_articles"}
    if not tools:
        st.warning("Сервер не предоставил ожидаемых инструментов.")
        return
    name = st.selectbox("Методика", tools.keys(), format_func=lambda n: NAMES.get(n, n))
    tool = tools[name]
    st.write(tool.get("description", ""))
    args = {}
    with st.form("calc_" + name):
        for key, spec in tool["inputSchema"].get("properties", {}).items():
            if key not in tool["inputSchema"].get("required", []) and source == "PubMed":
                continue
            if "enum" in spec:
                args[key] = st.selectbox(LABELS.get(key, key), spec["enum"], index=None)
                continue
            if spec.get("type") == "boolean":
                args[key] = st.checkbox(LABELS.get(key, key), value=spec.get("default", False))
                continue
            if spec.get("type") not in ("integer", "number"):
                args[key] = st.text_input(LABELS.get(key, key)).strip() or None
                continue
            cast = int if spec["type"] == "integer" else float
            low = spec.get("minimum")
            if "exclusiveMinimum" in spec:
                low = spec["exclusiveMinimum"] + 1
            high = spec.get("maximum")
            args[key] = st.number_input(LABELS.get(key, key), value=None, step=cast(1),
                                        min_value=cast(low) if low is not None else None,
                                        max_value=cast(high) if high is not None else None)
        submit = st.form_submit_button("Выполнить")
    if not submit:
        return
    if None in args.values():
        st.warning("Заполните все поля.")
        return
    with st.spinner("Вызов инструмента"):
        result, trace, error = dispatch(source, "tools/call", {"name": name, "arguments": args})
    if error:
        st.error(error)
    else:
        data = result.get("structuredContent")
        if data is None:
            try:
                data = json.loads(result["content"][0]["text"])
            except (IndexError, KeyError, ValueError):
                data = result.get("content", [])
        if name in ("list_calculators", "search_images", "search_corpus") or source == "PubMed":
            st.json(data)
        else:
            if "grade" in data:
                grade = data["grade"]
                [st.success, st.warning, st.error][grade - 1](f"Grade {grade} · {data['score']} баллов")
            elif "percent" in data:
                st.metric("Индекс Ki-67", f"{data['percent']} %")
            else:
                st.metric(NAMES.get(name, name), data["score"])
                if "group" in data:
                    st.write(f"Grade Group {data['group']}")
            st.write({LABELS.get(k, k): v for k, v in args.items()})
            st.link_button("Источник методики", data["source"])
    inspector(trace)
