# Pathomorfology RAG

Справочно-образовательный ИИ-ассистент для патологоанатомов и студентов: отвечает на вопросы по патоморфологии со ссылками на источники, подбирает микропрепараты и консультирует по ГОСТам для медицинского ПО.

> ⚠️ Система **не является медицинским изделием** и не предназначена для постановки диагноза конкретным пациентам ([обоснование Non-SaMD](pathology-rag/docs/gost/Intended_Use_and_Regulatory_Limitations.md)).

## Возможности

- **РМЖ и ИГХ** — HER2 (0 / 1+ / 2+ / 3+, HER2-low), FISH/SISH, Ki-67, ER/PR по Allred, Nottingham grade.
- **Общая патоморфология** — воспаление, некрозы, дистрофии, пневмонии, атеросклероз и др.
- **Атлас микропрепаратов** — фото с описанием, окраской, увеличением и ссылками на WSI.
- **Режим юриста** — ГОСТ 34.602, ГОСТ Р 56939 и др.; Роспатент / Минцифры / Росздравнадзор.
- **Научный поиск** — PubMed, arXiv, веб.

Ответ формирует выбранная LLM (Groq, Gemini, OpenRouter, GigaChat, локальные Ollama/vLLM, бесплатный Free Cloud) по фрагментам из встроенной базы знаний (~2170 фрагментов, 113 источников). Без LLM работает автономный режим — ответ собирается из найденных фрагментов.

## Быстрый старт

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py      # http://localhost:8501
```

Ключи LLM — в `.streamlit/secrets.toml` (шаблон: [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example)) или в переменных окружения. Без ключей доступны Free Cloud и автономный режим.

> Free Cloud отправляет вопрос публичному стороннему сервису — не используйте его с данными пациентов.

## Разработка

```bash
cd pathology-rag
pip install -e ".[ui,dev]"
ruff check . && pytest
```

Полный режим (векторный поиск Qdrant + загрузка своих PDF и сайтов): `pip install -e ".[index,ui]"`, `docker compose up -d`, затем `python scripts/ingest.py`.

## Структура

```
.
├── streamlit_app.py     # точка входа Streamlit Cloud -> pathology-rag/streamlit_app.py
├── requirements.txt     # зависимости облегчённого режима (Streamlit Cloud)
├── .streamlit/          # тема и шаблон secrets
├── .github/workflows/   # CI: ruff + pytest
├── scripts/             # запуск локальной LLM (Ollama / vLLM)
└── pathology-rag/       # приложение: агент LangGraph, поиск, база знаний, тесты, документация
```

## Документация

- [pathology-rag/README.md](pathology-rag/README.md) — подробное описание функций, архитектуры графа и скриптов.
- [pathology-rag/docs/gost/](pathology-rag/docs/gost/) — ТЗ по ГОСТ 34.602-2020, ведомость документов, описание архитектуры, регуляторный статус.
