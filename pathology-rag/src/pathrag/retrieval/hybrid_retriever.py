"""Гибридный ретривер (LlamaIndex + локальный fallback).

Предоставляет единый контракт:
    retrieve(query: str, filters: dict | None = None, top_k: int = 5) -> list[dict]
"""
import json
import logging
from pathlib import Path
from typing import Any

from ..config import SETTINGS

log = logging.getLogger("pathrag.retrieval.hybrid")
KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"


def _load_knowledge_corpus() -> list[dict[str, Any]]:
    """Загружает эталонные тексты из базы знаний и кэшированных чанков."""
    docs = []
    # 1. РМЖ и ИГХ
    brca_path = KNOWLEDGE_DIR / "breast_cancer.json"
    if brca_path.exists():
        with open(brca_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            source = data.get("source", "Клинические рекомендации")
            for t in data.get("topics", []):
                docs.append({
                    "id": t.get("id"),
                    "title": t.get("title"),
                    "nosology": t.get("nosology"),
                    "organ": t.get("organ"),
                    "text": t.get("content"),
                    "source": source,
                    "page_reference": t.get("page_reference", "Раздел 1.4"),
                    "category": "breast_cancer_ihc",
                })

    # 2. Общая патология
    gen_path = KNOWLEDGE_DIR / "general_pathology.json"
    if gen_path.exists():
        with open(gen_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            source = data.get("source", "Учебник по патологической анатомии")
            for t in data.get("topics", []):
                docs.append({
                    "id": t.get("id"),
                    "title": t.get("title"),
                    "nosology": t.get("nosology"),
                    "organ": t.get("organ"),
                    "text": t.get("content"),
                    "source": source,
                    "page_reference": t.get("page_reference", "Гл. 14"),
                    "category": "general_pathology",
                })

    # 3. Кэшированные чанки из data/processed/chunks/
    chunks_dir = SETTINGS.processed_dir / "chunks"
    if chunks_dir.exists():
        for chunk_file in chunks_dir.glob("*.jsonl"):
            try:
                with open(chunk_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        chunk_data = json.loads(line)
                        meta = chunk_data.get("metadata", {})
                        docs.append({
                            "id": f"{chunk_file.stem}_{meta.get('chunk_index', 0)}",
                            "title": meta.get("section_path") or meta.get("book_title", "Учебник"),
                            "nosology": meta.get("nosology", ""),
                            "organ": meta.get("organ", ""),
                            "text": chunk_data.get("text", ""),
                            "source": meta.get("book_title") or chunk_file.stem,
                            "page_reference": f"стр. {meta.get('page_start', '-')}",
                            "category": "textbook_chunk",
                        })
            except Exception as e:
                log.debug("Error reading chunks from %s: %s", chunk_file, e)

    return docs


def retrieve(
    query: str,
    filters: dict[str, Any] | None = None,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """Единый метод поиска по базе знаний патоморфологии.

    Сначала пробует VectorStoreIndex Qdrant (если запущен сервис Qdrant),
    иначе выполняет ранжирование по локальному корпусу знаний.
    """
    # 1. Попытка поиска через Qdrant + LlamaIndex
    try:
        from llama_index.core import VectorStoreIndex

        from ..index import get_embed_model, get_vector_store

        vs = get_vector_store()
        embed_model = get_embed_model()
        index = VectorStoreIndex.from_vector_store(vs, embed_model=embed_model)
        retriever = index.as_retriever(similarity_top_k=top_k)
        hits = retriever.retrieve(query)
        if hits:
            return [
                {
                    "text": h.node.text,
                    "score": getattr(h, "score", 0.0),
                    "title": h.node.metadata.get("book_title") or h.node.metadata.get("section_path", ""),
                    "source": h.node.metadata.get("book_title", "Qdrant Index"),
                    "page_reference": f"стр. {h.node.metadata.get('page_start', '-')}",
                    "nosology": h.node.metadata.get("nosology", ""),
                    "organ": h.node.metadata.get("organ", ""),
                }
                for h in hits
            ]
    except Exception as e:
        log.debug("Qdrant retriever not active or skipped: %s", e)

    # 2. Локальный гибридный поиск по эталонному корпусу знаний
    corpus = _load_knowledge_corpus()
    q = query.lower()
    q_words = [w.strip(".,;:?!()\"'") for w in q.split() if len(w.strip(".,;:?!()\"'")) >= 2]

    scored = []
    for doc in corpus:
        score = 0.0
        text = doc.get("text", "").lower()
        title = doc.get("title", "").lower()
        nosology = doc.get("nosology", "").lower()
        organ = doc.get("organ", "").lower()

        # Применение фильтров
        if filters:
            if "nosology" in filters and filters["nosology"].lower() not in nosology:
                continue
            if "organ" in filters and filters["organ"].lower() not in organ:
                continue

        # Взвешенное сопоставление терминов
        for term in q_words:
            if term in title:
                score += 8.0
            if term in nosology:
                score += 6.0
            if term in organ:
                score += 4.0
            if term in text:
                score += 2.0
                # Добавка за частоту вхождения
                count = text.count(term)
                score += min(count * 0.5, 4.0)

        # Бонус за точные ключевые фразы
        if "ki-67" in q or "ki67" in q:
            if "ki-67" in text or "ki67" in text or "пролифератив" in text:
                score += 10.0
        if "her2" in q or "her-2" in q or "хер2" in q:
            if "her2" in text or "erbb2" in text:
                score += 10.0
        if "люминальн" in q and "люминальн" in text:
            score += 8.0
        if "опеченен" in q and "опеченен" in text:
            score += 10.0
        if "стеатоз" in q and "стеатоз" in text:
            score += 10.0
        if "атеросклероз" in q and "атеросклероз" in text:
            score += 10.0

        if score > 0:
            scored.append({**doc, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    results = scored[:top_k]

    # Если точных совпадений нет, но база есть — вернем первые элементы релевантной категории
    if not results and corpus:
        results = [{**corpus[0], "score": 1.0}]

    return results
