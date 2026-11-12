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

    # 4. Открытый корпус (Wikipedia CC BY-SA, Europe PMC CC BY), коммитится в репозиторий
    corpus_dir = KNOWLEDGE_DIR / "corpus"
    if corpus_dir.exists():
        for chunk_file in sorted(corpus_dir.glob("*.jsonl")):
            try:
                with open(chunk_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        meta = row.get("metadata", {})
                        docs.append({
                            "id": f"{chunk_file.stem}_{meta.get('section_path', '')}_{meta.get('chunk_index', 0)}",
                            "title": meta.get("section_path") or meta.get("book_title", ""),
                            "nosology": "",
                            "organ": "",
                            "text": row.get("text", ""),
                            "source": meta.get("book_title", chunk_file.stem),
                            "page_reference": f"{meta.get('license', '')} · {meta.get('url', '')}".strip(" ·"),
                            "url": meta.get("url"),
                            "license": meta.get("license"),
                            "category": meta.get("source_type", "open_corpus"),
                        })
            except Exception as e:
                log.debug("Error reading corpus %s: %s", chunk_file, e)

    return docs


_CORPUS_CACHE: list[dict[str, Any]] | None = None


def _corpus() -> list[dict[str, Any]]:
    global _CORPUS_CACHE
    if _CORPUS_CACHE is None:
        _CORPUS_CACHE = _load_knowledge_corpus()
    return _CORPUS_CACHE


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

    # 2. Локальный BM25 по эталонному корпусу (curated JSON + открытый корпус)
    corpus = _corpus()
    bm = _bm25_index()
    q = query.lower()
    q_terms = [t for t in _tokens(q) if t in bm["df"]]

    scored = []
    for i, doc in enumerate(corpus):
        if filters:
            if "nosology" in filters and filters["nosology"].lower() not in (doc.get("nosology") or "").lower():
                continue
            if "organ" in filters and filters["organ"].lower() not in (doc.get("organ") or "").lower():
                continue
        tf, dl, title_terms = bm["tf"][i], bm["len"][i], bm["title"][i]
        score = 0.0
        for t in q_terms:
            f = tf.get(t, 0)
            idf = bm["idf"][t]
            if f:
                score += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * dl / bm["avgdl"]))
            if t in title_terms:
                score += idf * 1.5
        if score <= 0:
            continue
        text = (doc.get("text") or "").lower()
        # Доменные бонусы (сохраняем поведение для РМЖ/ИГХ)
        if ("ki-67" in q or "ki67" in q) and ("ki-67" in text or "ki67" in text):
            score += 3.0
        if any(k in q for k in ("her2", "her-2")) and ("her2" in text or "erbb2" in text):
            score += 3.0
        # Верифицированные учебные темы — небольшой приоритет перед энциклопедией
        if doc.get("category") in ("breast_cancer_ihc", "general_pathology", "textbook_chunk"):
            score *= 1.25
        scored.append({**doc, "score": round(score, 3)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    results = scored[:top_k]
    if not results and corpus:
        results = [{**corpus[0], "score": 0.0}]
    return results


_STOP = set("""и в во на не что как это для по из от до при или но а же ли бы то так все его её их он она они
мы вы я ты чем чём кто где когда какой какие каков какова каковы который которые также только уже
есть быть был была были этот эта эти тот та те с со к ко о об у за над под без между через про
опишите опиши расскажите расскажи объясните объясни назовите перечислите покажи покажите дайте
отличается отличие отличия разница сравните чем-то например очень более менее
the a an of and or in on to for with by is are was were be what how which who why does do from
describe explain list show tell about between difference vs versus""".split())


def _stem(w: str) -> str:
    # Грубый стемминг: обрезка окончаний (рус) и суффиксов (англ) до основы 4–6 символов
    if any("а" <= ch <= "я" or ch == "ё" for ch in w):
        w = w.replace("ё", "е")
        return w[:6] if len(w) > 6 else w
    for suf in ("ies", "es", "s", "ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            w = w[: -len(suf)]
            break
    return w[:7]


def _tokens(text: str) -> list[str]:
    import re as _re
    out = []
    for w in _re.findall(r"[a-zа-яё0-9][a-zа-яё0-9\-]+", text.lower()):
        if w in _STOP or len(w) < 3:
            continue
        out.append(_stem(w))
    return out


_BM25: dict[str, Any] | None = None


def _bm25_index() -> dict[str, Any]:
    global _BM25
    if _BM25 is None:
        import math
        from collections import Counter

        corpus = _corpus()
        tfs, lens, titles, df = [], [], [], Counter()
        for d in corpus:
            toks = _tokens(f"{d.get('title', '')} {d.get('nosology', '')} {d.get('organ', '')} {d.get('text', '')}")
            c = Counter(toks)
            tfs.append(c)
            lens.append(len(toks) or 1)
            titles.append(set(_tokens(f"{d.get('title', '')} {d.get('nosology', '')}")))
            df.update(c.keys())
        n = max(len(corpus), 1)
        _BM25 = {
            "tf": tfs, "len": lens, "title": titles, "df": df,
            "avgdl": sum(lens) / n,
            "idf": {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()},
        }
    return _BM25
