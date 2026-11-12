"""Гибридный ретривер: векторный поиск в Qdrant (если доступен) + локальный BM25, слияние через RRF.

Единый контракт:
    retrieve(query: str, filters: dict | None = None, top_k: int = 4) -> list[dict]
"""
import json
import logging
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..config import KNOWLEDGE_DIR, SETTINGS
from ..text import tokens as _tokens

log = logging.getLogger("pathrag.retrieval.hybrid")

# Курируемые темы: (файл, категория, источник по умолчанию, ссылка по умолчанию)
_CURATED = [
    ("breast_cancer.json", "breast_cancer_ihc", "Клинические рекомендации", "Раздел 1.4"),
    ("general_pathology.json", "general_pathology", "Учебник по патологической анатомии", "Гл. 14"),
]
# Верифицированные учебные темы получают небольшой приоритет перед энциклопедией
_PRIORITY_CATEGORIES = {"breast_cancer_ihc", "general_pathology", "textbook_chunk"}
_RRF_K = 60


def _iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _load_knowledge_corpus() -> list[dict[str, Any]]:
    """Загружает эталонные тексты из базы знаний и кэшированных чанков."""
    docs: list[dict[str, Any]] = []

    # 1–2. Курируемые темы (РМЖ/ИГХ, общая патология)
    for fname, category, default_source, default_ref in _CURATED:
        path = KNOWLEDGE_DIR / fname
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        source = data.get("source", default_source)
        for t in data.get("topics", []):
            docs.append({
                "id": t.get("id"),
                "title": t.get("title"),
                "nosology": t.get("nosology"),
                "organ": t.get("organ"),
                "text": t.get("content"),
                "source": source,
                "page_reference": t.get("page_reference", default_ref),
                "category": category,
            })

    # 3. Кэшированные чанки из data/processed/chunks/
    chunks_dir = SETTINGS.processed_dir / "chunks"
    if chunks_dir.exists():
        for chunk_file in chunks_dir.glob("*.jsonl"):
            try:
                for row in _iter_jsonl(chunk_file):
                    meta = row.get("metadata", {})
                    docs.append({
                        "id": f"{chunk_file.stem}_{meta.get('chunk_index', 0)}",
                        "title": meta.get("section_path") or meta.get("book_title", "Учебник"),
                        "nosology": meta.get("nosology", ""),
                        "organ": meta.get("organ", ""),
                        "text": row.get("text", ""),
                        "source": meta.get("book_title") or chunk_file.stem,
                        "page_reference": f"стр. {meta.get('page_start', '-')}",
                        "category": "textbook_chunk",
                    })
            except Exception as e:
                log.warning("Error reading chunks from %s: %s", chunk_file, e)

    # 4. Открытый корпус (Wikipedia CC BY-SA, Europe PMC CC BY), коммитится в репозиторий
    corpus_dir = KNOWLEDGE_DIR / "corpus"
    if corpus_dir.exists():
        for chunk_file in sorted(corpus_dir.glob("*.jsonl")):
            try:
                for row in _iter_jsonl(chunk_file):
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
                log.warning("Error reading corpus %s: %s", chunk_file, e)

    return docs


@lru_cache(maxsize=1)
def _corpus() -> list[dict[str, Any]]:
    return _load_knowledge_corpus()


@lru_cache(maxsize=1)
def _bm25_index() -> dict[str, Any]:
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
    return {
        "tf": tfs, "len": lens, "title": titles, "df": df,
        "avgdl": sum(lens) / n,
        "idf": {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()},
    }


def _passes_filters(doc: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    for key in ("nosology", "organ"):
        if key in filters and str(filters[key]).lower() not in (doc.get(key) or "").lower():
            return False
    return True


def _bm25_search(query: str, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
    corpus = _corpus()
    bm = _bm25_index()
    q = query.lower()
    q_terms = [t for t in _tokens(q) if t in bm["df"]]

    scored = []
    for i, doc in enumerate(corpus):
        if not _passes_filters(doc, filters):
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
        # Доменные бонусы для РМЖ/ИГХ
        if ("ki-67" in q or "ki67" in q) and ("ki-67" in text or "ki67" in text):
            score += 3.0
        if any(k in q for k in ("her2", "her-2")) and ("her2" in text or "erbb2" in text):
            score += 3.0
        if doc.get("category") in _PRIORITY_CATEGORIES:
            score *= 1.25
        if score > 0.05:
            scored.append({**doc, "score": round(score, 3)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


@lru_cache(maxsize=4)
def _qdrant_retriever(top_k: int):
    from llama_index.core import VectorStoreIndex

    from ..index import get_embed_model, get_vector_store

    index = VectorStoreIndex.from_vector_store(get_vector_store(), embed_model=get_embed_model())
    return index.as_retriever(similarity_top_k=top_k)


def _qdrant_search(query: str, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
    """Векторный поиск в Qdrant. Пустой список, если Qdrant/LlamaIndex недоступны."""
    try:
        from ..index import qdrant_collection_ready

        if not qdrant_collection_ready(SETTINGS.collection):
            return []
        # Берём с запасом, т.к. фильтры применяются после поиска
        hits = _qdrant_retriever(top_k * 3 if filters else top_k).retrieve(query)
    except Exception as e:
        log.debug("Qdrant retriever not active or skipped: %s", e)
        return []

    out = []
    for h in hits:
        meta = h.node.metadata
        doc = {
            "id": h.node.node_id,
            "text": h.node.text,
            "score": getattr(h, "score", 0.0),
            "title": meta.get("section_path") or meta.get("book_title", ""),
            "source": meta.get("book_title", "Qdrant Index"),
            "page_reference": f"стр. {meta.get('page_start', '-')}" if meta.get("page_start") else meta.get("url", ""),
            "url": meta.get("url"),
            "nosology": meta.get("nosology", ""),
            "organ": meta.get("organ", ""),
            "category": "textbook_chunk",
        }
        if _passes_filters(doc, filters):
            out.append(doc)
    return out[:top_k]


def _doc_key(doc: dict[str, Any]) -> str:
    return " ".join((doc.get("text") or "").split())[:200]


def _rrf_merge(*ranked_lists: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion: шкалы BM25 и косинусной близости несопоставимы, сливаем по рангам."""
    fused: dict[str, tuple[float, dict[str, Any]]] = {}
    for lst in ranked_lists:
        for rank, doc in enumerate(lst):
            key = _doc_key(doc)
            prev_score, prev_doc = fused.get(key, (0.0, doc))
            fused[key] = (prev_score + 1.0 / (_RRF_K + rank + 1), prev_doc)
    ordered = sorted(fused.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ordered[:top_k]]


def retrieve(
    query: str,
    filters: dict[str, Any] | None = None,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """Единый метод поиска по базе знаний патоморфологии.

    Локальный BM25 по курируемой базе и открытому корпусу выполняется всегда;
    если доступна коллекция Qdrant, её выдача сливается с локальной через RRF.
    """
    local_hits = _bm25_search(query, filters, top_k)
    vector_hits = _qdrant_search(query, filters, top_k)
    if not vector_hits:
        return local_hits
    return _rrf_merge(vector_hits, local_hits, top_k=top_k)
