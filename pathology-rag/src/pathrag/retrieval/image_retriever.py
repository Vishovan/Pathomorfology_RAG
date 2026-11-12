"""Поиск патоморфологических микрофотографий и ссылок на WSI-слайды."""
import json
import logging
from pathlib import Path
from typing import Any

from ..config import SETTINGS

log = logging.getLogger("pathrag.retrieval.image")
KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"


def load_local_images() -> list[dict[str, Any]]:
    path = KNOWLEDGE_DIR / "images_db.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def search_pathology_images(
    query: str,
    k: int = 4,
    stain: str | None = None,
    organ: str | None = None,
    magnification: str | None = None,
) -> list[dict[str, Any]]:
    """Ищет микропрепараты по тексту запроса и метаданным."""
    # 1. Попытка поиска через векторный индекс Qdrant, если база доступна
    try:
        from ..image_search import search_images as qdrant_search_images

        qdrant_results = qdrant_search_images(
            query, k=k, stain=stain, organ=organ, magnification=magnification
        )
        if qdrant_results:
            return qdrant_results
    except Exception as e:
        log.debug("Qdrant image search skipped or unavailable: %s", e)

    # 2. Локальный поиск по эталонной базе изображений (fallback & demo)
    images = load_local_images()
    q = query.lower()
    scored = []

    for img in images:
        score = 0
        diag = img.get("diagnosis", "").lower()
        title = img.get("title", "").lower()
        cap = img.get("caption", "").lower()
        img_stain = img.get("stain", "").lower()
        img_organ = img.get("organ", "").lower()
        kw = [w.lower() for w in img.get("keywords", [])]

        # Фильтры
        if stain and stain.lower() not in img_stain:
            continue
        if organ and organ.lower() not in img_organ:
            continue
        if magnification and magnification.lower() not in img.get("magnification", "").lower():
            continue

        # Подсчет совпадений
        for word in q.split():
            clean_word = word.strip(".,;:?!()")
            if len(clean_word) < 3:
                continue
            if clean_word in diag:
                score += 5
            if clean_word in title:
                score += 4
            if any(clean_word in k for k in kw):
                score += 4
            if clean_word in cap:
                score += 2
            if clean_word in img_stain:
                score += 3
            if clean_word in img_organ:
                score += 3

        if score > 0 or not q.strip():
            scored.append((score, img))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [s[1] for s in scored[:k]]

    # Если точных совпадений мало, но запрос медицинский — возвращаем наиболее релевантные
    if not results and images:
        results = images[:k]

    return results
