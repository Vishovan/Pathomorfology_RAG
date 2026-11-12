"""Поиск патоморфологических микрофотографий и ссылок на WSI-слайды."""
import json
import logging
import re
from functools import lru_cache
from typing import Any

from ..config import KNOWLEDGE_DIR, SETTINGS
from ..text import tokens

log = logging.getLogger("pathrag.retrieval.image")
_WORD_RE = re.compile(r"[a-zа-яё0-9][a-zа-яё0-9\-+&]*")

# Вес совпадения основы слова запроса с полем записи
_FIELD_WEIGHTS = (("diagnosis", 5), ("title", 4), ("keywords", 4), ("stain", 3), ("organ", 3), ("caption", 2))
# Служебные слова запросов на поиск изображений: не несут смысла, но встречаются в подписях
_NOISE_STEMS = frozenset(tokens(
    "микропрепарат препарат микрофото микрофотография фото фотография снимок слайд wsi "
    "изображение картинка найди открой продемонстрируй атлас"
))


@lru_cache(maxsize=1)
def _load_local_images() -> tuple[dict[str, Any], ...]:
    path = KNOWLEDGE_DIR / "images_db.json"
    if not path.exists():
        return ()
    return tuple(json.loads(path.read_text(encoding="utf-8")))


def load_local_images() -> list[dict[str, Any]]:
    return [dict(img) for img in _load_local_images()]


def _qdrant_images(query: str, k: int, **filters: str | None) -> list[dict[str, Any]]:
    try:
        from ..index import qdrant_collection_ready

        if not qdrant_collection_ready(SETTINGS.images_collection):
            return []
        from ..image_search import search_images

        return search_images(query, k=k, **filters)
    except Exception as e:
        log.debug("Qdrant image search skipped or unavailable: %s", e)
        return []


def _field_words(img: dict[str, Any]) -> dict[str, list[str]]:
    raw = {
        "diagnosis": img.get("diagnosis", ""),
        "title": img.get("title", ""),
        "keywords": " ".join(img.get("keywords", [])),
        "stain": img.get("stain", ""),
        "organ": img.get("organ", ""),
        "caption": img.get("caption", ""),
    }
    return {k: _WORD_RE.findall(v.lower().replace("ё", "е")) for k, v in raw.items()}


def _score(img: dict[str, Any], q_stems: list[str]) -> int:
    """Сумма весов полей, где какое-либо слово НАЧИНАЕТСЯ с основы из запроса.

    Совпадения только по коротким (3 символа) основам вроде «рак» не делают снимок релевантным:
    иначе к «раку желудка» прикладывались бы препараты рака молочной железы.
    """
    words = _field_words(img)
    score, has_specific = 0, False
    for s in q_stems:
        for name, weight in _FIELD_WEIGHTS:
            if any(w.startswith(s) for w in words[name]):
                score += weight
                has_specific = has_specific or len(s) >= 4
    return score if has_specific else 0


def search_pathology_images(
    query: str,
    k: int = 4,
    stain: str | None = None,
    organ: str | None = None,
    magnification: str | None = None,
) -> list[dict[str, Any]]:
    """Ищет микропрепараты по тексту запроса и метаданным.

    Возвращает только релевантные изображения: если совпадений нет, список пуст
    (чтобы к вопросу, например, о туберкулёзе не прикладывались снимки HER2).
    """
    qdrant_results = _qdrant_images(query, k, stain=stain, organ=organ, magnification=magnification)
    if qdrant_results:
        return qdrant_results

    q_stems = [s for s in tokens(query) if s not in _NOISE_STEMS]
    scored = []
    for img in load_local_images():
        if stain and stain.lower() not in img.get("stain", "").lower():
            continue
        if organ and organ.lower() not in img.get("organ", "").lower():
            continue
        if magnification and magnification.lower() not in img.get("magnification", "").lower():
            continue
        score = _score(img, q_stems) if q_stems else 0
        if score > 0 or not query.strip():
            scored.append((score, img))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [img for _, img in scored[:k]]
