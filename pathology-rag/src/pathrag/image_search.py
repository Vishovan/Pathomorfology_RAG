"""Поиск изображений по текстовому запросу. Это же будущий инструмент image_search для графа LangGraph.

Канал 1 (всегда): подпись + метаданные, вектор bge-m3, работает на ru и en.
Канал 2 (если задан PATHRAG_IMAGE_MODEL): вектор самого изображения (PLIP), только англоязычные запросы.
Каналы объединяются через RRF.
"""
from .config import SETTINGS
from .index import get_client, get_embed_model
from .index_images import embed_clip_text, image_channel_enabled, is_mostly_cyrillic

FIELDS = ["caption", "diagnosis", "stain", "magnification", "organ", "source_id", "book_title",
          "page", "page_url", "image_url", "thumb_path", "section_path", "language", "kind"]


def _filter(source_id=None, stain=None, magnification=None, organ=None):
    from qdrant_client import models

    must = [models.FieldCondition(key=k, match=models.MatchValue(value=v))
            for k, v in (("source_id", source_id), ("stain", stain), ("magnification", magnification), ("organ", organ)) if v]
    return models.Filter(must=must) if must else None


def search_images(query: str, k: int = 6, *, source_id: str | None = None, stain: str | None = None,
                  magnification: str | None = None, organ: str | None = None) -> list[dict]:
    from qdrant_client import models

    flt = _filter(source_id, stain, magnification, organ)
    qv = get_embed_model().get_query_embedding(query)
    use_image = image_channel_enabled() and not is_mostly_cyrillic(query)

    if use_image:
        res = get_client().query_points(
            SETTINGS.images_collection,
            prefetch=[
                models.Prefetch(query=qv, using="caption", limit=30, filter=flt),
                models.Prefetch(query=embed_clip_text(query), using="image", limit=30, filter=flt),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k, with_payload=True,
        )
    else:
        res = get_client().query_points(
            SETTINGS.images_collection, query=qv, using="caption",
            limit=k, query_filter=flt, with_payload=True,
        )
    return [{**{f: p.payload.get(f) for f in FIELDS}, "score": p.score, "image_id": str(p.id)} for p in res.points]
