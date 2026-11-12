"""Коллекция изображений в Qdrant: вектор подписи (bge-m3) + необязательный вектор изображения (PLIP)."""
import logging
from functools import lru_cache

from .config import ROOT, SETTINGS
from .index import get_client, get_embed_model

log = logging.getLogger("pathrag.index_images")


# ---------- модель изображений (PLIP / любая CLIP-подобная) ----------
@lru_cache(maxsize=1)
def _clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(SETTINGS.image_model).to(SETTINGS.device).eval()
    proc = CLIPProcessor.from_pretrained(SETTINGS.image_model)
    return model, proc, torch


def image_dim() -> int:
    return _clip()[0].config.projection_dim


def _as_tensor(f):
    return f if hasattr(f, "norm") else getattr(f, "pooler_output", f)


def embed_images(paths: list[str]) -> list[list[float] | None]:
    """Векторы изображений; None для файлов, которых нет."""
    from PIL import Image

    model, proc, torch = _clip()
    out: list[list[float] | None] = [None] * len(paths)
    todo = [(i, ROOT / p) for i, p in enumerate(paths) if p and (ROOT / p).exists()]
    for s in range(0, len(todo), 16):
        batch = todo[s:s + 16]
        imgs = [Image.open(p).convert("RGB") for _, p in batch]
        inputs = proc(images=imgs, return_tensors="pt").to(SETTINGS.device)
        with torch.no_grad():
            f = _as_tensor(model.get_image_features(**inputs))
            f = f / f.norm(dim=-1, keepdim=True)
        for (i, _), v in zip(batch, f.cpu().tolist(), strict=True):
            out[i] = v
    return out


def embed_clip_text(text: str) -> list[float]:
    model, proc, torch = _clip()
    inputs = proc(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77).to(SETTINGS.device)
    with torch.no_grad():
        f = _as_tensor(model.get_text_features(**inputs))
        f = f / f.norm(dim=-1, keepdim=True)
    return f[0].cpu().tolist()


def image_channel_enabled() -> bool:
    return bool(SETTINGS.image_model)


def is_mostly_cyrillic(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum("а" <= c.lower() <= "я" or c.lower() == "ё" for c in letters) / len(letters) > 0.3


# ---------- коллекция ----------
def ensure_collection() -> None:
    from qdrant_client import models

    client, name = get_client(), SETTINGS.images_collection
    vectors = {"caption": models.VectorParams(size=len(get_embed_model().get_text_embedding("x")), distance=models.Distance.COSINE)}
    if image_channel_enabled():
        vectors["image"] = models.VectorParams(size=image_dim(), distance=models.Distance.COSINE)
    if client.collection_exists(name):
        have = client.get_collection(name).config.params.vectors
        if image_channel_enabled() and "image" not in have:
            raise SystemExit(
                f"Коллекция '{name}' создана без вектора изображений. Удалите её или задайте другой "
                "PATHRAG_IMAGES_COLLECTION, затем повторите ингест."
            )
        return
    client.create_collection(collection_name=name, vectors_config=vectors)
    for field in ("source_id", "doc_id", "stain", "magnification", "organ", "language", "diagnosis"):
        try:
            client.create_payload_index(name, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD)
        except Exception as e:
            log.debug("payload index %s: %s", field, e)


def delete_doc(doc_id: str) -> None:
    from qdrant_client import models

    try:
        get_client().delete(
            collection_name=SETTINGS.images_collection,
            points_selector=models.FilterSelector(filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])),
        )
    except Exception as e:
        log.debug("delete_doc skipped: %s", e)


def upsert_images(records: list[dict], doc_id: str, keep_thumbs: bool = True) -> int:
    from qdrant_client import models

    if not records:
        return 0
    ensure_collection()
    delete_doc(doc_id)
    cap_vecs = get_embed_model().get_text_embedding_batch([r["search_text"] for r in records], show_progress=True)
    img_vecs = embed_images([r.get("thumb_path") or "" for r in records]) if image_channel_enabled() else [None] * len(records)

    points = []
    for r, cv, iv in zip(records, cap_vecs, img_vecs, strict=True):
        vec = {"caption": cv}
        if iv is not None:
            vec["image"] = iv
        payload = {k: v for k, v in r.items() if k != "search_text"}
        payload["doc_id"] = doc_id
        payload["has_image_vector"] = iv is not None
        if not keep_thumbs:
            payload["thumb_path"] = None
        points.append(models.PointStruct(id=r["image_id"], vector=vec, payload=payload))
    client = get_client()
    for s in range(0, len(points), 64):
        client.upsert(SETTINGS.images_collection, points=points[s:s + 64], wait=True)

    if not keep_thumbs:  # миниатюры были нужны только для эмбеддинга
        for r in records:
            p = r.get("thumb_path")
            if p and (ROOT / p).exists():
                (ROOT / p).unlink()
            r["thumb_path"] = None
    return len(points)


def count_images() -> int | None:
    try:
        return get_client().count(SETTINGS.images_collection, exact=True).count
    except Exception:
        return None
