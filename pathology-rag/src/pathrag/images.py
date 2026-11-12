"""Записи об изображениях («препарат -> описание»), миниатюры."""
import uuid
from pathlib import Path

from . import imeta
from .chunking import detect_language
from .config import ROOT, SETTINGS


def make_thumb(img, dest: Path, max_px: int | None = None) -> Path:
    img = img.convert("RGB")
    img.thumbnail((max_px or SETTINGS.thumb_max_px,) * 2)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, "JPEG", quality=85)
    return dest


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def make_record(*, source_id: str, doc_key: str, idx: int, caption: str, default_lang: str,
                section_path: str | None = None, nosology: str | None = None,
                overrides: dict | None = None, extra: dict | None = None) -> dict | None:
    """Собирает запись. Возвращает None, если нечем описать изображение (нет ни подписи, ни контекста)."""
    caption = (caption or "").strip()
    context = " ".join(x for x in (section_path, nosology) if x)
    if not caption and not context and not any((overrides or {}).values()):
        return None
    caption_source = "caption" if caption else "heading"
    text_for_meta = f"{caption} {context}".strip()

    diagnosis = imeta.extract_diagnosis(caption) if caption else None
    diagnosis_source = "caption" if diagnosis else None
    if not diagnosis and nosology:
        diagnosis, diagnosis_source = nosology, "heading"

    stain = imeta.extract_stain(text_for_meta)
    magnification = imeta.extract_magnification(caption)
    organ = imeta.extract_organ(text_for_meta)

    ov = {k: v for k, v in (overrides or {}).items() if v}  # ручные значения (например, из CSV) важнее эвристик
    if ov.get("diagnosis"):
        diagnosis, diagnosis_source = ov["diagnosis"], "manual"
    stain = ov.get("stain", stain)
    magnification = ov.get("magnification", magnification)
    organ = ov.get("organ", organ)

    parts, seen = [], set()
    for p in (diagnosis, nosology, organ, stain, magnification, caption or section_path):
        if p and p.lower() not in seen:
            seen.add(p.lower())
            parts.append(p)

    rec = {
        "image_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{doc_key}:{idx}")),
        "source_id": source_id,
        "caption": caption or None,
        "caption_source": caption_source,
        "diagnosis": diagnosis,
        "diagnosis_source": diagnosis_source,
        "stain": stain,
        "magnification": magnification,
        "organ": organ,
        "language": detect_language(text_for_meta, default_lang),
        "section_path": section_path,
        "search_text": " | ".join(parts),
    }
    rec.update(extra or {})
    return rec
