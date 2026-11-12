import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"


def _env(name: str, default: str | None = None):
    """Фабрика значения из окружения: читается при создании Settings, а не при импорте модуля."""
    return field(default_factory=lambda: os.getenv(name) or default)


@dataclass(frozen=True)
class Settings:
    qdrant_url: str = _env("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = _env("QDRANT_API_KEY")
    qdrant_path: str | None = _env("QDRANT_PATH")  # локальный режим без Docker
    collection: str = _env("PATHRAG_COLLECTION", "pathology_text")
    embed_model: str = _env("PATHRAG_EMBED_MODEL", "BAAI/bge-m3")
    device: str = _env("PATHRAG_DEVICE", "cpu")
    embed_batch_size: int = field(default_factory=lambda: int(os.getenv("PATHRAG_EMBED_BATCH", "16")))
    embed_max_tokens: int = 1024

    # чанкинг (в символах)
    chunk_max_chars: int = 1800
    chunk_overlap_chars: int = 200
    chunk_min_chars: int = 200
    split_level: int = 3  # заголовки глубже склеиваются в текст

    ocr_langs: tuple = ("ru", "en")

    # изображения (этап 3)
    images_collection: str = _env("PATHRAG_IMAGES_COLLECTION", "pathology_images")
    # "vinid/plip" включает векторный поиск по самим изображениям; пусто = только по подписям
    image_model: str = _env("PATHRAG_IMAGE_MODEL", "")
    images_dir: Path = ROOT / "data" / "images"
    thumb_max_px: int = 768
    min_image_px: int = 120

    sources_file: Path = ROOT / "sources.yaml"
    crawl_file: Path = ROOT / "configs" / "crawl.yaml"
    processed_dir: Path = ROOT / "data" / "processed"
    knowledge_dir: Path = KNOWLEDGE_DIR


SETTINGS = Settings()
