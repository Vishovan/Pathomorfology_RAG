import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    qdrant_path: str | None = os.getenv("QDRANT_PATH") or None  # локальный режим без Docker
    collection: str = os.getenv("PATHRAG_COLLECTION", "pathology_text")
    embed_model: str = os.getenv("PATHRAG_EMBED_MODEL", "BAAI/bge-m3")
    device: str = os.getenv("PATHRAG_DEVICE", "cpu")
    embed_batch_size: int = int(os.getenv("PATHRAG_EMBED_BATCH", "16"))
    embed_max_tokens: int = 1024

    # чанкинг (в символах)
    chunk_max_chars: int = 1800
    chunk_overlap_chars: int = 200
    chunk_min_chars: int = 200
    split_level: int = 3  # заголовки глубже склеиваются в текст

    ocr_langs: tuple = ("ru", "en")

    # изображения (этап 3)
    images_collection: str = os.getenv("PATHRAG_IMAGES_COLLECTION", "pathology_images")
    # "vinid/plip" включает векторный поиск по самим изображениям; пусто = только по подписям
    image_model: str = os.getenv("PATHRAG_IMAGE_MODEL", "")
    images_dir: Path = ROOT / "data" / "images"
    thumb_max_px: int = 768
    min_image_px: int = 120

    sources_file: Path = ROOT / "sources.yaml"
    crawl_file: Path = ROOT / "configs" / "crawl.yaml"
    processed_dir: Path = ROOT / "data" / "processed"


SETTINGS = Settings()
