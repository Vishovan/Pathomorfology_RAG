"""Манифесты ингеста: что уже обработано и проиндексировано, кэш чанков и записей изображений."""
import hashlib
import json
from pathlib import Path

from .blocks import Chunk
from .config import SETTINGS

CACHE_DIR = SETTINGS.processed_dir / "chunks"
IMG_CACHE_DIR = SETTINGS.processed_dir / "images"
TEXT_MANIFEST = "manifest.json"
IMAGES_MANIFEST = "images_manifest.json"


def load(name: str = TEXT_MANIFEST) -> dict:
    p = SETTINGS.processed_dir / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save(data: dict, name: str = TEXT_MANIFEST) -> None:
    p = SETTINGS.processed_dir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def make_doc_id(source_id: str, key: str) -> str:
    """Стабильный id документа: при изменении файла/страницы старые данные заменяются."""
    return hashlib.sha1(f"{source_id}:{key}".encode()).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_chunks(doc_id: str, chunks: list[Chunk]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / f"{doc_id}.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"text": c.text, "metadata": c.metadata}, ensure_ascii=False) + "\n")


def load_chunks(doc_id: str) -> list[Chunk] | None:
    p = CACHE_DIR / f"{doc_id}.jsonl"
    if not p.exists():
        return None
    return [Chunk(**json.loads(line)) for line in p.read_text(encoding="utf-8").splitlines() if line]


def save_records(doc_id: str, records: list[dict]) -> None:
    IMG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(IMG_CACHE_DIR / f"{doc_id}.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_records(doc_id: str) -> list[dict] | None:
    p = IMG_CACHE_DIR / f"{doc_id}.jsonl"
    if not p.exists():
        return None
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]
