"""Чтение реестра sources.yaml."""
import yaml

from .config import SETTINGS

_META_KEYS = {"version", "project", "updated", "defaults", "metadata_schema"}


def load_sources() -> dict[str, dict]:
    with open(SETTINGS.sources_file, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    out = {}
    for key, value in raw.items():
        if key in _META_KEYS or not isinstance(value, list):
            continue
        for item in value:
            item = dict(item)
            item["group"] = key
            out[item["id"]] = item
    return out


def get_source(source_id: str) -> dict:
    sources = load_sources()
    if source_id not in sources:
        raise KeyError(f"Источник '{source_id}' не найден в sources.yaml. Доступны: {', '.join(sources)}")
    return sources[source_id]
