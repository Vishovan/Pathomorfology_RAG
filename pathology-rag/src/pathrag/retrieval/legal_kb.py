"""Юридическая и нормативная база по ГОСТам и регуляторике медицинского ПО в РФ."""
import copy
import json
from functools import lru_cache
from typing import Any

from ..config import KNOWLEDGE_DIR


@lru_cache(maxsize=1)
def _legal_data() -> dict[str, Any]:
    file_path = KNOWLEDGE_DIR / "gost_legal.json"
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


def load_legal_data() -> dict[str, Any]:
    """Нормативная база (кэшируется в памяти; возвращается копия, чтобы вызывающий код не испортил кэш)."""
    return copy.deepcopy(_legal_data())


def search_gost_standards(query: str) -> list[dict[str, Any]]:
    """Поиск по стандартам ГОСТ и нормативным процедурам."""
    data = _legal_data()  # только чтение: результаты собираются в новые словари
    q = query.lower()
    results = []

    # Словарь синонимов и соответствий тем конкретным ГОСТам
    gost_synonyms = {
        "34.602": ["техническ", "задани", "тз", "требован"],
        "59793": ["стади", "этап", "жизненн", "создани"],
        "34.201": ["состав", "ведомост", "комплектн", "обозначен"],
        "59795": ["содержани", "структур", "описани"],
        "59792": ["испытан", "пмти", "приемк", "протокол"],
        "12207": ["жизненн", "цикл", "процесс", "разработк"],
        "25010": ["качеств", "надежност", "square", "производительност"],
        "56939": ["безопасн", "уязвимост", "защит", "разработк", "sast", "секьюр"],
        "52636": ["электронн", "истори", "болезн", "мис", "лис"],
        "56849": ["безопасност", "медицинск", "здоровь"],
        "51583": ["защищенн", "исполнени", "контур"],
        "27001": ["смиб", "менеджмент", "информационн"],
    }

    # 1. Поиск по ГОСТам
    for standard in data.get("gost_standards", []):
        score = 0
        gost_name = standard.get("gost", "").lower()
        title = standard.get("title", "").lower()
        summary = standard.get("summary", "").lower()
        practical = standard.get("practical_value", "").lower()

        # Проверка по синонимам
        for gost_code, stems in gost_synonyms.items():
            if gost_code in gost_name:
                for stem in stems:
                    if stem in q:
                        score += 15

        for term in q.split():
            term_clean = term.strip(".,;:?!()")
            if not term_clean or len(term_clean) < 3:
                continue
            stem = term_clean[:5]
            if stem in gost_name:
                score += 12
            if stem in title:
                score += 6
            if stem in practical or stem in summary:
                score += 3

        if score > 0 or any(t in gost_name for t in ["34.", "5979", "12207", "25010", "56939", "52636", "56849", "51583", "27001"] if t in q):
            results.append({**standard, "relevance_score": score, "type": "gost"})

    # 2. Поиск по процедурам (Роспатент, Минцифры, Росздравнадзор)
    for proc in data.get("procedures", []):
        direction = proc.get("direction", "").lower()
        auth = proc.get("authority", "").lower()
        purpose = proc.get("purpose", "").lower()

        score = 0
        for term in q.split():
            term_clean = term.strip(".,;:?!()")
            if not term_clean or len(term_clean) < 2:
                continue
            if term_clean in auth or term_clean in direction:
                score += 8
            if term_clean in purpose:
                score += 3

        if score > 0 or any(w in q for w in ["роспатент", "минцифры", "росздравнадзор", "медиздели", "samd", "реестр"]):
            results.append({
                "gost": proc.get("authority"),
                "title": proc.get("direction"),
                "summary": proc.get("result"),
                "practical_value": f"{proc.get('purpose')} Требования: {proc.get('requirements')}",
                "relevance_score": score,
                "type": "procedure"
            })

    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results


def get_medical_device_boundary_info() -> dict[str, Any]:
    """Возвращает правила разграничения справочной системы и медицинского изделия."""
    return copy.deepcopy(_legal_data().get("medical_device_boundary", {}))
