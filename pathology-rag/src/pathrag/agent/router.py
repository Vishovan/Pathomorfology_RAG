"""Классификатор намерений (Router) для мультимодального графа."""
import re


def route_intent(query: str, explicit_mode: str | None = None) -> str:
    """Определяет маршрут запроса:
    - 'legal_gost'
    - 'image_search'
    - 'literature_search'
    - 'breast_cancer_ihc'
    - 'pathology_qa'
    """
    if explicit_mode:
        mode_clean = explicit_mode.lower().strip()
        if mode_clean in ("legal", "law", "gost", "юрист", "гост"):
            return "legal_gost"
        if mode_clean in ("images", "image", "photo", "фото", "препарат"):
            return "image_search"
        if mode_clean in ("literature", "pubmed", "arxiv", "web", "литература"):
            return "literature_search"
        if mode_clean in ("breast_cancer", "рмж", "her2", "ki67"):
            return "breast_cancer_ihc"
        if mode_clean in ("pathology", "патология"):
            return "pathology_qa"

    q = query.lower()

    # 1. Проверка на юридический режим / ГОСТы
    legal_keywords = [
        "гост", "gost", "34.602", "59793", "34.201", "59795", "59792", "12207",
        "25010", "56939", "52636", "56849", "51583", "27001", "роспатент", "минцифры",
        "росздравнадзор", "медицинское изделие", "медиздели", "samd", "реестр российского по",
        "реестр по", "программа для эвм", "техническое задание", "пмти", "испытания ас",
        "безопасная разработка", "юрист", "нормативн", "регулятор", "сертификация",
        "не для постановки диагноза", "дисклеймер"
    ]
    if any(k in q for k in legal_keywords):
        return "legal_gost"

    # 2. Проверка на поиск изображений / микрофотографий
    image_keywords = [
        "покажи", "микрофото", "фотографи", "фото", "снимок",
        "слайд", "wsi", "картинк", "атлас препаратов", "открой препарат",
        "найди фото", "покажите", "продемонстрируй препарат"
    ]
    if any(k in q for k in image_keywords) and not ("опишите" in q or "описать" in q or "стадии" in q):
        return "image_search"

    # 3. Проверка на поиск внешней литературы / PubMed / препринты / веб
    lit_keywords = [
        "pubmed", "arxiv", "biorxiv", "препринт", "стать", "публикаци", "поиск в интернете",
        "веб-поиск", "найди в интернете", "найди в сети", "свежие исследования"
    ]
    if any(k in q for k in lit_keywords):
        return "literature_search"

    # 4. Проверка на специализацию: Рак молочной железы и ИГХ
    brca_keywords = [
        "рак молочной железы", "рмж", "молочн", "her2", "her-2", "ki-67", "ki67",
        "индекс пролиферации", "рецептор эстроген", "рецептор прогестерон", "allred",
        "люминальный", "люминальн", "тройной негативный", "трижды негативный", "tnbc",
        "nottingham", "ноттингем", "дольковый рак", "протоковый рак", "e-cadherin",
        "кадгерин", "asco/cap", "fish", "sish", "хер2", "ки67"
    ]
    if any(k in q for k in brca_keywords):
        return "breast_cancer_ihc"

    # 5. По умолчанию — общая патоморфология
    return "pathology_qa"
