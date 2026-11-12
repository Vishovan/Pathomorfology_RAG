"""Классификатор намерений (Router) для мультимодального графа.

Ключевые слова — это начала слов (основы): «молочн» совпадёт с «молочной», но не с серединой слова.
Порядок проверок важен: юридический → изображения → литература → РМЖ → разговор → общая патология.
"""
import re

_EXPLICIT_MODES = {
    "legal_gost": ("legal", "law", "gost", "юрист", "гост"),
    "image_search": ("images", "image", "photo", "фото", "препарат"),
    "literature_search": ("literature", "pubmed", "arxiv", "web", "литература"),
    "breast_cancer_ihc": ("breast_cancer", "рмж", "her2", "ki67"),
    "pathology_qa": ("pathology", "патология"),
}

LEGAL_KEYWORDS = (
    "гост", "gost", "34.602", "59793", "34.201", "59795", "59792", "12207",
    "25010", "56939", "52636", "56849", "51583", "27001", "роспатент", "минцифры",
    "росздравнадзор", "медицинское изделие", "медиздели", "samd", "реестр российского по",
    "реестр по", "программа для эвм", "техническое задание", "пмти", "испытания ас",
    "безопасная разработка", "юрист", "нормативн", "регулятор", "сертификация",
    "не для постановки диагноза", "дисклеймер",
)
IMAGE_KEYWORDS = (
    "покажи", "микрофото", "фотографи", "фото", "снимок",
    "слайд", "wsi", "картинк", "атлас препаратов", "открой препарат",
    "найди фото", "покажите", "продемонстрируй препарат",
)
# Слова, при которых запрос с «покажи/фото» — всё же вопрос на описание, а не поиск снимка
IMAGE_EXCLUSIONS = ("опишите", "описать", "стадии")
LITERATURE_KEYWORDS = (
    "pubmed", "arxiv", "biorxiv", "препринт", "стать", "публикаци", "поиск в интернете",
    "веб-поиск", "найди в интернете", "найди в сети", "свежие исследования",
)
BREAST_CANCER_KEYWORDS = (
    "рак молочной железы", "рмж", "молочн", "her2", "her-2", "ki-67", "ki67",
    "индекс пролиферации", "рецептор эстроген", "рецептор прогестерон", "allred",
    "люминальн", "тройной негативный", "трижды негативный", "tnbc",
    "nottingham", "ноттингем", "дольковый рак", "протоковый рак", "e-cadherin",
    "кадгерин", "asco/cap", "fish", "sish", "хер2", "ки67",
)
GREETING_TOKENS = frozenset({
    "привет", "здравствуй", "здравствуйте", "добрый", "салют", "hello", "hi", "hey",
    "помощь", "help", "меню", "start", "старт", "начать",
})
ABOUT_PHRASES = ("кто ты", "что ты умеешь", "как тебя зовут", "расскажи о себе", "твои функции")


def _prefix_matcher(keywords: tuple[str, ...]) -> re.Pattern[str]:
    # (?<!\w) — ключ должен начинаться с начала слова; окончание слова может быть любым
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(k) for k in keywords) + ")")


_LEGAL_RE = _prefix_matcher(LEGAL_KEYWORDS)
_IMAGE_RE = _prefix_matcher(IMAGE_KEYWORDS)
_IMAGE_EXCL_RE = _prefix_matcher(IMAGE_EXCLUSIONS)
_LITERATURE_RE = _prefix_matcher(LITERATURE_KEYWORDS)
_BRCA_RE = _prefix_matcher(BREAST_CANCER_KEYWORDS)
_ABOUT_RE = _prefix_matcher(ABOUT_PHRASES)


def _explicit_intent(mode: str) -> str | None:
    m = mode.lower().strip()
    return next((intent for intent, aliases in _EXPLICIT_MODES.items() if m in aliases), None)


def _is_conversational(q: str) -> bool:
    words = [w.strip(".,;:?!()") for w in q.split()]
    words = [w for w in words if w]
    if len(words) <= 5 and any(w in GREETING_TOKENS for w in words[:2]):
        return True
    return bool(_ABOUT_RE.search(q))


def route_intent(query: str, explicit_mode: str | None = None) -> str:
    """Определяет маршрут запроса: legal_gost | image_search | literature_search |
    breast_cancer_ihc | conversational | pathology_qa."""
    if explicit_mode:
        intent = _explicit_intent(explicit_mode)
        if intent:
            return intent

    q = query.lower()
    if _LEGAL_RE.search(q):
        return "legal_gost"
    if _IMAGE_RE.search(q) and not _IMAGE_EXCL_RE.search(q):
        return "image_search"
    if _LITERATURE_RE.search(q):
        return "literature_search"
    if _BRCA_RE.search(q):
        return "breast_cancer_ihc"
    if _is_conversational(q):
        return "conversational"
    return "pathology_qa"
