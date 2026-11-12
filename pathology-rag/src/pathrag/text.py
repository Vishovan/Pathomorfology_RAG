"""Общая лёгкая токенизация для лексического поиска (BM25, грейдер, поиск изображений)."""
import re

STOPWORDS = frozenset("""и в во на не что как это для по из от до при или но а же ли бы то так все его её их он она они
мы вы я ты чем чём кто где когда какой какие каков какова каковы который которые также только уже
есть быть был была были этот эта эти тот та те с со к ко о об у за над под без между через про
опишите опиши расскажите расскажи объясните объясни назовите перечислите покажи покажите дайте
отличается отличие отличия разница сравните чем-то например очень более менее
the a an of and or in on to for with by is are was were be what how which who why does do from
describe explain list show tell about between difference vs versus""".split())

_WORD_RE = re.compile(r"[a-zа-яё0-9][a-zа-яё0-9\-]+")


def stem(w: str) -> str:
    """Грубый стемминг: обрезка до основы 6 символов (рус) или снятие суффиксов (англ)."""
    if any("а" <= ch <= "я" or ch == "ё" for ch in w):
        w = w.replace("ё", "е")
        return w[:6] if len(w) > 6 else w
    for suf in ("ies", "es", "s", "ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            w = w[: -len(suf)]
            break
    return w[:7]


def tokens(text: str) -> list[str]:
    """Значимые основы слов (без стоп-слов и слов короче 3 символов)."""
    out = []
    for w in _WORD_RE.findall(text.lower()):
        if w in STOPWORDS or len(w) < 3:
            continue
        out.append(stem(w))
    return out
