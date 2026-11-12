"""Чанкинг по разделам с сохранением пути заголовков и страниц.

Вход: список Block (заголовки, абзацы, таблицы) в порядке документа.
Выход: список Chunk с метаданными (глава, раздел, нозология, страницы).
"""
import re

from .blocks import Block, Chunk

_SENT = re.compile(r"(?<=[.!?…])\s+")


def detect_language(text: str, default: str) -> str:
    if default != "ru_en":
        return default
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    cyr = sum("а" <= c.lower() <= "я" or c.lower() == "ё" for c in letters)
    return "ru" if cyr / len(letters) > 0.3 else "en"


def _split_long(text: str, max_chars: int, by_line: bool = False) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces = text.split("\n") if by_line else _SENT.split(text)
    parts, cur = [], ""
    for p in pieces:
        while len(p) > max_chars:  # жёсткая нарезка
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(p[:max_chars])
            p = p[max_chars:]
        sep = "\n" if by_line else " "
        if cur and len(cur) + len(sep) + len(p) > max_chars:
            parts.append(cur)
            cur = p
        else:
            cur = f"{cur}{sep}{p}" if cur else p
    if cur:
        parts.append(cur)
    return parts


def _pack(paras: list[tuple[str, int | None, str]], max_chars: int) -> list[tuple[str, list]]:
    """Упаковка абзацев в группы. Возвращает [(kind, [(text, page), ...])]."""
    groups: list[tuple[str, list]] = []
    cur: list = []
    cur_len = 0
    for text, page, kind in paras:
        if kind == "table":
            if cur:
                groups.append(("text", cur))
                cur, cur_len = [], 0
            for part in _split_long(text, max_chars, by_line=True):
                groups.append(("table", [(part, page)]))
            continue
        for part in _split_long(text, max_chars):
            if cur and cur_len + len(part) + 2 > max_chars:
                groups.append(("text", cur))
                cur, cur_len = [], 0
            cur.append((part, page))
            cur_len += len(part) + 2
    if cur:
        groups.append(("text", cur))
    return groups


def _tail(text: str, n: int) -> str:
    if n <= 0 or len(text) <= n:
        return ""
    t = text[-n:]
    i = t.find(" ")
    return t[i + 1:] if 0 <= i < len(t) - 1 else t


def chunk_blocks(
    blocks: list[Block],
    base_meta: dict,
    *,
    max_chars: int = 1800,
    overlap: int = 200,
    min_chars: int = 200,
    split_level: int = 3,
) -> list[Chunk]:
    default_lang = base_meta.get("language", "en")
    stack: list[tuple[int, str]] = []
    paras: list[tuple[str, int | None, str]] = []
    carry: list[tuple[str, int | None, str]] = []
    chunks: list[Chunk] = []

    def emit(items, path):
        groups = _pack(items, max_chars)
        prev_text = ""
        for kind, group in groups:
            body = "\n\n".join(t for t, _ in group)
            text = body
            if kind == "text" and prev_text:
                tail = _tail(prev_text, overlap)
                if tail:
                    text = f"… {tail}\n\n{body}"
            prev_text = body if kind == "text" else ""
            pages = [p for _, p in group if p is not None]
            meta = dict(base_meta)
            meta.update(
                chapter=path[0][1] if path and path[0][0] == 1 else None,
                section_path=" > ".join(t for _, t in path) or None,
                nosology=path[-1][1] if path else None,  # эвристика: самый глубокий заголовок
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                content_type="table" if kind == "table" else base_meta.get("content_type", "text"),
                language=detect_language(body, default_lang),
                char_len=len(body),
            )
            chunks.append(Chunk(text=text, metadata=meta))

    def flush(final: bool = False):
        nonlocal paras, carry
        if not paras and not (final and carry):
            return
        total = sum(len(t) for t, _, _ in paras)
        if total < min_chars and not final:
            # слишком короткий раздел: переносим в следующий вместе с заголовком
            if stack:
                carry.append((stack[-1][1], paras[0][1] if paras else None, "text"))
            carry.extend(paras)
            paras = []
            return
        emit(carry + paras, list(stack))
        carry, paras = [], []

    for b in blocks:
        text = b.text.strip()
        if not text:
            continue
        if b.kind == "heading":
            if b.level <= split_level:
                flush()
                while stack and stack[-1][0] >= b.level:
                    stack.pop()
                stack.append((b.level, text))
            else:
                paras.append((text, b.page, "text"))
        else:
            paras.append((text, b.page, b.kind))
    flush(final=True)

    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
        c.metadata["chunk_total"] = len(chunks)
    return chunks
