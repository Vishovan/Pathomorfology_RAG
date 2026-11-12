import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathrag.blocks import Block  # noqa: E402
from pathrag.chunking import chunk_blocks, detect_language  # noqa: E402
from pathrag.parse_pdf import normalize_headings  # noqa: E402

META = {"source_id": "t", "book_title": "Book", "language": "ru_en", "content_type": "text"}
LONG = "Воспаление характеризуется сосудистой реакцией. " * 12  # ~570 симв.


def run(blocks, **kw):
    return chunk_blocks(blocks, META, max_chars=800, overlap=100, min_chars=200, **kw)


def test_sections_and_metadata():
    blocks = normalize_headings([
        Block("heading", "Глава 1. Воспаление", 1, 10),
        Block("heading", "Крупозная пневмония", 3, 11),
        Block("text", LONG, page=11),
        Block("text", LONG, page=12),
        Block("heading", "Фибринозное воспаление", 3, 15),
        Block("text", LONG, page=15),
    ])
    chunks = run(blocks)
    assert len(chunks) >= 3
    first = chunks[0].metadata
    assert first["chapter"].startswith("Глава 1")
    assert first["nosology"] == "Крупозная пневмония"
    assert first["page_start"] == 11
    assert any(c.metadata["nosology"] == "Фибринозное воспаление" for c in chunks)
    assert all(c.metadata["char_len"] <= 800 for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_tiny_section_is_carried_not_lost():
    blocks = [
        Block("heading", "Глава 2", 1, 1),
        Block("heading", "Короткий раздел", 2, 1),
        Block("text", "Одна строка.", page=1),
        Block("heading", "Большой раздел", 2, 2),
        Block("text", LONG, page=2),
    ]
    chunks = run(blocks)
    joined = "\n".join(c.text for c in chunks)
    assert "Одна строка." in joined and "Короткий раздел" in joined


def test_table_is_separate_chunk():
    blocks = [
        Block("heading", "Глава 3", 1, 1),
        Block("text", LONG, page=1),
        Block("table", "| a | b |\n| 1 | 2 |", page=2),
    ]
    chunks = run(blocks)
    assert any(c.metadata["content_type"] == "table" for c in chunks)


def test_long_paragraph_is_split():
    chunks = run([Block("heading", "Глава 4", 1, 1), Block("text", "Слово. " * 600, page=1)])
    assert len(chunks) > 1
    assert all(len(c.text) <= 800 + 120 for c in chunks)  # + префикс overlap


def test_deep_headings_inlined():
    blocks = [Block("heading", "Глава 5", 1, 1), Block("heading", "Подпункт", 5, 1), Block("text", LONG, page=1)]
    chunks = run(blocks)
    assert "Подпункт" in chunks[0].text


def test_language():
    assert detect_language("Inflammation is a response", "ru_en") == "en"
    assert detect_language("Воспаление это реакция", "ru_en") == "ru"
    assert detect_language("anything", "ru") == "ru"
