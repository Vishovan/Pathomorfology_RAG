"""Проверка открытого корпуса: загрузка, лицензии, попадание в выдачу ретривера."""
import json
from pathlib import Path

import pytest

from pathrag.retrieval.hybrid_retriever import KNOWLEDGE_DIR, _load_knowledge_corpus, retrieve

CORPUS = KNOWLEDGE_DIR / "corpus"
ALLOWED_LICENSES = {"CC BY", "CC BY-SA 4.0"}

pytestmark = pytest.mark.skipif(not any(CORPUS.glob("*.jsonl")), reason="корпус не собран")


def test_corpus_rows_have_license_and_url():
    n = 0
    for f in CORPUS.glob("*.jsonl"):
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            meta = row["metadata"]
            assert meta["license"] in ALLOWED_LICENSES
            assert meta["url"].startswith("https://")
            assert len(row["text"]) > 40
            n += 1
    assert n >= 200, f"слишком маленький корпус: {n}"


def test_corpus_is_loaded_by_retriever():
    docs = _load_knowledge_corpus()
    assert sum(1 for d in docs if d.get("license")) >= 200


@pytest.mark.parametrize("query, expected", [
    ("Что такое казеозный некроз при туберкулёзе", "некроз"),
    ("гранулёма морфология", "гранул"),
    ("amyloidosis Congo red", "amyloid"),
])
def test_retrieval_hits_open_corpus(query, expected):
    hits = retrieve(query, top_k=6)
    assert any(expected in (h.get("text", "") + h.get("title", "")).lower() for h in hits)
