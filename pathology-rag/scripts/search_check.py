#!/usr/bin/env python
"""Быстрая проверка поиска: python scripts/search_check.py "морфология крупозной пневмонии" [-k 5]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llama_index.core import VectorStoreIndex  # noqa: E402

from pathrag.index import get_embed_model, get_vector_store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args()

    index = VectorStoreIndex.from_vector_store(get_vector_store(), embed_model=get_embed_model())
    for i, hit in enumerate(index.as_retriever(similarity_top_k=a.k).retrieve(a.query), 1):
        m = hit.node.metadata
        pages = f"{m.get('page_start')}-{m.get('page_end')}" if m.get("page_start") else "-"
        print(f"\n#{i}  score={hit.score:.3f}  {m.get('book_title')}  стр. {pages}")
        print(f"    раздел: {m.get('section_path')}")
        print(f"    {hit.node.text[:300].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()
