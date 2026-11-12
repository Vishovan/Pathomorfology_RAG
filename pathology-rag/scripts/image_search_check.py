#!/usr/bin/env python
"""Проверка поиска картинок: python scripts/image_search_check.py "крупозная пневмония" [-k 6] [--stain H&E]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathrag.image_search import search_images  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--source")
    ap.add_argument("--stain")
    ap.add_argument("--magnification")
    ap.add_argument("--organ")
    a = ap.parse_args()
    for i, r in enumerate(search_images(a.query, a.k, source_id=a.source, stain=a.stain,
                                        magnification=a.magnification, organ=a.organ), 1):
        where = r["page_url"] or f"{r['book_title']}, стр. {r['page']}"
        print(f"\n#{i} score={r['score']:.3f}  {r['diagnosis'] or '-'}  | {r['organ'] or '-'} | {r['stain'] or '-'} | {r['magnification'] or '-'}")
        print(f"    {(r['caption'] or '')[:200]}")
        print(f"    источник: {where}")
        print(f"    файл: {r['thumb_path'] or r['image_url']}")


if __name__ == "__main__":
    main()
