#!/usr/bin/env python
"""Ингест источников в Qdrant.

Текст:
  python scripts/ingest.py pdf [--source ID] [--path FILE_OR_DIR] [--no-ocr] [--with-images] [--dry-run] [--force]
  python scripts/ingest.py web --source ID [--limit N] [--dry-run] [--force]
Изображения:
  python scripts/ingest.py images pdf [--source ID] [--path FILE_OR_DIR] [--no-ocr] [--dry-run] [--force]
  python scripts/ingest.py images web --source ID [--limit N] [--dry-run] [--force]
  python scripts/ingest.py images wsi-links --csv data/raw/wsi_links.csv [--fetch-previews] [--dry-run]
Служебное:
  python scripts/ingest.py status
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathrag import manifest as mf  # noqa: E402
from pathrag.pipeline import ingest_pdfs, ingest_web, ingest_web_images, ingest_wsi_links  # noqa: E402
from pathrag.sources import get_source  # noqa: E402

DEFAULT_PDF_SOURCE = "local_textbooks_pathology"


def _common(sp, with_force=True):
    sp.add_argument("--dry-run", action="store_true", help="разбор без эмбеддингов и записи в Qdrant")
    if with_force:
        sp.add_argument("--force", action="store_true", help="переобработать, даже если не менялось")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pdf = sub.add_parser("pdf", help="локальные PDF-учебники (текст)")
    pdf.add_argument("--source", default=DEFAULT_PDF_SOURCE)
    pdf.add_argument("--path", type=Path)
    pdf.add_argument("--no-ocr", action="store_true")
    pdf.add_argument("--with-images", action="store_true", help="заодно извлечь рисунки (один проход Docling)")
    _common(pdf)

    web = sub.add_parser("web", help="веб-источники (текст), см. configs/crawl.yaml")
    web.add_argument("--source", required=True)
    web.add_argument("--limit", type=int)
    _common(web)

    img = sub.add_parser("images", help="изображения")
    isub = img.add_subparsers(dest="icmd", required=True)
    ipdf = isub.add_parser("pdf", help="рисунки из PDF с подписями")
    ipdf.add_argument("--source", default=DEFAULT_PDF_SOURCE)
    ipdf.add_argument("--path", type=Path)
    ipdf.add_argument("--no-ocr", action="store_true")
    _common(ipdf)
    iweb = isub.add_parser("web", help="изображения с веб-атласов")
    iweb.add_argument("--source", required=True)
    iweb.add_argument("--limit", type=int)
    _common(iweb)
    iwsi = isub.add_parser("wsi-links", help="ссылки на WSI из CSV")
    iwsi.add_argument("--csv", type=Path, required=True)
    iwsi.add_argument("--fetch-previews", action="store_true")
    _common(iwsi, with_force=False)

    sub.add_parser("status", help="что обработано и сколько точек в Qdrant")

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "pdf":
        ingest_pdfs(get_source(args.source), args.path, ocr=not args.no_ocr, dry_run=args.dry_run,
                    force=args.force, with_images=args.with_images)
    elif args.cmd == "web":
        ingest_web(get_source(args.source), limit=args.limit, dry_run=args.dry_run, force=args.force)
    elif args.cmd == "images":
        if args.icmd == "pdf":
            ingest_pdfs(get_source(args.source), args.path, ocr=not args.no_ocr, dry_run=args.dry_run,
                        force=args.force, images_only=True)
        elif args.icmd == "web":
            ingest_web_images(get_source(args.source), limit=args.limit, dry_run=args.dry_run, force=args.force)
        else:
            ingest_wsi_links(args.csv, dry_run=args.dry_run, fetch_previews=args.fetch_previews)
    else:
        from pathrag.index import collection_count
        from pathrag.index_images import count_images

        for title, name, key in (("Текст", mf.TEXT_MANIFEST, "chunks"), ("Изображения", mf.IMAGES_MANIFEST, "images")):
            m = mf.load(name)
            print(f"{title}: документов {len(m)}, проиндексировано {sum(1 for e in m.values() if e.get('indexed'))}")
            for e in m.values():
                print(f"  [{'x' if e.get('indexed') else ' '}] {e['source_id']:28} {e.get(key, 0):5}  {e['path']}")
        print(f"Точек в Qdrant: текст {collection_count()}, изображения {count_images()}")


if __name__ == "__main__":
    main()
