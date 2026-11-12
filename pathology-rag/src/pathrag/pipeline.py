"""Оркестрация: парсинг -> чанки/записи -> кэш -> индексация (текст и изображения)."""
import csv
import io
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import manifest as mf
from .chunking import chunk_blocks
from .config import ROOT, SETTINGS
from .images import make_record, make_thumb, rel

log = logging.getLogger("pathrag.pipeline")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _chunk(blocks, base_meta):
    return chunk_blocks(
        blocks, base_meta,
        max_chars=SETTINGS.chunk_max_chars,
        overlap=SETTINGS.chunk_overlap_chars,
        min_chars=SETTINGS.chunk_min_chars,
        split_level=SETTINGS.split_level,
    )


def _stats(chunks) -> str:
    if not chunks:
        return "0 чанков"
    avg = sum(c.metadata["char_len"] for c in chunks) / len(chunks)
    tables = sum(c.metadata["content_type"] == "table" for c in chunks)
    return f"{len(chunks)} чанков (ср. {avg:.0f} симв., таблиц: {tables})"


def _finish(manifest, doc_id, entry, chunks, dry_run):
    """Текст: кэш чанков + (если не dry-run) индексация + запись манифеста."""
    mf.save_chunks(doc_id, chunks)
    entry["chunks"] = len(chunks)
    if not dry_run and chunks:
        from .index import upsert_chunks

        upsert_chunks(chunks, doc_id)
        entry["indexed"] = True
        entry["ingested_at"] = _now()
    manifest[doc_id] = entry
    mf.save(manifest)


def _finish_images(manifest, doc_id, entry, records, dry_run, keep_thumbs=True):
    """Изображения: индексация (подписи + при наличии векторы изображений) + кэш записей + манифест."""
    entry["images"] = len(records)
    if not dry_run and records:
        from .index_images import upsert_images

        upsert_images(records, doc_id, keep_thumbs=keep_thumbs)
        entry["indexed"] = True
        entry["ingested_at"] = _now()
    mf.save_records(doc_id, records)
    manifest[doc_id] = entry
    mf.save(manifest, mf.IMAGES_MANIFEST)


def _img_stats(records) -> str:
    n = len(records)
    if not n:
        return "0 изображений"
    pct = lambda k: 100 * sum(bool(r.get(k)) for r in records) // n  # noqa: E731
    return f"{n} изображений (диагноз: {pct('diagnosis')}%, окраска: {pct('stain')}%, увеличение: {pct('magnification')}%, орган: {pct('organ')}%)"


# ---------------------------------------------------------------- PDF
def iter_pdfs(path: Path):
    return [path] if path.is_file() else sorted(path.rglob("*.pdf"))


def figures_to_records(source, doc_id, rel_s, pdf, figs) -> list[dict]:
    out_dir = SETTINGS.images_dir / source["id"] / doc_id
    shutil.rmtree(out_dir, ignore_errors=True)
    records, skipped = [], 0
    for f in figs:
        thumb = make_thumb(f.image, out_dir / f"{f.index:04d}.jpg")
        rec = make_record(
            source_id=source["id"], doc_key=rel_s, idx=f.index, caption=f.caption,
            default_lang=source.get("language", "ru"), section_path=f.section_path, nosology=f.nosology,
            extra={"kind": "figure", "book_title": pdf.stem.replace("_", " ").strip(), "page": f.page,
                   "file": rel_s, "thumb_path": rel(thumb), "image_url": None, "page_url": None},
        )
        if rec is None:
            thumb.unlink(missing_ok=True)
            skipped += 1
            continue
        records.append(rec)
    if skipped:
        print(f"  пропущено без подписи и контекста: {skipped}")
    return records


def ingest_pdfs(source: dict, path: Path | None, *, ocr: bool, dry_run: bool, force: bool,
                with_images: bool = False, images_only: bool = False) -> None:
    root = path or (ROOT / source["path"])
    pdfs = iter_pdfs(root)
    if not pdfs:
        print(f"PDF не найдены: {root}")
        return
    want_text, want_images = not images_only, with_images or images_only
    tm, im = mf.load(), mf.load(mf.IMAGES_MANIFEST)
    converter = None

    for pdf in pdfs:
        rel_s = str(pdf.relative_to(ROOT)) if pdf.is_relative_to(ROOT) else str(pdf)
        doc_id = mf.make_doc_id(source["id"], rel_s)
        sha = mf.file_sha256(pdf)
        te, ie = tm.get(doc_id, {}), im.get(doc_id, {})
        t_same, i_same = te.get("sha") == sha and not force, ie.get("sha") == sha and not force
        text_done = (not want_text) or (t_same and (te.get("indexed") or dry_run))
        img_done = (not want_images) or (i_same and (ie.get("indexed") or dry_run))
        if text_done and img_done:
            print(f"= {pdf.name}: без изменений, пропуск")
            continue

        chunks = mf.load_chunks(doc_id) if (not text_done and t_same) else None
        records = mf.load_records(doc_id) if (not img_done and i_same) else None
        doc = None
        if (not text_done and chunks is None) or (not img_done and records is None):
            if converter is None:
                from .parse_pdf import build_converter

                converter = build_converter(SETTINGS.ocr_langs, do_ocr=ocr, pictures=want_images)
            from .parse_pdf import convert_pdf

            print(f"> Парсинг {pdf.name} ...")
            doc = convert_pdf(pdf, converter)

        if not text_done:
            if chunks is None:
                from .parse_pdf import doc_to_blocks

                base_meta = {
                    "source_id": source["id"], "book_title": pdf.stem.replace("_", " ").strip(), "file": rel_s,
                    "language": source.get("language", "ru"), "content_type": "text", "sha": sha[:16],
                }
                chunks = _chunk(doc_to_blocks(doc), base_meta)
            print(f"  {pdf.name}: {_stats(chunks)}")
            _finish(tm, doc_id, {"sha": sha, "source_id": source["id"], "path": rel_s, "indexed": False}, chunks, dry_run)

        if not img_done:
            if records is None:
                from .parse_pdf import doc_to_figures

                records = figures_to_records(source, doc_id, rel_s, pdf, doc_to_figures(doc, SETTINGS.min_image_px))
            print(f"  {pdf.name}: {_img_stats(records)}")
            _finish_images(im, doc_id, {"sha": sha, "source_id": source["id"], "path": rel_s, "indexed": False}, records, dry_run)


# ---------------------------------------------------------------- Web
def _load_crawl_cfg(source: dict, need_images: bool = False):
    with open(SETTINGS.crawl_file, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    scfg = (cfg.get("sources") or {}).get(source["id"])
    if not scfg:
        raise SystemExit(f"Для '{source['id']}' нет записи в configs/crawl.yaml")
    if not (scfg.get("enabled") and scfg.get("tos_checked")):
        raise SystemExit(
            f"Краулинг '{source['id']}' выключен. Прочитайте условия использования сайта и "
            "поставьте enabled: true и tos_checked: true в configs/crawl.yaml."
        )
    if need_images and not (scfg.get("images") or {}).get("enabled"):
        raise SystemExit(f"Для '{source['id']}' не включён блок images (images.enabled: true) в configs/crawl.yaml.")
    return cfg, scfg


def ingest_web(source: dict, *, limit: int | None, dry_run: bool, force: bool) -> None:
    from .web import Crawler, html_to_blocks

    cfg, scfg = _load_crawl_cfg(source)
    manifest = mf.load()
    crawler = Crawler(scfg, cfg["user_agent"], float(cfg.get("delay_seconds", 2.0)))
    for url, html in crawler.crawl(limit):
        title, blocks = html_to_blocks(html)
        sha = mf.text_sha256(html)
        doc_id = mf.make_doc_id(source["id"], url)
        entry = manifest.get(doc_id, {})
        if entry.get("sha") == sha and not force and (entry.get("indexed") or dry_run):
            print(f"= {url}: без изменений")
            continue
        base_meta = {
            "source_id": source["id"], "book_title": title or url, "url": url,
            "language": scfg.get("language", source.get("language", "en")),
            "content_type": "web", "sha": sha[:16],
        }
        chunks = _chunk(blocks, base_meta)
        print(f"  {url}: {_stats(chunks)}")
        _finish(manifest, doc_id, {"sha": sha, "source_id": source["id"], "path": url, "indexed": False}, chunks, dry_run)


def ingest_web_images(source: dict, *, limit: int | None, dry_run: bool, force: bool) -> None:
    from PIL import Image

    from .web import Crawler
    from .web_images import extract_images

    cfg, scfg = _load_crawl_cfg(source, need_images=True)
    icfg = scfg["images"]
    keep = bool(icfg.get("save_thumbnails", False))
    per_page = int(icfg.get("max_per_page", 20))
    lang = scfg.get("language", source.get("language", "en"))
    im = mf.load(mf.IMAGES_MANIFEST)
    crawler = Crawler(scfg, cfg["user_agent"], float(cfg.get("delay_seconds", 2.0)))

    for url, html in crawler.crawl(limit):
        sha = mf.text_sha256(html)
        doc_id = mf.make_doc_id(source["id"], url)
        entry = im.get(doc_id, {})
        if entry.get("sha") == sha and not force and (entry.get("indexed") or dry_run):
            print(f"= {url}: без изменений")
            continue
        _, items = extract_images(html, url, SETTINGS.min_image_px)
        out_dir = SETTINGS.images_dir / source["id"] / doc_id
        shutil.rmtree(out_dir, ignore_errors=True)
        records = []
        for i, it in enumerate(items[:per_page], 1):
            data = crawler.fetch_bytes(it["image_url"])
            if not data:
                continue
            try:
                img = Image.open(io.BytesIO(data))
                img.load()
            except Exception:
                continue
            if min(img.size) < SETTINGS.min_image_px:
                continue
            thumb = make_thumb(img, out_dir / f"{i:03d}.jpg")
            rec = make_record(
                source_id=source["id"], doc_key=url, idx=i, caption=it["caption"], default_lang=lang,
                section_path=it["heading"] or it["page_title"] or None,
                extra={"kind": "photo", "book_title": it["page_title"] or url, "page": None,
                       "image_url": it["image_url"], "page_url": url, "thumb_path": rel(thumb)},
            )
            if rec is None:
                thumb.unlink(missing_ok=True)
                continue
            if it["caption"]:
                rec["caption_source"] = it["caption_how"]
            records.append(rec)
        print(f"  {url}: {_img_stats(records)}")
        _finish_images(im, doc_id, {"sha": sha, "source_id": source["id"], "path": url, "indexed": False},
                       records, dry_run, keep_thumbs=keep)


# ---------------------------------------------------------------- WSI-ссылки из CSV
def ingest_wsi_links(csv_path: Path, *, dry_run: bool, fetch_previews: bool) -> None:
    """CSV-колонки: source_id, page_url, preview_url, caption, diagnosis, stain, magnification, organ, language.
    Обязательны source_id и page_url. Слайды не скачиваются: хранятся ссылка и (по желанию) превью."""
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("source_id") and r.get("page_url"):
            by_source.setdefault(r["source_id"], []).append(r)
    im = mf.load(mf.IMAGES_MANIFEST)
    client = None
    for sid, items in by_source.items():
        doc_id = mf.make_doc_id(sid, "wsi_links")
        out_dir = SETTINGS.images_dir / sid / doc_id
        records = []
        for i, r in enumerate(items, 1):
            thumb_rel = None
            if fetch_previews and r.get("preview_url"):
                try:
                    import httpx
                    from PIL import Image

                    client = client or httpx.Client(timeout=30, follow_redirects=True)
                    resp = client.get(r["preview_url"])
                    resp.raise_for_status()
                    thumb_rel = rel(make_thumb(Image.open(io.BytesIO(resp.content)), out_dir / f"{i:04d}.jpg"))
                except Exception as e:
                    print(f"  превью недоступно ({r['preview_url']}): {e}")
            rec = make_record(
                source_id=sid, doc_key=r["page_url"], idx=i, caption=r.get("caption") or "",
                default_lang=r.get("language") or "en", nosology=r.get("diagnosis") or None,
                overrides={k: r.get(k) for k in ("diagnosis", "stain", "magnification", "organ")},
                extra={"kind": "wsi_link", "book_title": sid, "page": None, "image_url": r.get("preview_url") or None,
                       "page_url": r["page_url"], "thumb_path": thumb_rel},
            )
            if rec:
                records.append(rec)
        print(f"  {sid}: {_img_stats(records)}")
        sha = mf.text_sha256(csv_path.read_text(encoding="utf-8-sig"))
        _finish_images(im, doc_id, {"sha": sha, "source_id": sid, "path": str(csv_path), "indexed": False},
                       records, dry_run, keep_thumbs=True)
