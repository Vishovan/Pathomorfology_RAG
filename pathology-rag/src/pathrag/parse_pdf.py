"""PDF -> блоки текста и рисунки через Docling (с OCR для сканов)."""
import re
from dataclasses import dataclass
from pathlib import Path

from .blocks import Block

CHAPTER_RE = re.compile(r"^\s*(глава|часть|раздел|chapter|part)\s+[\dIVXLCivxlc]+", re.I)


@dataclass
class Figure:
    index: int
    page: int | None
    caption: str
    section_path: str | None
    nosology: str | None
    image: object  # PIL.Image


def build_converter(ocr_langs=("ru", "en"), do_ocr: bool = True, pictures: bool = False):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = do_ocr
    opts.do_table_structure = True
    if do_ocr:
        opts.ocr_options = EasyOcrOptions(lang=list(ocr_langs))
    if pictures:  # нужно, чтобы Docling отдавал картинки рисунков
        opts.generate_picture_images = True
        opts.images_scale = 2.0
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def convert_pdf(path: Path, converter):
    return converter.convert(str(path)).document


def _heading_level(label, item, text) -> int:
    from docling_core.types.doc import DocItemLabel as L

    level = 1 if label == L.TITLE else getattr(item, "level", 1)
    return 1 if CHAPTER_RE.match(text) else max(level, 2)


def normalize_headings(blocks: list[Block]) -> list[Block]:
    """Docling часто отдаёт плоские уровни заголовков: «Глава N» считаем уровнем 1, остальное не выше 2."""
    out = []
    for b in blocks:
        if b.kind == "heading":
            level = 1 if CHAPTER_RE.match(b.text) else max(b.level, 2)
            out.append(Block("heading", b.text, level, b.page))
        else:
            out.append(b)
    return out


def doc_to_blocks(doc) -> list[Block]:
    from docling_core.types.doc import DocItemLabel as L

    skip = {L.PAGE_HEADER, L.PAGE_FOOTER, L.PICTURE}
    blocks: list[Block] = []
    for item, _ in doc.iterate_items():
        label = getattr(item, "label", None)
        if label in skip:
            continue
        prov = getattr(item, "prov", None)
        page = prov[0].page_no if prov else None
        if label == L.TABLE:
            try:
                md = item.export_to_markdown(doc=doc)
            except TypeError:
                md = item.export_to_markdown()
            if md.strip():
                blocks.append(Block("table", md, page=page))
        elif label in (L.TITLE, L.SECTION_HEADER):
            text = (getattr(item, "text", "") or "").strip()
            if text:
                level = 1 if label == L.TITLE else getattr(item, "level", 1)
                blocks.append(Block("heading", text, level, page))
        else:
            text = (getattr(item, "text", "") or "").strip()
            if text:
                blocks.append(Block("text", text, page=page))
    return normalize_headings(blocks)


def pdf_to_blocks(path: Path, converter) -> list[Block]:
    return doc_to_blocks(convert_pdf(path, converter))


def doc_to_figures(doc, min_px: int = 120) -> list[Figure]:
    """Рисунки документа с подписью, страницей и путём заголовков. Требует converter(pictures=True)."""
    from docling_core.types.doc import DocItemLabel as L
    from docling_core.types.doc import PictureItem

    figs: list[Figure] = []
    heads: list[tuple[int, str]] = []
    for item, _ in doc.iterate_items():
        label = getattr(item, "label", None)
        prov = getattr(item, "prov", None)
        page = prov[0].page_no if prov else None
        if label in (L.TITLE, L.SECTION_HEADER):
            text = (getattr(item, "text", "") or "").strip()
            if text:
                level = _heading_level(label, item, text)
                while heads and heads[-1][0] >= level:
                    heads.pop()
                heads.append((level, text))
            continue
        if label == L.CAPTION:  # подпись, не привязанная к рисунку самим Docling
            text = (getattr(item, "text", "") or "").strip()
            if figs and text and not figs[-1].caption and figs[-1].page == page:
                figs[-1].caption = text
            continue
        if not isinstance(item, PictureItem):
            continue
        try:
            caption = (item.caption_text(doc) or "").strip()
        except Exception:
            caption = ""
        try:
            img = item.get_image(doc)
        except Exception:
            img = None
        if img is None or min(img.size) < min_px:
            continue
        figs.append(Figure(
            index=len(figs) + 1, page=page, caption=caption,
            section_path=" > ".join(t for _, t in heads) or None,
            nosology=heads[-1][1] if heads else None, image=img,
        ))
    return figs
