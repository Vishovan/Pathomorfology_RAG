"""Поиск изображений с подписями на HTML-странице."""
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

_NOISE = ["script", "style", "nav", "footer", "aside", "form", "noscript", "iframe"]
_JUNK = re.compile(r"(logo|icon|sprite|banner|button|avatar|spacer|pixel|arrow|bullet|favicon|/ads?/|counter)", re.I)
_FILELIKE = re.compile(r"\.(jpe?g|png|gif|webp|tiff?)\b", re.I)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _dim(v) -> int | None:
    digits = re.sub(r"\D", "", str(v or ""))
    return int(digits) if digits else None


def _caption_for(img, limit: int = 400) -> tuple[str, str]:
    fig = img.find_parent("figure")
    if fig:
        fc = fig.find("figcaption")
        if fc and _clean(fc.get_text(" ")):
            return _clean(fc.get_text(" "))[:limit], "figcaption"
    for attr in ("alt", "title"):
        v = _clean(img.get(attr, ""))
        if len(v) >= 8 and not _FILELIKE.search(v) and v.lower() not in ("image", "picture", "photo"):
            return v[:limit], attr
    box = img.find_parent(["td", "li", "div", "p"])
    if box is not None:
        t = _clean(box.get_text(" "))
        if 10 <= len(t) <= limit:
            return t, "container"
        sib = box.find_next_sibling()
        if sib is not None:
            t = _clean(sib.get_text(" "))
            if 10 <= len(t) <= limit:
                return t, "next_sibling"
    return "", "none"


def extract_images(html: str, page_url: str, min_px: int = 120) -> tuple[str, list[dict]]:
    soup = BeautifulSoup(html, "lxml")
    title = _clean(soup.title.get_text()) if soup.title else ""
    for t in soup(_NOISE):
        t.decompose()
    out, seen = [], set()
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or ""
        if not src or src.startswith("data:") or _JUNK.search(src):
            continue
        w, h = _dim(img.get("width")), _dim(img.get("height"))
        if (w and w < min_px) or (h and h < min_px):
            continue
        url = urljoin(page_url, src)
        if url in seen:
            continue
        seen.add(url)
        caption, how = _caption_for(img)
        head = img.find_previous(["h1", "h2", "h3", "h4"])
        out.append({
            "image_url": url,
            "caption": caption,
            "caption_how": how,
            "heading": _clean(head.get_text(" ")) if head else "",
            "page_title": title,
            "page_url": page_url,
        })
    return title, out
