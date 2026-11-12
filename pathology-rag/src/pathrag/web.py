"""Краулер и разбор HTML в Block. Уважает robots.txt и работает только с разрешёнными в crawl.yaml источниками."""
import logging
import re
import time
from collections import deque
from urllib import robotparser
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from .blocks import Block

log = logging.getLogger("pathrag.web")
_NOISE = ["script", "style", "nav", "footer", "aside", "form", "noscript", "iframe"]


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def html_to_blocks(html: str) -> tuple[str, list[Block]]:
    soup = BeautifulSoup(html, "lxml")
    title = _clean(soup.title.get_text()) if soup.title else ""
    for t in soup(_NOISE):
        t.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    blocks: list[Block] = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "table", "pre"]):
        if el.name != "table" and el.find_parent("table"):
            continue
        if el.name == "p" and el.find_parent("li"):
            continue
        if el.name == "li" and el.find("li"):
            continue
        if el.name == "table":
            rows = []
            for tr in el.find_all("tr"):
                cells = [_clean(c.get_text(" ")) for c in tr.find_all(["td", "th"])]
                if any(cells):
                    rows.append(" | ".join(cells))
            text = "\n".join(rows)
            kind = "table"
        else:
            text = _clean(el.get_text(" "))
            kind = "text"
        if len(text) < 3:
            continue
        if el.name[0] == "h" and len(el.name) == 2:
            blocks.append(Block("heading", text, int(el.name[1])))
        elif el.name == "li":
            blocks.append(Block("text", f"- {text}"))
        else:
            blocks.append(Block(kind, text))
    return title, blocks


class Crawler:
    def __init__(self, cfg: dict, user_agent: str, delay: float):
        self.cfg, self.delay = cfg, delay
        self.client = httpx.Client(headers={"User-Agent": user_agent}, timeout=30, follow_redirects=True)
        self.allow = [re.compile(p) for p in cfg.get("allow", [])]
        self.deny = [re.compile(p) for p in cfg.get("deny", [])]
        self.hosts = {urlparse(u).netloc for u in cfg["seeds"]}
        self._robots: dict[str, robotparser.RobotFileParser] = {}
        self.ua = user_agent

    def _can_fetch(self, url: str) -> bool:
        p = urlparse(url)
        origin = f"{p.scheme}://{p.netloc}"
        rp = self._robots.get(origin)
        if rp is None:
            rp = robotparser.RobotFileParser()
            try:
                r = self.client.get(f"{origin}/robots.txt")
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except httpx.HTTPError:
                rp.parse([])
            self._robots[origin] = rp
        return rp.can_fetch(self.ua, url)

    def _ok(self, url: str) -> bool:
        if urlparse(url).netloc not in self.hosts:
            return False
        if self.allow and not any(r.search(url) for r in self.allow):
            return False
        return not any(r.search(url) for r in self.deny)

    def crawl(self, limit: int | None = None):
        max_pages = min(limit or 10**9, self.cfg.get("max_pages", 50))
        queue = deque(self.cfg["seeds"])
        seen, n = set(), 0
        while queue and n < max_pages:
            url = urldefrag(queue.popleft())[0]
            if url in seen or not self._ok(url) or not self._can_fetch(url):
                continue
            seen.add(url)
            try:
                r = self.client.get(url)
            except httpx.HTTPError as e:
                log.warning("skip %s: %s", url, e)
                continue
            time.sleep(self.delay)
            if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
                continue
            n += 1
            yield url, r.text
            for a in BeautifulSoup(r.text, "lxml").find_all("a", href=True):
                queue.append(urljoin(url, a["href"]))

    def fetch_bytes(self, url: str, max_bytes: int = 15_000_000) -> bytes | None:
        """Скачивает изображение с учётом robots.txt и паузы между запросами."""
        if not self._can_fetch(url):
            return None
        try:
            r = self.client.get(url)
        except httpx.HTTPError as e:
            log.warning("skip image %s: %s", url, e)
            return None
        time.sleep(self.delay)
        if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image/"):
            return None
        return r.content if len(r.content) <= max_bytes else None
