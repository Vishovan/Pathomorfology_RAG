#!/usr/bin/env python
"""Сбор ОТКРЫТО лицензированного корпуса по патоморфологии для RAG.

Источники (только легально доступные для скачивания и переиспользования):
  * Wikipedia (ru/en), CC BY-SA 4.0 — обзорные статьи по общей и частной патологии;
  * Europe PMC, Open Access subset с лицензией CC BY — обзоры (в т.ч. по WHO Classification).

Коммерческие учебники (Robbins, WHO Blue Books и т.п.) НЕ скачиваются: они защищены
авторским правом. Ваши собственные лицензионные PDF индексируйте через scripts/ingest.py pdf.

Результат: src/pathrag/knowledge/corpus/*.jsonl (чанки с метаданными и лицензией).
Запуск: python scripts/fetch_open_corpus.py [--max-epmc 40]
"""
import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

OUT = Path(__file__).resolve().parents[1] / "src" / "pathrag" / "knowledge" / "corpus"
UA = {"User-Agent": "pathrag-internal-bot/0.2 (internal pathology RAG; non-scraping API use)"}
CHUNK = 1400

WIKI_RU = [
    "Патологическая анатомия", "Альтерация", "Дистрофия (медицина)", "Некроз", "Апоптоз",
    "Воспаление", "Гранулёма", "Регенерация (биология)", "Атрофия", "Гипертрофия", "Гиперплазия",
    "Метаплазия", "Дисплазия", "Опухоль", "Рак (заболевание)", "Карцинома", "Саркома",
    "Аденокарцинома", "Плоскоклеточный рак", "Метастаз", "Амилоидоз", "Тромбоз", "Эмболия",
    "Инфаркт (медицина)", "Атеросклероз", "Отёк", "Полнокровие", "Кровоизлияние", "Гангрена",
    "Стеатоз печени", "Цирроз печени", "Гепатит", "Пневмония", "Крупозная пневмония", "Туберкулёз",
    "Гломерулонефрит", "Рак молочной железы", "Рак лёгкого", "Колоректальный рак", "Рак желудка",
    "Лимфома", "Лейкоз", "Меланома", "Глиобластома", "Иммуногистохимия",
    "Гистологическое исследование", "Окраска гематоксилином и эозином", "Биопсия", "Аутопсия",
    "Шкала Глисона", "Ki-67", "HER2",
]
WIKI_EN = [
    "Anatomical pathology", "Histopathology", "Cell injury", "Necrosis", "Apoptosis",
    "Inflammation", "Granuloma", "Neoplasm", "Carcinoma", "Sarcoma", "Adenocarcinoma",
    "Squamous-cell carcinoma", "Metastasis", "Amyloidosis", "Thrombosis", "Infarction",
    "Atherosclerosis", "Dysplasia", "Metaplasia", "Immunohistochemistry", "Frozen section procedure",
    "Breast cancer", "Invasive carcinoma of no special type", "Ductal carcinoma in situ",
    "Lobular carcinoma in situ", "HER2", "Ki-67 (protein)", "Estrogen receptor", "Gleason grading system",
    "Nottingham histologic grade", "Glioblastoma", "Astrocytoma", "Oligodendroglioma",
    "Hepatocellular carcinoma", "Colorectal cancer", "Lymphoma", "Hodgkin lymphoma",
    "Diffuse large B-cell lymphoma", "Melanoma", "Lung cancer", "Cervical intraepithelial neoplasia",
    "Hematoxylin and eosin stain", "WHO Classification of Tumours", "TNM staging system",
]
EPMC_QUERIES = [
    '(TITLE:"WHO classification") AND (TITLE:tumours OR TITLE:tumors OR TITLE:neoplasms)',
    'TITLE:"histopathology" AND TITLE:review',
    'TITLE:"immunohistochemistry" AND (TITLE:diagnostic OR TITLE:pathology) AND TITLE:review',
    'TITLE:"breast cancer" AND (TITLE:HER2 OR TITLE:"Ki-67") AND TITLE:pathology',
    'TITLE:"digital pathology" AND TITLE:review',
]


def chunk_text(text: str, size: int = CHUNK) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n{1,}", text) if len(p.strip()) > 40]
    out, buf = [], ""
    for p in paras:
        if len(buf) + len(p) > size and buf:
            out.append(buf.strip())
            buf = ""
        buf += p + "\n"
    if buf.strip():
        out.append(buf.strip())
    return out


def write(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {name}.jsonl: {len(rows)} чанков")


def fetch_wiki(client: httpx.Client, lang: str, titles: list[str]) -> list[dict]:
    rows = []
    for t in titles:
        r = None
        for attempt in range(4):
            r = client.get(f"https://{lang}.wikipedia.org/w/api.php", params={
                "action": "query", "format": "json", "prop": "extracts|info", "inprop": "url",
                "explaintext": 1, "redirects": 1, "titles": t})
            if r.status_code == 200:
                break
            time.sleep(2 * (attempt + 1))
        if r is None or r.status_code != 200:
            print(f"  ! wiki {lang} {t}: HTTP {r.status_code if r else '-'}")
            continue
        for page in r.json().get("query", {}).get("pages", {}).values():
            text = page.get("extract") or ""
            text = re.split(r"\n== (Примечания|Литература|Ссылки|See also|References|External links|Further reading) ==", text)[0]
            if len(text) < 300:
                print(f"  ! wiki {lang} {t}: пусто")
                continue
            for i, c in enumerate(chunk_text(text)):
                rows.append({
                    "text": c,
                    "metadata": {
                        "book_title": f"Wikipedia ({lang}): {page['title']}",
                        "section_path": page["title"], "chunk_index": i, "page_start": "-",
                        "url": page.get("fullurl"), "license": "CC BY-SA 4.0", "language": lang,
                        "source_type": "encyclopedia",
                    },
                })
        time.sleep(1.0)
    return rows


def _xml_text(el) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def fetch_epmc(client: httpx.Client, max_docs: int) -> list[dict]:
    rows, seen = [], set()
    per_q = max(1, max_docs // len(EPMC_QUERIES))
    for q in EPMC_QUERIES:
        r = client.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params={
            "query": f'{q} AND OPEN_ACCESS:y AND LICENSE:"cc by" AND PUB_YEAR:[2015 TO 2026]',
            "format": "json", "pageSize": per_q, "resultType": "core", "sort": "CITED desc"})
        for hit in r.json().get("resultList", {}).get("result", []):
            pmcid = hit.get("pmcid")
            lic = (hit.get("license") or "").lower()
            if not pmcid or pmcid in seen or lic != "cc by":
                continue
            seen.add(pmcid)
            x = client.get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML")
            if x.status_code != 200:
                continue
            try:
                root = ET.fromstring(x.content)
            except ET.ParseError:
                continue
            body = root.find(".//body")
            if body is None:
                continue
            parts = []
            for sec in body.iter("sec"):
                title = sec.find("title")
                if title is not None:
                    parts.append("\n" + _xml_text(title))
                parts += [_xml_text(p) for p in sec.findall("p")]
            text = "\n".join(parts)
            abstract = root.find(".//abstract")
            if abstract is not None:
                text = _xml_text(abstract) + "\n" + text
            for i, c in enumerate(chunk_text(text)):
                rows.append({
                    "text": c,
                    "metadata": {
                        "book_title": f"{hit.get('title', '').rstrip('.')} ({hit.get('journalTitle') or hit.get('bookOrReportDetails', {}).get('publisher', '')}, {hit.get('pubYear')})",
                        "section_path": hit.get("title"), "chunk_index": i, "page_start": "-",
                        "url": f"https://europepmc.org/article/PMC/{pmcid}", "doi": hit.get("doi"),
                        "license": "CC BY", "language": "en", "source_type": "journal_review",
                    },
                })
            print(f"  + {pmcid}: {hit.get('title', '')[:90]}")
            time.sleep(0.3)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-epmc", type=int, default=40)
    ap.add_argument("--only", choices=["ru", "en", "epmc"], action="append")
    args = ap.parse_args()
    only = set(args.only or ["ru", "en", "epmc"])
    with httpx.Client(headers=UA, timeout=60, follow_redirects=True) as client:
        if "ru" in only:
            print("Wikipedia RU…", flush=True)
            write("wikipedia_ru", fetch_wiki(client, "ru", WIKI_RU))
        if "en" in only:
            print("Wikipedia EN…", flush=True)
            write("wikipedia_en", fetch_wiki(client, "en", WIKI_EN))
        if "epmc" in only:
            print("Europe PMC (CC BY)…", flush=True)
            write("europepmc_ccby", fetch_epmc(client, args.max_epmc))


if __name__ == "__main__":
    main()
