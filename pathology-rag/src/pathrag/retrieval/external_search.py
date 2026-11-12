"""Инструменты внешнего поиска: PubMed, arXiv, bioRxiv и Веб-поиск."""
import logging
import xml.etree.ElementTree as ET
from typing import Any

import httpx

log = logging.getLogger("pathrag.retrieval.external")


def search_pubmed(query: str, max_results: int = 4) -> list[dict[str, Any]]:
    """Поиск статей в базе данных PubMed через NCBI E-utilities API."""
    results = []
    try:
        # 1. Поиск идентификаторов PMID
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
            "sort": "relevance",
        }
        with httpx.Client(timeout=8.0) as client:
            resp = client.get(search_url, params=params)
            if resp.status_code != 200:
                return []
            data = resp.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []

            # 2. Получение информации о статьях по списку PMID
            summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            sum_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "json",
            }
            sum_resp = client.get(summary_url, params=sum_params)
            if sum_resp.status_code != 200:
                return []
            sum_data = sum_resp.json().get("result", {})

            for pmid in id_list:
                doc = sum_data.get(pmid)
                if not doc:
                    continue
                title = doc.get("title", "Без названия")
                authors = [a.get("name") for a in doc.get("authors", []) if a.get("name")]
                source = doc.get("source", "PubMed")
                pubdate = doc.get("pubdate", "")
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                results.append({
                    "title": title,
                    "authors": authors[:3],
                    "journal": source,
                    "date": pubdate,
                    "url": url,
                    "pmid": pmid,
                    "source_type": "PubMed peer-reviewed",
                })
    except Exception as e:
        log.warning("PubMed search failed: %s", e)
    return results


def search_arxiv(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    """Поиск препринтов в arXiv API."""
    results = []
    try:
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": "0",
            "max_results": str(max_results),
            "sortBy": "relevance",
        }
        with httpx.Client(timeout=8.0) as client:
            resp = client.get(url, params=params)
            if resp.status_code != 200:
                return []

            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                title_elem = entry.find("atom:title", ns)
                summary_elem = entry.find("atom:summary", ns)
                id_elem = entry.find("atom:id", ns)

                title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else ""
                summary = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None else ""
                paper_url = id_elem.text.strip() if id_elem is not None else ""

                authors = []
                for author in entry.findall("atom:author", ns):
                    name_elem = author.find("atom:name", ns)
                    if name_elem is not None and name_elem.text:
                        authors.append(name_elem.text)

                if title:
                    results.append({
                        "title": title,
                        "authors": authors[:3],
                        "summary": summary[:300] + "..." if len(summary) > 300 else summary,
                        "url": paper_url,
                        "source_type": "arXiv preprint",
                    })
    except Exception as e:
        log.warning("arXiv search failed: %s", e)
    return results


def search_web(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    """Поиск в интернете через DuckDuckGo Search."""
    results = []
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))
            for r in raw_results:
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                    "source_type": "Web search",
                })
    except Exception as e:
        log.warning("Web search failed: %s", e)
    return results
