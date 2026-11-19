"""Сквозная проверка MCP через те же клиенты, что и в Streamlit."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathrag.mcp_calculators.client import invoke
from pathrag.mcp_calculators.network import invoke_network


def check(source):
    def request(method, params):
        if source == "stdio":
            return invoke(method, params)
        return invoke_network(source, method, params)

    deadline = time.monotonic() + 15
    while True:
        catalog, _, error = request("tools/list", {})
        if not error:
            break
        if source != "SSE" or time.monotonic() >= deadline:
            raise RuntimeError(f"{source}: {error}")
        time.sleep(0.3)

    tools = {tool["name"] for tool in catalog["tools"]}
    if source == "PubMed":
        expected = {"pubmed_search_articles"}
        name, args = "pubmed_search_articles", {"query": "33369635[PMID]"}
    elif source == "SSE":
        expected = {"gleason", "fnclcc", "allred", "search_images", "search_corpus"}
        name, args = "gleason", {"primary": 3, "secondary": 4}
    else:
        expected = {"nottingham", "ki67", "h_score", "list_calculators"}
        name, args = "nottingham", {"tubules": 2, "nuclei": 3, "mitoses": 2}
    if missing := expected - tools:
        raise RuntimeError(f"{source}: отсутствуют {sorted(missing)}")
    result, trace, error = request("tools/call", {"name": name, "arguments": args})
    if error:
        raise RuntimeError(error)
    if source == "PubMed":
        if "33369635" not in json.dumps(result):
            raise RuntimeError("В результате поиска отсутствует ожидаемый PMID.")
    else:
        data = result.get("structuredContent") or json.loads(result["content"][0]["text"])
        category = "grade" if source == "stdio" else "group"
        if (data["score"], data[category]) != (7, 2):
            raise RuntimeError(f"Неожиданный результат: {data}")
        bad = {"name": "ki67", "arguments": {"positive": 2, "total": 1}} if source == "stdio" else {
            "name": "allred", "arguments": {"proportion": 0, "intensity": 2}}
        failure, _, error = request("tools/call", bad)
        if not error or not failure.get("isError"):
            raise RuntimeError("Ожидалась структурированная ошибка инструмента.")
    print(json.dumps(trace, ensure_ascii=False, indent=2))
    print(f"{source}: проверка пройдена")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["stdio", "SSE", "PubMed"])
    check(parser.parse_args().source)
