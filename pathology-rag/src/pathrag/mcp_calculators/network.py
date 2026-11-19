import asyncio
import json
from urllib.parse import urljoin, urlsplit

import httpx

from . import local_service
from .client import REVISION, ProtocolError

LOCAL = "http://127.0.0.1:8000/sse"
PUBMED = "https://pubmed.caseyjhand.com/mcp"


async def events(response):
    kind, data = "message", []
    async for line in response.aiter_lines():
        if not line:
            if data:
                yield kind, "\n".join(data)
            kind, data = "message", []
        elif line.startswith("event:"):
            kind = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))


async def dialog(send, method, params, trace):
    async def request(packet):
        trace.append(("Запрос", packet))
        response = await send(packet)
        if "id" not in packet:
            return
        trace.append(("Ответ", response))
        if response.get("id") != packet["id"] or response.get("jsonrpc") != "2.0":
            raise ProtocolError("Некорректный идентификатор или версия JSON-RPC.")
        if "error" in response:
            raise ProtocolError(response["error"]["message"])
        return response["result"]

    await request({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                   "params": {"protocolVersion": REVISION, "capabilities": {},
                              "clientInfo": {"name": "pathrag", "version": "1"}}})
    await request({"jsonrpc": "2.0", "method": "notifications/initialized"})
    return await request({"jsonrpc": "2.0", "id": 2, "method": method, "params": params})


async def local(client, method, params, trace):
    async with client.stream("GET", LOCAL, headers={"Accept": "text/event-stream"}) as response:
        response.raise_for_status()
        stream = events(response)
        kind, path = await stream.__anext__()
        endpoint = urljoin(LOCAL, path)
        if kind != "endpoint" or urlsplit(endpoint).netloc != urlsplit(LOCAL).netloc or urlsplit(endpoint).scheme != "http":
            raise ProtocolError("Локальный сервер вернул неизвестный адрес сессии.")

        async def send(packet):
            reply = await client.post(endpoint, json=packet)
            reply.raise_for_status()
            if "id" in packet:
                async for kind, data in stream:
                    if kind == "message":
                        value = json.loads(data)
                        if "id" in value:
                            return value
                raise ProtocolError("Сервер закрыл поток SSE.")

        return await dialog(send, method, params, trace)


async def remote(client, method, params, trace):
    headers = {"Accept": "application/json, text/event-stream"}

    async def send(packet):
        async with client.stream("POST", PUBMED, json=packet, headers=headers) as response:
            response.raise_for_status()
            if "mcp-session-id" in response.headers:
                headers["Mcp-Session-Id"] = response.headers["mcp-session-id"]
            if "id" not in packet:
                return
            if "text/event-stream" in response.headers.get("content-type", ""):
                async for kind, data in events(response):
                    if kind == "message":
                        value = json.loads(data)
                        if "id" in value:
                            break
                else:
                    raise ProtocolError("PubMed закрыл поток без результата.")
            else:
                value = json.loads(await response.aread())
            if packet["method"] == "initialize" and "result" in value:
                headers["MCP-Protocol-Version"] = value["result"]["protocolVersion"]
            return value

    try:
        return await dialog(send, method, params, trace)
    finally:
        if "Mcp-Session-Id" in headers:
            await client.delete(PUBMED, headers=headers)


async def run(source, method, params, trace, autostart=False):
    async with httpx.AsyncClient(timeout=8, trust_env=source == "PubMed") as client:
        if source == "PubMed":
            return await remote(client, method, params, trace)
        try:
            return await local(client, method, params, trace)
        except httpx.ConnectError:
            if trace:
                raise
            process = local_service.start() if autostart else None
            for _ in range(20 if autostart else 1):
                await asyncio.sleep(0.3)
                try:
                    return await local(client, method, params, trace)
                except httpx.ConnectError:
                    if trace:
                        raise
                    if process is not None and process.poll() is not None:
                        raise RuntimeError(f"SSE-сервер завершился с кодом {process.returncode}. Подробности в журнале Streamlit.") from None
            raise RuntimeError(f"SSE-сервер недоступен по адресу {LOCAL}. Проверьте запуск сервера на порту 8000.") from None


def invoke_network(source, method, params, *, autostart=False):
    trace = []
    try:
        result = asyncio.run(asyncio.wait_for(run(source, method, params, trace, autostart), timeout=12))
    except (httpx.HTTPError, TimeoutError, ValueError, KeyError, StopAsyncIteration, ProtocolError, RuntimeError, OSError) as exc:
        return {}, trace, str(exc) or "Сервис не ответил вовремя."
    error = "\n".join(x.get("text", "") for x in result.get("content", [])) if result.get("isError") else None
    return result, trace, error
