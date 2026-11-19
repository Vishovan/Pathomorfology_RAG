"""Короткоживущая stdio-сессия с записью переданных пакетов."""
import asyncio
import json
import sys
from pathlib import Path

MODULE = "pathrag.mcp_calculators.server"
REVISION = "2025-11-25"


class ProtocolError(Exception):
    pass


async def exchange(method, params, trace):
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", MODULE, cwd=Path(__file__).resolve().parents[2],
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, limit=2**20,
    )

    async def send(packet):
        process.stdin.write(json.dumps(packet, ensure_ascii=False).encode() + b"\n")
        await process.stdin.drain()
        trace.append(("Запрос", packet))

    async def request(number, name, arguments):
        await send({"jsonrpc": "2.0", "id": number, "method": name, "params": arguments})
        while line := await process.stdout.readline():
            packet = json.loads(line)
            trace.append(("Ответ", packet))
            if packet.get("jsonrpc") != "2.0":
                raise ProtocolError("Сервер отправил некорректный пакет JSON-RPC.")
            if "id" not in packet and "method" in packet:
                continue
            if packet.get("id") != number:
                raise ProtocolError("Идентификатор ответа не совпадает с запросом.")
            if "error" in packet:
                raise ProtocolError(packet["error"].get("message", "Ошибка JSON-RPC"))
            return packet["result"]
        raise ProtocolError("Сервер закрыл stdout. Проверьте установку MCP SDK и журнал запуска.")

    try:
        info = await request(1, "initialize", {"protocolVersion": REVISION, "capabilities": {},
                                               "clientInfo": {"name": "pathrag", "version": "1"}})
        if info.get("protocolVersion") != REVISION or "tools" not in info.get("capabilities", {}):
            raise ProtocolError("Сервер не подтвердил версию протокола или поддержку инструментов.")
        await send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return await request(2, method, params)
    finally:
        process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), 1)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()


def invoke(method, params):
    trace = []
    try:
        result = asyncio.run(asyncio.wait_for(exchange(method, params, trace), timeout=12))
    except asyncio.TimeoutError:
        return {}, trace, "Время ожидания ответа истекло."
    except (OSError, ValueError, KeyError, ProtocolError) as exc:
        return {}, trace, str(exc)
    if result.get("isError"):
        message = "\n".join(c.get("text", "") for c in result.get("content", []))
        return result, trace, message or "Ошибка инструмента."
    return result, trace, None
