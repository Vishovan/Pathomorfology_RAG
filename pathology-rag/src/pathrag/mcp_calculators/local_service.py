"""Один локальный SSE-процесс на экземпляр Streamlit."""
import atexit
import importlib.util
import subprocess
import sys
import threading
from pathlib import Path

_lock = threading.Lock()
_process = None


def start():
    global _process
    with _lock:
        if _process is not None and _process.poll() is None:
            return _process
        if importlib.util.find_spec("mcp") is None:
            raise RuntimeError("В окружении Streamlit отсутствует MCP SDK. Установите зависимости из requirements.txt.")
        _process = subprocess.Popen(
            [sys.executable, "-m", "pathrag.mcp_calculators.network_server"],
            cwd=Path(__file__).resolve().parents[2],
        )
        return _process


@atexit.register
def stop():
    with _lock:
        if _process is not None and _process.poll() is None:
            _process.terminate()
            try:
                _process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                _process.kill()
                _process.wait()
