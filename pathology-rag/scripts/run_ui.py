#!/usr/bin/env python
"""Запуск веб-сервера UI для Pathology RAG."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run Pathology RAG Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    # Импортируем app напрямую
    from ui.app import app

    print("=" * 65)
    print(f"🚀 Запуск Pathology RAG Web UI на http://{args.host}:{args.port}")
    print("=" * 65)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
