#!/usr/bin/env python
"""Интерактивный терминальный чат с Pathology RAG агентом на LangGraph."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathrag.agent import ask_agent


def main():
    parser = argparse.ArgumentParser(description="Pathology RAG CLI Agent")
    parser.add_argument("--mode", choices=["auto", "pathology", "images", "legal", "literature"], default="auto")
    args = parser.parse_args()

    print("=" * 70)
    print("🔬 PATHOLOGY RAG AGENT (LangGraph + LlamaIndex)")
    print(f"Режим: {args.mode.upper()}")
    print("Доступные команды: :mode <name>, :exit, :quit")
    print("=" * 70)

    current_mode = None if args.mode == "auto" else args.mode

    while True:
        try:
            query = input("\n[Вы]> ").strip()
            if not query:
                continue
            if query.lower() in (":exit", ":quit", "exit", "quit"):
                print("Выход из ассистента.")
                break
            if query.startswith(":mode"):
                parts = query.split()
                if len(parts) > 1:
                    new_mode = parts[1].lower()
                    current_mode = None if new_mode == "auto" else new_mode
                    print(f"Режим изменен на: {new_mode}")
                continue

            print("\n⏳ Обработка графом LangGraph...")
            result = ask_agent(query, mode=current_mode)

            print(f"\n[Ассистент | Маршрут: {result.get('intent')}]")
            print("-" * 50)
            print(result.get("final_answer"))

            images = result.get("retrieved_images", [])
            if images:
                print("\n🖼️ НАЙДЕННЫЕ МИКРОПРЕПАРАТЫ:")
                for i, img in enumerate(images, 1):
                    print(f"  #{i} {img.get('title')} ({img.get('stain')}, {img.get('magnification')})")
                    if img.get("wsi_url"):
                        print(f"      WSI Слайд: {img.get('wsi_url')}")

            print("\n" + result.get("disclaimer", ""))
            print("-" * 50)

        except (KeyboardInterrupt, EOFError):
            print("\nСессия завершена.")
            break


if __name__ == "__main__":
    main()
