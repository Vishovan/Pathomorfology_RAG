"""Точка входа для Streamlit Cloud (Main file path: streamlit_app.py).

Всё приложение живёт в pathology-rag/streamlit_app.py — здесь только запуск, чтобы не держать две копии.
Локально можно запускать напрямую: streamlit run pathology-rag/streamlit_app.py
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "pathology-rag" / "streamlit_app.py"), run_name="__main__")
