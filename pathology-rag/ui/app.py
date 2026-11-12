"""FastAPI веб-приложение для взаимодействия с Pathology RAG агентом."""
import sys
from pathlib import Path

# Добавляем src в PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pathrag.agent import ask_agent

app = FastAPI(title="Pathology RAG Agent", version="1.0.0")

# Монтирование статических файлов
static_dir = ROOT / "ui" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class ChatRequest(BaseModel):
    message: str
    mode: str | None = None  # pathology | images | legal | literature | None


class ChatResponse(BaseModel):
    answer: str
    intent: str
    citations: list[dict]
    disclaimer: str
    images: list[dict] = []
    literature: list[dict] = []


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = ROOT / "ui" / "templates" / "index.html"
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Pathology RAG Web Interface</h1>"


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    result = ask_agent(req.message, mode=req.mode)
    return ChatResponse(
        answer=result.get("final_answer", ""),
        intent=result.get("intent", ""),
        citations=result.get("citations", []),
        disclaimer=result.get("disclaimer", ""),
        images=result.get("retrieved_images", []),
        literature=result.get("literature_results", []),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ui.app:app", host="0.0.0.0", port=8000, reload=False)
