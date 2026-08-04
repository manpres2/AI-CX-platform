"""
main_launcher.py - Single front-door page for the local AI platform.
No auth, no data — just a tile grid linking out (new tab) to the four apps:
Apex Bank Bot (8000), TechCare Support Bot (8001), Meeting Intelligence (8002),
Unified Ops Portal (8003).

Run from app/ folder: uvicorn main_launcher:app --host 0.0.0.0 --port 8004
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static_launcher"

app = FastAPI(title="Local AI Platform Launcher")


@app.get("/")
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Place index.html in static_launcher/ folder.</h2>")


@app.get("/health")
async def health():
    return {"status": "ok"}
