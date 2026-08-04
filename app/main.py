"""
main.py - Voice AI BFSI Demo with Admin Panel
FastAPI WebSocket: Whisper STT + ChromaDB RAG + Ollama LLaMA + Kokoro TTS
Admin panel at: http://localhost:8000/admin  (login required)
Run from app/ folder: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import difflib
import io
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import whisper
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.websockets import WebSocketState
from fastapi.staticfiles import StaticFiles
import httpx
from kokoro import KPipeline

import auth

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import docx as python_docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    import io as _io
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    try:
        load_dotenv(_env_path, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(_env_path, encoding="utf-16")

BASE_DIR     = Path(__file__).parent.parent
STATIC_DIR   = BASE_DIR / "static"
LOG_DIR      = BASE_DIR / "logs"
KB_DOCS      = BASE_DIR / "kb_docs"
KB_STORE     = BASE_DIR / "kb_store"
PROMPT_FILE  = BASE_DIR / "prompt_config.json"
BRAND_FILE   = BASE_DIR / "branding.json"
BRAND_LOGO   = STATIC_DIR / "brand-logo"   # no extension — stored with original ext alongside

for d in [LOG_DIR, STATIC_DIR, KB_DOCS, KB_STORE]:
    d.mkdir(exist_ok=True)

DEFAULT_BRANDING = {
    "bank_name":  "Apex Bank",
    "tagline":    "AI-Powered Customer Voice Assistant",
    "badge_text": "Live Demo — Manpreet Singh | CPaaS Presales",
    "logo_emoji": "🏦",
}

def load_branding() -> dict:
    if BRAND_FILE.exists():
        try:
            return json.loads(BRAND_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return DEFAULT_BRANDING.copy()

def save_branding(data: dict):
    BRAND_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def find_logo_file() -> Path | None:
    for ext in ("png", "jpg", "jpeg", "svg", "webp", "gif"):
        p = STATIC_DIR / f"brand-logo.{ext}"
        if p.exists():
            return p
    return None

OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_BASE   = OLLAMA_URL.split("/api/")[0]
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
SAMPLE_RATE   = 24000
EMBED_MODEL   = "all-MiniLM-L6-v2"
COLLECTION    = "apex_bank_bfsi"
RAG_TOP_K     = 3

# Admin credentials — set in .env or fall back to defaults
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "apexbank2026")

# ── Runtime config: TTS voice + LLM model (editable via admin panel) ───────────
RUNTIME_FILE = BASE_DIR / "runtime_config.json"

# Kokoro-82M bundled American-English voices (KPipeline lang_code="a")
AVAILABLE_VOICES = {
    "female": ["af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
               "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky"],
    "male":   ["am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
               "am_michael", "am_onyx", "am_puck", "am_santa"],
}

DEFAULT_RUNTIME_CONFIG = {
    "kokoro_voice": os.getenv("KOKORO_VOICE", "af_heart"),
    "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
}

def load_runtime_config() -> dict:
    if RUNTIME_FILE.exists():
        try:
            return {**DEFAULT_RUNTIME_CONFIG, **json.loads(RUNTIME_FILE.read_text(encoding="utf-8"))}
        except Exception:
            pass
    return DEFAULT_RUNTIME_CONFIG.copy()

def save_runtime_config(cfg: dict):
    RUNTIME_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

_runtime_config = load_runtime_config()
KOKORO_VOICE  = _runtime_config["kokoro_voice"]
OLLAMA_MODEL  = _runtime_config["ollama_model"]

# ── Prompt config (editable via admin panel) ──────────────────────────────────
DEFAULT_PROMPT_CONFIG = {
    "system_prompt": (
        "You are a helpful voice assistant for Apex Bank, a fictional Indian bank.\n"
        "You assist customers on a phone IVR system.\n"
        "Keep every answer to 2-3 SHORT sentences maximum — this is a voice call, not text.\n"
        "Never use bullet points, markdown, or numbered lists.\n"
        "Speak in a warm, professional, conversational tone.\n"
        "If the customer asks something outside banking, politely say you can only help with banking queries.\n"
        "Always be factual — only state what is in the context provided.\n"
        "CRITICAL VOICE RULES:\n"
        "- Always write 'Rs.' as 'Rupees' in full so it is pronounced correctly by text-to-speech.\n"
        "- Never start or end a reply with 'Thank you for calling', 'Thank you for contacting', or any variation — the customer is already on the call.\n"
        "- Get straight to the answer. No preamble, no sign-off phrases."
    ),
    "greeting": "Thank you for calling Apex Bank. This is your AI assistant. How may I help you today?",
    "guardrails": [
        "Never discuss competitor banks by name",
        "Never provide specific account balances or personal financial data",
        "Never discuss politics, religion, or non-banking topics",
        "If asked about specific interest rates, always say 'subject to change — please visit your nearest branch'",
        "Never make loan approval promises — only describe eligibility criteria",
    ],
    "kb_filter": "",
    "rag_top_k": 3,
    "fallback_message": "I'm sorry, I'm unable to help with that right now. Please call our helpline at 1800-123-4567.",
}

def load_prompt_config() -> dict:
    if PROMPT_FILE.exists():
        try:
            return json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return DEFAULT_PROMPT_CONFIG.copy()

def save_prompt_config(cfg: dict):
    PROMPT_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

prompt_config = load_prompt_config()

LOG_FILE = LOG_DIR / "server.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),                                      # console
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),         # file
    ]
)
log = logging.getLogger(__name__)
SERVER_START = time.time()

# ── Model loading ─────────────────────────────────────────────────────────────
log.info("Loading Whisper (%s)...", WHISPER_MODEL)
stt_model = whisper.load_model(WHISPER_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
log.info("Whisper on %s", "CUDA" if torch.cuda.is_available() else "CPU")

log.info("Loading Kokoro TTS...")
tts_pipeline = KPipeline(lang_code="a")
log.info("Kokoro ready.")

# ── Document ingestion (.txt / .pdf / .docx) ────────────────────────────────
KB_EXTENSIONS = (".txt", ".pdf", ".docx")

def _ocr_image_bytes(img_bytes: bytes) -> str:
    """Best-effort OCR on a single embedded image. Returns '' if OCR isn't
    available (no pytesseract / no Tesseract-OCR binary on this machine) or
    the image has no readable text — never raises, ingestion must not break
    just because a picture couldn't be read."""
    if not OCR_AVAILABLE:
        return ""
    try:
        text = pytesseract.image_to_string(Image.open(_io.BytesIO(img_bytes))).strip()
        return text
    except Exception as e:
        log.debug("OCR skipped for one image: %s", e)
        return ""

def _flatten_table(rows: list[list[str]]) -> str:
    """Turn a table's rows into self-contained 'Header: value | Header: value'
    lines. Plain markdown/pipe tables fall apart once naive word-based
    chunking slices across row/column boundaries — each line here repeats its
    column headers so a row stays meaningful even if it ends up alone in a
    chunk."""
    rows = [[(c or "").strip().replace("\n", " ") for c in r] for r in rows if r]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    header, body = rows[0], rows[1:]
    if not body:  # single-row "table" — just join it as one line
        return " | ".join(c for c in header if c)
    lines = []
    for row in body:
        pairs = [f"{h}: {v}" for h, v in zip(header, row) if h or v]
        if pairs:
            lines.append(" | ".join(pairs))
    return "\n".join(lines)

def _extract_pdf_text(path: Path, quick: bool = False) -> str:
    parts = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            if quick:
                # Fast path for the KB file-list view — it only needs a rough
                # size/line-count estimate, not perfect table/OCR fidelity. Skipping
                # table detection and per-image OCR is what makes this fast: on a
                # large image-heavy PDF, the full extraction below can take 20+
                # seconds per file, which made the file list itself take that long
                # to load. Full extraction is still used for Inspect and KB rebuild.
                text = page.get_text()
                if text.strip():
                    parts.append(text.strip())
                continue
            table_rects = []
            try:
                tables = page.find_tables()
                for tbl in tables.tables:
                    flat = _flatten_table(tbl.extract())
                    if flat:
                        parts.append(flat)
                    table_rects.append(fitz.Rect(tbl.bbox))
            except Exception as e:
                log.debug("Table extraction failed on a PDF page: %s", e)

            # Reconstruct body text word-by-word, skipping words inside a table
            # region — those are already covered (in proper row/column form) by
            # the flattened table above, so this avoids dumping the same cells
            # again as a jumbled, headerless run of text.
            words = page.get_text("words")
            kept = [w for w in words if not any(fitz.Rect(w[:4]).intersects(r) for r in table_rects)]
            kept.sort(key=lambda w: (w[5], w[6], w[7]))  # block, line, word order
            page_text = " ".join(w[4] for w in kept)
            if page_text.strip():
                parts.append(page_text.strip())

            for img in page.get_images(full=True):
                try:
                    img_bytes = doc.extract_image(img[0])["image"]
                    ocr_text = _ocr_image_bytes(img_bytes)
                    if ocr_text:
                        parts.append(ocr_text)
                except Exception as e:
                    log.debug("Image extraction failed on a PDF page: %s", e)
    return "\n\n".join(parts)

def _extract_docx_text(path: Path, quick: bool = False) -> str:
    parts = []
    d = python_docx.Document(str(path))
    for para in d.paragraphs:
        if para.text.strip():
            parts.append(para.text.strip())
    for table in d.tables:
        rows = [[cell.text for cell in row.cells] for row in table.rows]
        flat = _flatten_table(rows)
        if flat:
            parts.append(flat)
    if quick:
        # Fast path for the KB file-list view — skips per-image OCR (see
        # _extract_pdf_text for why). Full extraction still runs for Inspect/rebuild.
        return "\n\n".join(parts)
    for shape in d.inline_shapes:
        try:
            rel_id = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
            img_bytes = d.part.related_parts[rel_id].blob
            ocr_text = _ocr_image_bytes(img_bytes)
            if ocr_text:
                parts.append(ocr_text)
        except Exception as e:
            log.debug("Image extraction failed on a DOCX shape: %s", e)
    return "\n\n".join(parts)

def extract_document_text(path: Path, quick: bool = False) -> str:
    """Extract plain text from a .txt/.pdf/.docx KB doc — tables are flattened
    into header:value lines and any text found in embedded pictures (OCR) is
    appended, so downstream chunking/embedding sees it all as ordinary text.
    Pass quick=True for a fast, approximate extraction (skips OCR and PDF table
    detection) — used by the KB file-list view, which only needs a rough size
    estimate, not the full-fidelity text that Inspect/rebuild require."""
    ext = path.suffix.lower()
    try:
        if ext == ".txt":
            return path.read_text(encoding="utf-8", errors="ignore")
        if ext == ".pdf":
            if not PDF_AVAILABLE:
                log.warning("Skipping %s — PyMuPDF not installed", path.name)
                return ""
            return _extract_pdf_text(path, quick=quick)
        if ext == ".docx":
            if not DOCX_AVAILABLE:
                log.warning("Skipping %s — python-docx not installed", path.name)
                return ""
            return _extract_docx_text(path, quick=quick)
    except Exception as e:
        log.error("Failed to extract text from %s: %s", path.name, e)
    return ""

# ── RAG setup ─────────────────────────────────────────────────────────────────
chroma_client = None
kb_collection  = None
embedder       = None

def init_rag():
    global chroma_client, kb_collection, embedder
    if not RAG_AVAILABLE:
        return False
    if not KB_STORE.exists() or not any(KB_STORE.iterdir()):
        log.warning("KB store empty — run index_kb.py first.")
        return False
    try:
        chroma_client = chromadb.PersistentClient(path=str(KB_STORE))
        kb_collection = chroma_client.get_collection(COLLECTION)
        if embedder is None:
            log.info("Loading sentence-transformer...")
            embedder = SentenceTransformer(EMBED_MODEL)
        log.info("RAG ready. %d chunks.", kb_collection.count())
        return True
    except Exception as e:
        log.warning("RAG init failed: %s", e)
        return False

init_rag()

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="BFSI Voice AI Demo")
security = HTTPBasic()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Multi-user admin auth — shared users.db at the repo root (one level up from
# BASE_DIR here, since this app's app/ folder sits directly at the repo root).
# See app/auth.py for the shared implementation (duplicated per app).
auth.configure(BASE_DIR / "users.db", app_key="bank")
auth.ensure_bootstrap_user(ADMIN_USER, ADMIN_PASS)
verify_admin = auth.verify_admin

# ── Public routes ─────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Place index.html in static/ folder.</h2>")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "whisper": WHISPER_MODEL,
        "tts": KOKORO_VOICE,
        "ollama_model": OLLAMA_MODEL,
        "rag": kb_collection.count() if kb_collection else "not loaded",
        "cuda": torch.cuda.is_available(),
    }

@app.get("/api/logs")
async def list_logs():
    files = sorted(LOG_DIR.glob("*.wav"), key=lambda f: f.stat().st_mtime, reverse=True)
    result = []
    total = 0
    for f in files[:30]:
        sz = f.stat().st_size // 1024
        total += sz
        result.append({
            "name": f.name,
            "size_kb": sz,
            "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%d %b %H:%M:%S")
        })
    return {"files": result, "total_kb": total}

@app.get("/api/logs/{filename}")
async def get_log_file(filename: str):
    p = LOG_DIR / filename
    if not p.exists() or not p.suffix == ".wav":
        raise HTTPException(404, "File not found")
    return FileResponse(str(p), media_type="audio/wav", filename=filename)

@app.get("/api/sysinfo")
async def sysinfo():
    wavs = list(LOG_DIR.glob("*.wav"))
    uptime_s = int(time.time() - SERVER_START)
    h, m, s = uptime_s//3600, (uptime_s%3600)//60, uptime_s%60
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "wav_count": len(wavs),
        "wav_total_kb": sum(f.stat().st_size for f in wavs) // 1024,
        "kb_chunks": kb_collection.count() if kb_collection else 0,
        "uptime": f"{h}h {m}m {s}s",
    }

@app.get("/admin/api/serverlogs")
async def get_server_logs(username: str = Depends(verify_admin), lines: int = 500):
    """Return last N lines of server.log with error/warning counts."""
    if not LOG_FILE.exists():
        return {"lines": [], "errors": 0, "warnings": 0}
    all_lines = LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
    tail = all_lines[-lines:]
    errors   = sum(1 for l in tail if "[ERROR]" in l)
    warnings = sum(1 for l in tail if "[WARNING]" in l or "[WARN]" in l)
    return {"lines": tail, "errors": errors, "warnings": warnings, "total": len(all_lines)}

@app.post("/admin/api/serverlogs/analyze")
async def analyze_server_logs(data: dict, username: str = Depends(verify_admin)):
    """Send recent error/warning log lines through the active LLM provider (local or
    cloud, whichever is configured) for a plain-English diagnosis. Reuses the same
    generate_llm_reply() dispatcher the conversation flow uses, so this respects
    whatever provider is currently selected in the admin panel."""
    if not LOG_FILE.exists():
        return {"error": "No server log file yet."}
    level = data.get("level", "error")
    max_lines = min(int(data.get("lines", 400)), 1000)
    recent = LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()[-max_lines:]
    if level == "error":
        relevant = [l for l in recent if "[ERROR]" in l]
    elif level == "warning":
        relevant = [l for l in recent if "[ERROR]" in l or "[WARNING]" in l or "[WARN]" in l]
    else:
        relevant = recent
    if not relevant:
        return {"error": f"No {level} lines found in the last {max_lines} log lines."}
    # Cap what actually gets sent to the LLM — a long log dump risks blowing the
    # context window or timing out, and the most recent lines matter most.
    log_text = "\n".join(relevant[-150:])
    prompt = (
        "You are a senior backend engineer helping troubleshoot a FastAPI voice-AI "
        "server (Whisper STT + local/cloud LLM + Kokoro/cloud TTS + ChromaDB RAG over "
        "a WebSocket). Below are recent log lines from the running server. Identify "
        "what's going wrong, explain the likely root cause in plain English, and "
        "suggest concrete next steps to investigate or fix it. If there are multiple "
        "distinct issues, list each separately. Keep it concise and actionable — this "
        "is read by the person operating the server, not a formal report.\n\n"
        f"LOG LINES:\n{log_text}"
    )
    try:
        analysis = await generate_llm_reply(prompt)
        return {"analysis": analysis, "lines_analyzed": len(relevant)}
    except Exception as e:
        return {"error": f"LLM request failed: {e}"}

@app.delete("/admin/api/logs/{filename}")
async def delete_wav(filename: str, username: str = Depends(verify_admin)):
    p = LOG_DIR / filename
    if not p.exists() or p.suffix != ".wav":
        raise HTTPException(404, "WAV file not found")
    p.unlink()
    log.info("WAV deleted by admin: %s", filename)
    return {"status": "deleted"}

@app.post("/admin/api/logs/clear")
async def clear_all_wav(username: str = Depends(verify_admin)):
    wavs = list(LOG_DIR.glob("*.wav"))
    for f in wavs:
        f.unlink()
    log.info("All WAV files cleared by admin (%d files)", len(wavs))
    return {"status": "cleared", "count": len(wavs)}

@app.post("/admin/api/shutdown")
async def shutdown_server(username: str = Depends(verify_admin)):
    """Terminate this server process (and its uvicorn --reload parent, if any)
    so GPU-resident models (Whisper/TTS/LLM) are unloaded and VRAM is freed."""
    log.warning("Server shutdown requested by admin (%s)", username)

    def _kill():
        time.sleep(1)  # let the HTTP response reach the client first
        pid, ppid = os.getpid(), os.getppid()
        for target in {ppid, pid}:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(target)],
                            capture_output=True)

    threading.Thread(target=_kill, daemon=True).start()
    return {"status": "shutting down"}

# ── Admin routes ──────────────────────────────────────────────────────────────
@app.get("/admin")
async def admin_panel(username: str = Depends(verify_admin)):
    admin_html = STATIC_DIR / "admin.html"
    if admin_html.exists():
        return HTMLResponse(admin_html.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Place admin.html in static/ folder.</h2>")

@app.get("/admin/api/prompt")
async def get_prompt(username: str = Depends(verify_admin)):
    return load_prompt_config()

@app.post("/admin/api/prompt")
async def save_prompt(data: dict, username: str = Depends(verify_admin)):
    global prompt_config
    prompt_config = data
    save_prompt_config(data)
    log.info("Prompt config updated by admin.")
    return {"status": "saved"}

@app.get("/admin/api/runtime-config")
async def get_runtime_config(username: str = Depends(verify_admin)):
    installed_models = []
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            resp.raise_for_status()
            installed_models = [m["name"] for m in resp.json().get("models", [])]
    except Exception as e:
        log.warning("Could not fetch Ollama model list: %s", e)
    return {
        "kokoro_voice": KOKORO_VOICE,
        "ollama_model": OLLAMA_MODEL,
        "available_voices": AVAILABLE_VOICES,
        "available_models": installed_models,
    }

@app.post("/admin/api/runtime-config")
async def save_runtime_config_api(data: dict, username: str = Depends(verify_admin)):
    global KOKORO_VOICE, OLLAMA_MODEL
    voice = (data.get("kokoro_voice") or KOKORO_VOICE).strip()
    model = (data.get("ollama_model") or OLLAMA_MODEL).strip()
    KOKORO_VOICE = voice
    OLLAMA_MODEL = model
    save_runtime_config({"kokoro_voice": voice, "ollama_model": model})
    log.info("Runtime config updated by admin: voice=%s model=%s", voice, model)
    return {"status": "saved", "kokoro_voice": voice, "ollama_model": model}

VOICE_SAMPLE_TEXT = "Hi, thanks for calling Apex Bank. This is a quick preview of this voice."

@app.get("/admin/api/voice-sample")
async def voice_sample(voice: str, username: str = Depends(verify_admin)):
    """Synthesize a short sample line in the requested voice, without touching the
    saved KOKORO_VOICE setting or the call-recording WAV logs — this is a preview
    only, so it renders straight to an in-memory response."""
    valid_voices = AVAILABLE_VOICES["female"] + AVAILABLE_VOICES["male"]
    if voice not in valid_voices:
        raise HTTPException(400, "Unknown voice")
    loop = asyncio.get_event_loop()
    pcm = await loop.run_in_executor(None, synthesize, VOICE_SAMPLE_TEXT, voice)
    return Response(content=pcm_to_wav_bytes(pcm), media_type="audio/wav")

# ── Pluggable provider config (local ↔ cloud LLM/TTS) ──────────────────────────
def _mask_provider_config(cfg: dict) -> dict:
    """Never send real API keys to the browser — replace each with a has_key flag."""
    out = json.loads(json.dumps(cfg))
    out["llm_cloud"]["has_key"] = bool(out["llm_cloud"].pop("api_key", ""))
    for vals in out["tts_cloud"].values():
        vals["has_key"] = bool(vals.pop("api_key", ""))
    return out

@app.get("/admin/api/providers")
async def get_providers(username: str = Depends(verify_admin)):
    return _mask_provider_config(load_provider_config())

@app.post("/admin/api/providers")
async def save_providers(data: dict, username: str = Depends(verify_admin)):
    """Save/'make live' the provider config. A blank api_key in the payload keeps
    the previously saved key rather than overwriting it with an empty string."""
    cfg = load_provider_config()
    cfg["llm_mode"] = data.get("llm_mode", cfg["llm_mode"])
    incoming_llm = data.get("llm_cloud", {})
    cfg["llm_cloud"]["base_url"] = incoming_llm.get("base_url", cfg["llm_cloud"]["base_url"])
    cfg["llm_cloud"]["model"] = incoming_llm.get("model", cfg["llm_cloud"]["model"])
    if incoming_llm.get("api_key"):
        cfg["llm_cloud"]["api_key"] = incoming_llm["api_key"]

    cfg["tts_mode"] = data.get("tts_mode", cfg["tts_mode"])
    cfg["tts_cloud_engine"] = data.get("tts_cloud_engine", cfg["tts_cloud_engine"])
    for engine, incoming in data.get("tts_cloud", {}).items():
        existing = cfg["tts_cloud"].setdefault(engine, {})
        for k, v in incoming.items():
            if k == "api_key" and not v:
                continue
            existing[k] = v

    save_provider_config(cfg)
    log.info("Provider config updated by admin: llm_mode=%s tts_mode=%s", cfg["llm_mode"], cfg["tts_mode"])
    return {"status": "saved"}

@app.post("/admin/api/providers/test-llm")
async def test_llm(data: dict, username: str = Depends(verify_admin)):
    """Test candidate (not-yet-saved) cloud LLM settings before committing them."""
    saved = load_provider_config()["llm_cloud"]
    cfg = {
        "base_url": data.get("base_url") or saved.get("base_url", ""),
        "model": data.get("model") or saved.get("model", ""),
        "api_key": data.get("api_key") or saved.get("api_key", ""),
    }
    try:
        reply = await generate_llm_cloud("Say OK if you can hear me.", ["\nUser:"], cfg)
        return {"reply": reply}
    except Exception as e:
        return {"error": str(e)}

@app.post("/admin/api/providers/test-tts")
async def test_tts(data: dict, username: str = Depends(verify_admin)):
    """Test candidate (not-yet-saved) cloud TTS settings before committing them."""
    engine = data.get("engine", "elevenlabs")
    saved = load_provider_config()["tts_cloud"].get(engine, {})
    cfg = {**saved, **{k: v for k, v in data.items() if k not in ("engine", "text") and v}}
    text = data.get("text") or "Hi, this is a quick preview of this cloud voice."
    fn = _TTS_CLOUD_ENGINES.get(engine, synthesize_elevenlabs)
    try:
        loop = asyncio.get_event_loop()
        pcm = await loop.run_in_executor(None, fn, text, cfg)
        return Response(content=pcm_to_wav_bytes(pcm), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(400, str(e))

def _kb_files() -> list[Path]:
    return sorted(p for ext in KB_EXTENSIONS for p in KB_DOCS.glob(f"*{ext}"))

@app.get("/admin/api/kb/files")
async def list_kb_files(username: str = Depends(verify_admin)):
    files = []
    for f in _kb_files():
        text = extract_document_text(f, quick=True)
        files.append({
            "name": f.name,
            "type": f.suffix.lstrip(".").upper(),
            "size_kb": round(f.stat().st_size / 1024, 1),
            "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%d %b %Y %H:%M"),
            "lines": len(text.splitlines()),
        })
    return {"files": files, "count": len(files)}

@app.post("/admin/api/kb/upload")
async def upload_kb_file(file: UploadFile = File(...), username: str = Depends(verify_admin)):
    ext = Path(file.filename).suffix.lower()
    if ext not in KB_EXTENSIONS:
        raise HTTPException(400, "Only .txt, .pdf, and .docx files allowed")
    if ext == ".pdf" and not PDF_AVAILABLE:
        raise HTTPException(400, "PDF support not installed on server (PyMuPDF missing)")
    if ext == ".docx" and not DOCX_AVAILABLE:
        raise HTTPException(400, "DOCX support not installed on server (python-docx missing)")
    dest = KB_DOCS / file.filename
    content = await file.read()
    dest.write_bytes(content)
    log.info("KB file uploaded: %s (%d bytes)", file.filename, len(content))
    return {"status": "uploaded", "filename": file.filename, "size": len(content)}

@app.delete("/admin/api/kb/files/{filename}")
async def delete_kb_file(filename: str, username: str = Depends(verify_admin)):
    p = KB_DOCS / filename
    if not p.exists():
        raise HTTPException(404, "File not found")
    p.unlink()
    log.info("KB file deleted: %s", filename)
    return {"status": "deleted"}

@app.get("/admin/api/kb/view/{filename}")
async def view_kb_file(filename: str, username: str = Depends(verify_admin)):
    p = KB_DOCS / filename
    if not p.exists():
        raise HTTPException(404, "File not found")
    # For .pdf/.docx this returns the extracted plain text (tables flattened,
    # OCR'd image text appended) — i.e. exactly what gets embedded, not the raw file.
    return {"content": extract_document_text(p)}

@app.get("/admin/api/kb/download/{filename}")
async def download_kb_file(filename: str, username: str = Depends(verify_admin)):
    p = KB_DOCS / filename
    if not p.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(p), filename=filename)

@app.post("/admin/api/kb/rebuild")
async def rebuild_kb(username: str = Depends(verify_admin)):
    """Rebuild ChromaDB index from kb_docs/."""
    global chroma_client, kb_collection, embedder, WHISPER_NAME_HINT
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb as cdb

        log.info("Rebuilding KB index...")
        client = cdb.PersistentClient(path=str(KB_STORE))
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
        collection = client.create_collection(COLLECTION)

        if embedder is None:
            embedder = SentenceTransformer(EMBED_MODEL)

        doc_files = _kb_files()
        if not doc_files:
            return {"status": "error", "message": "No .txt/.pdf/.docx files in kb_docs/"}

        all_chunks, all_ids, all_metas = [], [], []
        skipped = []
        for doc_path in doc_files:
            text = extract_document_text(doc_path)
            if not text.strip():
                skipped.append(doc_path.name)
                continue
            words = text.split()
            i = 0
            chunk_i = 0
            while i < len(words):
                chunk = " ".join(words[i:i+300])
                if len(chunk.split()) > 20:
                    all_chunks.append(chunk)
                    all_ids.append(f"{doc_path.stem}_{chunk_i:04d}")
                    all_metas.append({"source": doc_path.name, "chunk": chunk_i})
                    chunk_i += 1
                i += 250

        if not all_chunks:
            return {"status": "error", "message": "No extractable text found in kb_docs/ files"}

        embeddings = embedder.encode(all_chunks, show_progress_bar=False).tolist()
        collection.add(documents=all_chunks, embeddings=embeddings,
                       ids=all_ids, metadatas=all_metas)

        # Reload global references
        chroma_client = client
        kb_collection = collection
        refresh_customer_records()
        WHISPER_NAME_HINT = build_whisper_name_hint()
        log.info("KB rebuilt: %d chunks from %d files (skipped: %s)",
                  len(all_chunks), len(doc_files), skipped or "none")
        return {"status": "ok", "chunks": len(all_chunks), "files": len(doc_files), "skipped": skipped}
    except Exception as e:
        log.error("KB rebuild failed: %s", e)
        return {"status": "error", "message": str(e)}

@app.post("/admin/api/kb/search")
async def search_kb(data: dict, username: str = Depends(verify_admin)):
    """Test a query against ChromaDB."""
    query = data.get("query", "")
    top_k = int(data.get("top_k", 3))
    if not query:
        raise HTTPException(400, "query required")
    if not kb_collection or not embedder:
        return {"error": "RAG not loaded — rebuild KB first"}
    try:
        qvec = embedder.encode([query]).tolist()
        results = kb_collection.query(query_embeddings=qvec, n_results=top_k)
        docs  = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        return {"query": query, "results": [
            {"text": d, "source": m.get("source","?"), "score": round(1-s, 3)}
            for d, m, s in zip(docs, metas, dists)
        ]}
    except Exception as e:
        return {"error": str(e)}

# ── Branding routes ───────────────────────────────────────────────────────────
@app.get("/admin/api/branding")
async def get_branding(username: str = Depends(verify_admin)):
    data = load_branding()
    data["has_logo_image"] = find_logo_file() is not None
    return data

@app.get("/api/branding")
async def get_branding_public():
    """Public endpoint — customer dashboard reads this on load."""
    data = load_branding()
    data["has_logo_image"] = find_logo_file() is not None
    return data

@app.post("/admin/api/branding")
async def save_branding_route(data: dict, username: str = Depends(verify_admin)):
    save_branding(data)
    log.info("Branding updated by admin.")
    return {"status": "saved"}

@app.get("/admin/api/branding/logo")
async def get_logo(username: str = Depends(verify_admin)):
    logo = find_logo_file()
    if not logo:
        raise HTTPException(404, "No logo uploaded")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "svg": "image/svg+xml", "webp": "image/webp", "gif": "image/gif"}
    return FileResponse(str(logo), media_type=mime.get(logo.suffix.lstrip("."), "image/png"))

@app.post("/admin/api/branding/logo")
async def upload_logo(file: UploadFile = File(...), username: str = Depends(verify_admin)):
    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in ("png", "jpg", "jpeg", "svg", "webp", "gif"):
        raise HTTPException(400, "Unsupported image type")
    # Remove any existing logo files first
    for old in STATIC_DIR.glob("brand-logo.*"):
        old.unlink()
    dest = STATIC_DIR / f"brand-logo.{ext}"
    dest.write_bytes(await file.read())
    log.info("Brand logo uploaded: %s", dest.name)
    return {"status": "uploaded", "filename": dest.name}

@app.delete("/admin/api/branding/logo")
async def delete_logo(username: str = Depends(verify_admin)):
    for f in STATIC_DIR.glob("brand-logo.*"):
        f.unlink()
    log.info("Brand logo removed by admin.")
    return {"status": "removed"}

# ── Background ambience audio ───────────────────────────────────────────────
BGNOISE_CONFIG_FILE = BASE_DIR / "bgnoise_config.json"
BGNOISE_EXTENSIONS  = ("mp3", "wav", "ogg", "m4a")
BGNOISE_MIME = {"mp3": "audio/mpeg", "wav": "audio/wav", "ogg": "audio/ogg", "m4a": "audio/mp4"}
DEFAULT_BGNOISE_CONFIG = {"enabled": False, "volume": 0.3}

def load_bgnoise_config() -> dict:
    if BGNOISE_CONFIG_FILE.exists():
        try:
            return {**DEFAULT_BGNOISE_CONFIG, **json.loads(BGNOISE_CONFIG_FILE.read_text(encoding="utf-8"))}
        except Exception:
            pass
    return DEFAULT_BGNOISE_CONFIG.copy()

def save_bgnoise_config(cfg: dict):
    BGNOISE_CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

def find_bgnoise_file() -> Path | None:
    for ext in BGNOISE_EXTENSIONS:
        p = STATIC_DIR / f"bg-noise.{ext}"
        if p.exists():
            return p
    return None

@app.get("/api/bgnoise")
async def get_bgnoise_public():
    """Public endpoint — the customer-facing call page polls this to decide
    whether to loop-play the uploaded ambience track, and at what volume."""
    cfg = load_bgnoise_config()
    return {"enabled": cfg["enabled"], "volume": cfg["volume"], "has_file": find_bgnoise_file() is not None}

@app.get("/api/bgnoise/file")
async def get_bgnoise_file_public():
    f = find_bgnoise_file()
    if not f:
        raise HTTPException(404, "No background audio uploaded")
    return FileResponse(str(f), media_type=BGNOISE_MIME.get(f.suffix.lstrip("."), "audio/mpeg"))

@app.get("/admin/api/bgnoise")
async def get_bgnoise_admin(username: str = Depends(verify_admin)):
    cfg = load_bgnoise_config()
    f = find_bgnoise_file()
    return {"enabled": cfg["enabled"], "volume": cfg["volume"], "has_file": f is not None,
            "filename": f.name if f else None}

@app.post("/admin/api/bgnoise/upload")
async def upload_bgnoise(file: UploadFile = File(...), username: str = Depends(verify_admin)):
    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in BGNOISE_EXTENSIONS:
        raise HTTPException(400, "Only .mp3, .wav, .ogg, and .m4a files allowed")
    for old in STATIC_DIR.glob("bg-noise.*"):
        old.unlink()
    dest = STATIC_DIR / f"bg-noise.{ext}"
    content = await file.read()
    dest.write_bytes(content)
    # A fresh upload always starts as not-live — the admin must test it and
    # explicitly re-enable it, so a new track never goes live unreviewed.
    cfg = load_bgnoise_config()
    cfg["enabled"] = False
    save_bgnoise_config(cfg)
    log.info("Background audio uploaded: %s (%d bytes)", dest.name, len(content))
    return {"status": "uploaded", "filename": dest.name}

@app.post("/admin/api/bgnoise")
async def save_bgnoise_route(data: dict, username: str = Depends(verify_admin)):
    try:
        volume = max(0.0, min(1.0, float(data.get("volume", DEFAULT_BGNOISE_CONFIG["volume"]))))
    except (TypeError, ValueError):
        volume = DEFAULT_BGNOISE_CONFIG["volume"]
    enabled = bool(data.get("enabled", False))
    if enabled and not find_bgnoise_file():
        raise HTTPException(400, "No background audio file uploaded yet")
    cfg = {"enabled": enabled, "volume": volume}
    save_bgnoise_config(cfg)
    log.info("Background audio config saved: enabled=%s volume=%.2f", enabled, volume)
    return {"status": "saved", **cfg}

@app.delete("/admin/api/bgnoise")
async def delete_bgnoise(username: str = Depends(verify_admin)):
    for f in STATIC_DIR.glob("bg-noise.*"):
        f.unlink()
    save_bgnoise_config(DEFAULT_BGNOISE_CONFIG.copy())
    log.info("Background audio removed by admin.")
    return {"status": "removed"}

# ── Core helpers ──────────────────────────────────────────────────────────────
def pcm_to_numpy(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

WHISPER_BASE_HINT = (
    "Apex Bank customer service call. Topics include savings accounts, fixed deposits, "
    "home loans, personal loans, credit cards, KYC, net banking, insurance, and mutual funds."
)

MAX_NAME_HINT_COUNT = 25  # Whisper only conditions on roughly the last ~224 tokens of
                          # initial_prompt — keep this well under that budget so nothing
                          # (names or the base instructions) risks being truncated away

# Full customer roster (uncapped), parsed directly from kb_docs — bypasses ChromaDB's
# embedding-similarity search entirely for name lookups. Semantic vector search over
# chunks that each bundle several customers together (300-word chunks ≈ 6-7 customers
# each) is unreliable for exact/short name queries like a bare surname — the correct
# chunk may simply not be among the embedding's top-K results. Since kb_docs follows a
# strict "User Profile: <Name>. ..." format per customer, we can index it exactly.
ALL_CUSTOMER_NAMES: list[str] = []
CUSTOMER_RECORDS: dict[str, dict] = {}  # normalized name -> {"name": ..., "record": ...}

def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())

def refresh_customer_records():
    """(Re)load every customer record from kb_docs/ (.txt/.pdf/.docx) into
    CUSTOMER_RECORDS / ALL_CUSTOMER_NAMES. If the same customer's "User Profile: <Name>"
    block appears in more than one file (e.g. a base profile in one .txt plus loan or
    KYC details added later in a .pdf), the bodies are merged under that one name —
    each tagged with its source file — instead of the later file silently overwriting
    the earlier one. This is what correlates a customer's data across documents."""
    global ALL_CUSTOMER_NAMES, CUSTOMER_RECORDS
    records: dict[str, dict] = {}  # normalized name -> {"name", "sources": [(file, body)]}
    try:
        for doc_path in _kb_files():
            text = extract_document_text(doc_path)
            for m in re.finditer(
                r"User Profile:\s*([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\.\s*(.*?)(?=User Profile:|\Z)",
                text, re.DOTALL
            ):
                name = m.group(1).strip()
                body = m.group(2).strip()
                if not body:
                    continue
                key = _normalize_name(name)
                entry = records.setdefault(key, {"name": name, "sources": []})
                entry["sources"].append((doc_path.name, body))
    except Exception as e:
        log.warning("Could not load customer records from kb_docs: %s", e)

    merged: dict[str, dict] = {}
    for key, entry in records.items():
        name = entry["name"]
        if len(entry["sources"]) == 1:
            body = entry["sources"][0][1]
            record_text = f"User Profile: {name}. {body}"
        else:
            # Multiple docs describe this customer — fold them into one record,
            # each part labelled by its source file so the LLM can see where it came from.
            parts = [f"[{fname}] {body}" for fname, body in entry["sources"]]
            record_text = f"User Profile: {name}. " + " ".join(parts)
        merged[key] = {"name": name, "record": record_text.strip()}

    CUSTOMER_RECORDS = merged
    ALL_CUSTOMER_NAMES = [rec["name"] for rec in merged.values()]

refresh_customer_records()

def build_whisper_name_hint() -> str:
    """Prime Whisper with a sample of real customer names from kb_docs so it recognizes
    names like 'Jasmeet' instead of mishearing them (e.g. as 'Just meet').
    A single hardcoded name (as before) biases the decoder toward that one name
    over all others, so callers with other names get misheard. Only a bounded sample
    is used here — the full roster (ALL_CUSTOMER_NAMES) is matched fuzzily post-transcription."""
    if not ALL_CUSTOMER_NAMES:
        return WHISPER_BASE_HINT
    sample = ALL_CUSTOMER_NAMES[:MAX_NAME_HINT_COUNT]
    return WHISPER_BASE_HINT + " Customer names include: " + ", ".join(sample) + "."

WHISPER_NAME_HINT = build_whisper_name_hint()

async def transcribe(pcm_bytes: bytes) -> str:
    audio = pcm_to_numpy(pcm_bytes)
    loop  = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: stt_model.transcribe(audio, language="en",
                                            fp16=torch.cuda.is_available(),
                                            initial_prompt=WHISPER_NAME_HINT))
    return result["text"].strip()

def retrieve_kb(query: str, top_k: int = None) -> str:
    cfg = load_prompt_config()
    k   = top_k or cfg.get("rag_top_k", RAG_TOP_K)
    kb_filter = cfg.get("kb_filter", "").strip()
    if not kb_collection or not embedder:
        return ""
    try:
        qvec    = embedder.encode([query]).tolist()
        where   = {"source": {"$contains": kb_filter}} if kb_filter else None
        results = kb_collection.query(
            query_embeddings=qvec, n_results=k,
            where=where if where else None
        )
        return "\n\n".join(results.get("documents", [[]])[0])
    except Exception as e:
        log.warning("RAG query failed: %s", e)
        return ""

_CURRENCY_RE = re.compile(r"(?:₹|\bRs\.?)\s*(?=\d)", re.IGNORECASE)

def _normalize_for_speech(text: str) -> str:
    """Kokoro reads 'Rs.'/'₹' literally instead of as a currency word — rewrite
    currency markers to 'Rupees' just for TTS input; the on-screen transcript
    and logs keep the original 'Rs.' text untouched."""
    return _CURRENCY_RE.sub("Rupees ", text)

def synthesize(text: str, voice: str | None = None) -> bytes:
    text = _normalize_for_speech(text)
    chunks = [a for _, _, a in tts_pipeline(text, voice=voice or KOKORO_VOICE) if a is not None]
    if not chunks:
        return b""
    combined = np.clip(np.concatenate(chunks), -1.0, 1.0)
    return (combined * 32767).astype(np.int16).tobytes()

def pcm_to_wav_bytes(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()

def save_wav(pcm: bytes, label: str = "out"):
    ts = int(time.time())
    p  = LOG_DIR / f"{label}_{ts}.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)

def _resample_pcm(pcm_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
    """Linear-interpolation resample of 16-bit mono PCM — good enough for an
    archival call recording (not fed back into STT/TTS, just written to disk)."""
    if from_rate == to_rate or not pcm_bytes:
        return pcm_bytes
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    new_len = int(len(audio) * to_rate / from_rate)
    if new_len <= 0:
        return b""
    x_old = np.linspace(0, 1, num=len(audio))
    x_new = np.linspace(0, 1, num=new_len)
    resampled = np.interp(x_new, x_old, audio)
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()

class CallRecorder:
    """Writes one continuous WAV file per call session, with both the caller's
    microphone audio and the bot's spoken replies appended in chronological
    order as they happen — a single playable recording of the whole call,
    distinct from the existing per-turn greeting_*/reply_*.wav snippets."""
    def __init__(self, path: Path, rate: int = SAMPLE_RATE):
        self.rate = rate
        self._wf = wave.open(str(path), "wb")
        self._wf.setnchannels(1)
        self._wf.setsampwidth(2)
        self._wf.setframerate(rate)
        self._closed = False

    def write(self, pcm: bytes, source_rate: int):
        if self._closed or not pcm:
            return
        try:
            self._wf.writeframes(_resample_pcm(pcm, source_rate, self.rate))
        except Exception as e:
            log.warning("Call recorder write failed: %s", e)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._wf.close()
        except Exception as e:
            log.warning("Call recorder close failed: %s", e)

# ── Pluggable LLM / TTS providers (local ↔ cloud, admin-configurable) ──────────
# Kept in a separate, git-ignored file since cloud mode stores API keys — unlike
# runtime_config.json (voice/model name only, no secrets), this must never be
# committed. "local" mode is untouched: it still just uses KOKORO_VOICE/
# OLLAMA_MODEL exactly as before this feature existed.
PROVIDER_FILE = BASE_DIR / "provider_config.json"
DEFAULT_PROVIDER_CONFIG = {
    "llm_mode": "local",   # "local" | "cloud"
    "llm_cloud": {"base_url": "https://api.openai.com/v1", "api_key": "", "model": ""},
    "tts_mode": "local",   # "local" | "cloud"
    "tts_cloud_engine": "elevenlabs",   # "elevenlabs" | "openai"
    "tts_cloud": {
        "elevenlabs": {"api_key": "", "voice_id": ""},
        "openai":     {"api_key": "", "voice": "alloy", "base_url": "https://api.openai.com/v1"},
    },
}

def load_provider_config() -> dict:
    cfg = json.loads(json.dumps(DEFAULT_PROVIDER_CONFIG))  # deep copy of defaults
    if PROVIDER_FILE.exists():
        try:
            saved = json.loads(PROVIDER_FILE.read_text(encoding="utf-8"))
            cfg["llm_mode"] = saved.get("llm_mode", cfg["llm_mode"])
            cfg["llm_cloud"].update(saved.get("llm_cloud", {}))
            cfg["tts_mode"] = saved.get("tts_mode", cfg["tts_mode"])
            cfg["tts_cloud_engine"] = saved.get("tts_cloud_engine", cfg["tts_cloud_engine"])
            for engine, vals in saved.get("tts_cloud", {}).items():
                cfg["tts_cloud"].setdefault(engine, {}).update(vals)
        except Exception as e:
            log.warning("Could not load provider_config.json: %s", e)
    return cfg

def save_provider_config(cfg: dict):
    PROVIDER_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

async def generate_llm_cloud(prompt: str, stop: list[str] | None, cfg: dict) -> str:
    """Generic OpenAI-compatible chat-completions call — works with OpenAI, Azure
    OpenAI, Groq, OpenRouter, Together.ai, etc. by pointing base_url at them."""
    base_url = (cfg.get("base_url") or "https://api.openai.com/v1").rstrip("/")
    payload = {
        "model": cfg.get("model", ""),
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if stop:
        payload["stop"] = stop
    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {cfg.get('api_key', '')}"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

async def generate_llm_reply(prompt: str, stop: list[str] | None = None) -> str:
    """Dispatches to the active LLM provider. Raises on failure — callers already
    catch and fall back to the configured fallback_message, unchanged from before
    this dispatcher existed. stop is optional — callers outside the turn-based
    conversation flow (e.g. log analysis) don't need a stop sequence."""
    cfg = load_provider_config()
    if cfg["llm_mode"] == "cloud":
        return await generate_llm_cloud(prompt, stop, cfg["llm_cloud"])
    options = {"stop": stop} if stop else {}
    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                  "options": options},
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

def synthesize_elevenlabs(text: str, cfg: dict) -> bytes:
    text = _normalize_for_speech(text)
    voice_id = cfg.get("voice_id", "")
    resp = httpx.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        params={"output_format": "pcm_24000"},
        headers={"xi-api-key": cfg.get("api_key", ""), "Content-Type": "application/json"},
        json={"text": text},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.content

def synthesize_openai_tts(text: str, cfg: dict) -> bytes:
    text = _normalize_for_speech(text)
    base_url = (cfg.get("base_url") or "https://api.openai.com/v1").rstrip("/")
    resp = httpx.post(
        f"{base_url}/audio/speech",
        headers={"Authorization": f"Bearer {cfg.get('api_key', '')}"},
        json={"model": "tts-1", "input": text, "voice": cfg.get("voice", "alloy"),
              "response_format": "pcm"},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.content

_TTS_CLOUD_ENGINES = {"elevenlabs": synthesize_elevenlabs, "openai": synthesize_openai_tts}

async def synthesize_active(text: str) -> bytes:
    """Dispatches to the active TTS provider — local Kokoro (existing synthesize(),
    unchanged) or the configured cloud engine. Runs the blocking call in a thread,
    same as how synthesize() was already invoked via run_in_executor before this
    dispatcher existed."""
    cfg = load_provider_config()
    loop = asyncio.get_event_loop()
    if cfg["tts_mode"] == "cloud":
        engine = cfg.get("tts_cloud_engine", "elevenlabs")
        fn = _TTS_CLOUD_ENGINES.get(engine, synthesize_elevenlabs)
        engine_cfg = cfg["tts_cloud"].get(engine, {})
        return await loop.run_in_executor(None, fn, text, engine_cfg)
    return await loop.run_in_executor(None, synthesize, text)

# ── WebSocket ─────────────────────────────────────────────────────────────────
# ── Session state helper ─────────────────────────────────────────────────────
def make_session():
    """Fresh session state for each WebSocket connection."""
    return {
        "stage":         "ask_name",  # ask_name | ask_pin | verified
        "customer_name": None,        # name of the account holder currently being looked up
        "caller_name":   None,        # the actual caller's own name (only set on self-identification)
        "spoken_name":   None,        # name the caller gave us (may not match KB)
        "customer_id":   None,
        "pin_attempts":  0,
        "name_attempts": 0,
        "customer_data": None,
        "on_behalf":     False,       # True while looking up an account for someone other than the caller
        "verified_since": None,       # index into `conversation` where the verified stage began
    }

import re as _re

def fuzzy_match_customer_name(spoken: str, cutoff: float = 0.72) -> str | None:
    """Match a possibly mis-transcribed name against the FULL customer roster
    (ALL_CUSTOMER_NAMES, unbounded — not the small Whisper-hint sample), so phonetic
    STT slips (e.g. 'Just meet Chauhan' for 'Jasmeet Chauhan') still resolve correctly."""
    norm_spoken = _re.sub(r"[^a-z]", "", spoken.lower())
    if not norm_spoken or not ALL_CUSTOMER_NAMES:
        return None
    best_name, best_ratio = None, 0.0
    for name in ALL_CUSTOMER_NAMES:
        norm_name = _re.sub(r"[^a-z]", "", name.lower())
        ratio = difflib.SequenceMatcher(None, norm_spoken, norm_name).ratio()
        if ratio > best_ratio:
            best_ratio, best_name = ratio, name
    return best_name if best_ratio >= cutoff else None

def find_customer(name_or_id: str, _fuzzy_tried: bool = False) -> dict | None:
    """Find a customer record matching name or ID.

    Checks the exact/fuzzy CUSTOMER_RECORDS index first (deterministic — parsed
    directly from kb_docs, not dependent on embedding similarity), then falls back
    to ChromaDB semantic search for anything not in that structured format."""
    needle_words = [w for w in name_or_id.lower().split() if len(w) > 2]

    # 1) Deterministic lookup against the full customer roster.
    if needle_words and CUSTOMER_RECORDS:
        for rec in CUSTOMER_RECORDS.values():
            name_lower = rec["name"].lower()
            if all(w in name_lower for w in needle_words):
                log.info("Customer matched (roster): %s → %s", name_or_id, rec["name"])
                return {"record": rec["record"], "name": rec["name"]}
        if not _fuzzy_tried:
            corrected = fuzzy_match_customer_name(name_or_id)
            if corrected:
                rec = CUSTOMER_RECORDS.get(_normalize_name(corrected))
                if rec:
                    log.info("Customer matched (fuzzy): %r → %s", name_or_id, rec["name"])
                    return {"record": rec["record"], "name": rec["name"]}

    # 2) Fallback: ChromaDB embedding search (covers kb_docs content that doesn't
    #    follow the "User Profile: Name." format, e.g. custom admin-uploaded files).
    if not kb_collection or not embedder:
        return None
    try:
        candidates = [name_or_id] + name_or_id.split()
        seen_docs: set[str] = set()
        all_docs: list[str] = []
        for candidate in candidates:
            qvec = embedder.encode([candidate]).tolist()
            results = kb_collection.query(query_embeddings=qvec, n_results=5)
            for doc in results.get("documents", [[]])[0]:
                if doc not in seen_docs:
                    seen_docs.add(doc)
                    all_docs.append(doc)

        for doc in all_docs:
            doc_lower = doc.lower()
            # Require ALL name words to appear in the doc (prevents wrong-surname matches)
            if needle_words and all(w in doc_lower for w in needle_words):
                m = _re.search(r"(?:User Profile|Customer Name|Name)\s*[:\-]\s*([A-Za-z]+(?:\s+[A-Za-z]+)?)", doc, _re.IGNORECASE)
                real_name = m.group(1).strip() if m else name_or_id.strip().title()
                log.info("Customer matched (chromadb): %s → %s", name_or_id, real_name)
                return {"record": doc, "name": real_name}

        # Log the top doc snippet to help debug format mismatches
        if all_docs:
            log.info("No customer match for: %s | top doc snippet: %.120s", name_or_id, all_docs[0])
        else:
            log.info("No customer match for: %s | no docs returned", name_or_id)
    except Exception as e:
        log.warning("Customer lookup failed: %s", e)
    return None

def check_pin(customer_data: dict, pin: str) -> bool:
    """Check PIN against customer record."""
    record = customer_data.get("record", "")
    # Look for pattern: PIN: XXXX or Phone Banking PIN: XXXX
    import re
    match = re.search(r"PIN\D*?(\d{4})", record, re.IGNORECASE)
    if match:
        return match.group(1).strip() == pin.strip()
    return False

def first_name_of(name: str | None) -> str:
    """First word of a name, Title Cased — "" if name is None/empty (never crashes
    on .split()[0] like the bare pattern does when the string has no words)."""
    words = (name or "").split()
    return words[0].title() if words else ""

_INTENT_SCHEMA_HINT = (
    'Respond with ONLY a single-line JSON object — no prose, no markdown fences. '
    'Schema: {"intent": one of '
    '["greeting","self_name","other_name","unclear"], '
    '"name": full name if the caller stated a person\'s name (their own, or someone '
    'else\'s), in Title Case, else null}'
)

def _regex_intent_fallback(transcript: str) -> dict:
    """Best-effort stand-in used ONLY when Ollama itself is unreachable (connection
    error/timeout), so an LLM outage doesn't take identification down completely.
    Deliberately simpler than the old regex pipeline — this is a safety net, not
    the primary path."""
    t = transcript.strip()
    if _re.fullmatch(r"(hi+|hey+|hello+|howdy|good\s*(morning|afternoon|evening))[.!?,\s]*", t, _re.IGNORECASE):
        return {"intent": "greeting", "name": None}
    m = _re.search(r"\bfor\s+(?:my\s+\w+[, ]+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", transcript)
    if m:
        return {"intent": "other_name", "name": m.group(1).strip()}
    if _re.search(r"\b(on behalf of|for someone|for somebody|checking for|account for)\b", transcript, _re.IGNORECASE):
        return {"intent": "other_name", "name": None}
    m = _re.search(r"(?:my name is|i(?:'m| am)|this is)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)", transcript, _re.IGNORECASE)
    if m:
        return {"intent": "self_name", "name": m.group(1).strip()}
    m = _re.fullmatch(r"([A-Za-z]+(?:\s+[A-Za-z]+){0,2})[.!?,]*", t)
    if m and len(m.group(1)) >= 3:
        return {"intent": "self_name", "name": m.group(1).strip()}
    return {"intent": "unclear", "name": None}

async def extract_call_intent(transcript: str, stage: str) -> dict:
    """Use the local LLM to understand what the caller said, instead of brittle regex
    pattern matching. This is the natural-language-understanding layer for the identify
    stage: it classifies intent and extracts a name (correcting for speech-to-text
    misheard names using the known customer roster) in one call.
    If Ollama can't be reached at all (connection error/timeout), falls back to a
    lightweight regex parser rather than going silent. If Ollama responds but is
    genuinely unsure, that {'intent': 'unclear'} verdict is trusted as-is."""
    stage_context = {
        "ask_name": "The assistant just asked the caller for their full name or Customer ID.",
    }.get(stage, "")
    roster_hint = ""
    if ALL_CUSTOMER_NAMES:
        roster_hint = (
            "\nKnown customer names — if the caller's stated name sounds like a "
            "speech-to-text mis-transcription of one of these (e.g. 'Just meet Chauhan' "
            f"sounds like 'Jasmeet Chauhan'), correct it to the closest match: "
            f"{', '.join(ALL_CUSTOMER_NAMES)}"
        )
    prompt = (
        f"You are the natural-language-understanding layer for an Apex Bank phone call. "
        f"{stage_context}\n"
        f'Caller said: "{transcript}"'
        f"{roster_hint}\n\n"
        f"{_INTENT_SCHEMA_HINT}"
    )
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"}
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        if isinstance(data, dict) and data.get("intent"):
            return {
                "intent": str(data.get("intent") or "unclear").strip().lower(),
                "name":   (str(data.get("name")).strip() if data.get("name") else None),
            }
    except (httpx.HTTPError, httpx.TimeoutException) as e:
        log.warning("Intent extraction: Ollama unreachable (stage=%s): %s — using regex fallback", stage, e)
        return _regex_intent_fallback(transcript)
    except Exception as e:
        log.warning("Intent extraction: bad response (stage=%s): %s", stage, e)
    return {"intent": "unclear", "name": None}


INACTIVITY_PROMPT_SECS = 20   # seconds of silence before each "still here?" nudge

@app.websocket("/ws/voice")
async def voice_ws(ws: WebSocket):
    await ws.accept()
    log.info("Client connected")

    call_recorder = CallRecorder(LOG_DIR / f"call_{int(time.time())}.wav")

    cfg        = load_prompt_config()
    fallback   = cfg.get("fallback_message", DEFAULT_PROMPT_CONFIG["fallback_message"])
    conversation: list[dict] = []
    session    = make_session()
    processing = asyncio.Lock()
    loop       = asyncio.get_event_loop()

    # ── Inactivity tracking ───────────────────────────────────────────────────
    last_activity   = time.time()
    nudge_count     = 0          # how many "still there?" prompts sent this silence window
    session_closed  = asyncio.Event()
    nudges_disabled = False      # set by the client's "Stop waiting" control — suppresses
                                  # "Are you still there?" nudges and the auto-hangup that follows them

    # ── Barge-in state ────────────────────────────────────────────────────────
    interrupted       = asyncio.Event()   # set when user speaks over bot
    barge_in_pending  = None              # raw PCM bytes queued during processing

    def touch(reset_nudges: bool = True):
        nonlocal last_activity, nudge_count
        last_activity = time.time()
        if reset_nudges:
            nudge_count = 0   # reset only when customer actually speaks

    async def say(text: str, msg_type: str = "reply", _nudge: bool = False):
        # Don't try to send on a socket that's already closed/closing
        if session_closed.is_set() or ws.client_state != WebSocketState.CONNECTED:
            return
        try:
            log.info("🤖 BOT  said: %s", text)
            await ws.send_json({"type": msg_type, "text": text})
            ts = time.time()
            pcm = await synthesize_active(text)
            save_wav(pcm, msg_type)
            call_recorder.write(pcm, source_rate=SAMPLE_RATE)
            if interrupted.is_set():
                log.info("STEP 6 ▶ TTS done (%.2fs) but BARGE-IN active — skipping audio playback", time.time() - ts)
                await ws.send_json({"type": "barge_in_ack"})
            else:
                log.info("STEP 6 ▶ TTS synthesized (%.2fs): %.1f KB → sending to client", time.time() - ts, len(pcm) / 1024)
                await ws.send_bytes(pcm)
            await ws.send_json({"type": "turn_complete"})
        except (WebSocketDisconnect, RuntimeError):
            # Client went away mid-reply — stop the session cleanly
            session_closed.set()
            return
        conversation.append({"role": "assistant", "content": text})
        # After a nudge, restart the silence clock but keep the nudge count
        touch(reset_nudges=not _nudge)

    async def end_session(reason: str):
        """Server-initiated goodbye (farewell phrase, inactivity nudges exhausted, too many
        failed PIN attempts). Tells the client this close is expected *before* closing the
        socket, so the client doesn't mistake it for a dropped connection and auto-reconnect
        — which used to re-greet the caller seconds after they'd just said "that's all"."""
        if session_closed.is_set() or ws.client_state != WebSocketState.CONNECTED:
            return
        try:
            await ws.send_json({"type": "session_ended", "reason": reason})
        except (WebSocketDisconnect, RuntimeError):
            pass
        session_closed.set()
        try:
            await ws.close()
        except RuntimeError:
            pass

    async def inactivity_watcher():
        """Background task: nudge up to 3 times, then close the session."""
        nonlocal nudge_count
        MAX_NUDGES = 3
        nudge_messages = [
            # nudge 1
            {
                "ask_name": "Are you still there? May I please have your name or Customer ID?",
                "ask_pin":  "Are you still there? Please say your 4-digit phone banking PIN when ready.",
                "verified": "Are you still there? Please go ahead — I'm listening.",
            },
            # nudge 2
            {
                "ask_name": "I'm still here and happy to help. Could you please tell me your name or Customer ID?",
                "ask_pin":  "I'm still waiting for your PIN. Please say all four digits when you're ready.",
                "verified": "I'm still here whenever you're ready. What can I help you with?",
            },
            # nudge 3 (final)
            {
                "ask_name": "This is my last attempt — I didn't catch your name. Please call back when you're ready.",
                "ask_pin":  "I wasn't able to verify your PIN. For security, please call back when you're ready.",
                "verified": "I haven't heard from you for a while. I'll be closing this session now.",
            },
        ]

        while not session_closed.is_set():
            await asyncio.sleep(5)
            if nudges_disabled:
                continue  # user asked the bot to stop waiting on them — no nudges, no auto-hangup
            idle = time.time() - last_activity
            if processing.locked():
                continue  # bot is working — don't count that as customer silence
            # A verified customer gets a much longer window — they may pause to think
            interval = 45 if session.get("stage") == "verified" else INACTIVITY_PROMPT_SECS
            if idle >= interval and nudge_count < MAX_NUDGES:
                stage = session["stage"]
                stage_key = stage if stage in ("ask_name", "ask_pin") else "verified"
                prompt = nudge_messages[nudge_count][stage_key]
                nudge_count += 1
                log.info("Inactivity nudge %d/%d sent (idle %.0fs)", nudge_count, MAX_NUDGES, idle)
                await say(prompt, "reply", _nudge=True)

                if nudge_count >= MAX_NUDGES:
                    # All nudges exhausted — wait one more interval then close
                    await asyncio.sleep(interval)
                    if not session_closed.is_set():
                        log.info("Session closed after %d nudges with no response.", MAX_NUDGES)
                        await say(
                            "We haven't been able to reach you after several attempts. "
                            "Your session has now ended for security. "
                            "Thank you for calling Apex Bank — please call us back when you're ready. Goodbye.",
                            "reply", _nudge=True
                        )
                        await end_session("inactivity_timeout")
                        return

    watcher_task = asyncio.create_task(inactivity_watcher())

    # ── Greeting ──────────────────────────────────────────────────────────────
    greeting = "Hi there, thanks for calling Apex Bank! I'm your AI assistant. Could I get your name or Customer ID to get started?"
    await ws.send_json({"type": "status", "msg": "Preparing greeting..."})
    pcm = await synthesize_active(greeting)
    save_wav(pcm, "greeting")
    call_recorder.write(pcm, source_rate=SAMPLE_RATE)
    await ws.send_bytes(pcm)
    await ws.send_json({"type": "turn_complete"})
    conversation.append({"role": "assistant", "content": greeting})

    async def process_audio(raw: bytes):
        """Transcribe + stage machine + reply. Must be called inside the processing lock."""
        nonlocal barge_in_pending
        interrupted.clear()
        call_recorder.write(raw, source_rate=16000)  # caller's mic audio, always 16kHz over the wire
        kb = len(raw) / 1024
        log.info("STEP 1 ▶ Audio received from client: %.1f KB (%d bytes)", kb, len(raw))
        await ws.send_json({"type": "status", "msg": f"Received {kb:.1f} KB — transcribing..."})
        t0 = time.time()
        transcript = await transcribe(raw)
        dt = time.time() - t0
        if not transcript:
            log.info("STEP 2 ▶ Whisper returned EMPTY transcript (%.2fs) — likely silence/too short", dt)
            await ws.send_json({"type": "status", "msg": "Couldn't hear clearly — please try again."})
            await ws.send_json({"type": "turn_complete"})
            return
        log.info("STEP 2 ▶ Whisper transcript (%.2fs): %r [stage=%s]", dt, transcript, session.get("stage"))
        # Redact any 4-digit sequences from the transcript log when in PIN stage
        log_transcript = _re.sub(r'\b\d{4}\b', '****', transcript) if session.get("stage") == "ask_pin" else transcript
        log.info("👤 USER said: %s", log_transcript)
        await ws.send_json({"type": "transcript", "text": log_transcript})
        conversation.append({"role": "user", "content": transcript})

        # ── Farewell detection (runs before stage machine) ────────────────────
        _farewell = _re.search(
            r"\b(bye|goodbye|good ?bye|see you|take care|that'?s? ?(it|all)|"
            r"thank(s| you)( so much| very much)?|cheers|have a (good|great|nice) (day|one)|"
            r"no (more )?questions?|i('m| am) (done|good|all set|okay now)|all good|"
            r"nothing else|that will be all|end (the )?(call|session))\b",
            transcript, _re.IGNORECASE
        )
        if _farewell:
            name = (session.get("caller_name") or "").split()
            first = name[0].title() if name else ""
            farewell_reply = (
                f"It was a pleasure helping you{', ' + first if first else ''}! "
                f"Have a wonderful day. Goodbye!"
            )
            await say(farewell_reply)
            log.info("Farewell detected — closing session.")
            await asyncio.sleep(0.5)
            await end_session("farewell")
            return

        stage = session["stage"]

        if stage == "ask_name":
            await ws.send_json({"type": "status", "msg": "Looking up customer..."})

            # Understand what the caller said via the local LLM, instead of trying to
            # anticipate every phrasing with regex (fixes: names buried mid-sentence,
            # "for my wife, X" style on-behalf requests, and mis-transcribed names —
            # the roster hint lets the model correct STT slips like "Just meet Chauhan").
            intent_data = await extract_call_intent(transcript, "ask_name")
            intent = intent_data["intent"]
            name   = intent_data["name"]
            on_behalf = intent == "other_name"

            if intent == "greeting" and not name:
                await say("Hey! Good to hear from you. Could you share your full name or Customer ID so I can pull up your account?")
                return

            if intent == "other_name" and not name:
                # They've indicated it's on behalf of someone else, but no name yet.
                session["on_behalf"] = True
                await say("Of course! Could you give me the full name or Customer ID of the account holder you'd like to check?")
                return

            if intent not in ("self_name", "other_name") or not (name and name.strip()):
                # Nothing usable extracted — ask again naturally
                await say("I didn't quite catch your name. Could you tell me your full name or Customer ID?")
                return

            lookup_text = name.strip()
            session["on_behalf"] = on_behalf
            spoken_first = first_name_of(lookup_text)
            session["spoken_name"] = spoken_first
            session["name_attempts"] = session.get("name_attempts", 0) + 1

            only_first_name = len(lookup_text.split()) == 1

            # Always ask for last name if only one word given — don't attempt a DB lookup yet
            if only_first_name:
                if on_behalf:
                    reply = (f"Got it — I just need {spoken_first}'s last name too, "
                             f"could you give me the full name so I can find the account?")
                else:
                    reply = (f"Nice to meet you, {spoken_first}! I just need your last name too — "
                             f"could you give me your full name so I can find your account?")
                await say(reply)
                return

            customer = find_customer(lookup_text)
            if customer:
                session["customer_name"] = customer.get("name", lookup_text)
                if not on_behalf:
                    session["caller_name"] = session["customer_name"]
                session["on_behalf"]     = False
                session["customer_data"] = customer
                session["stage"]         = "ask_pin"
                session["name_attempts"] = 0
                first = first_name_of(session["customer_name"])
                if on_behalf:
                    reply = (f"Got it — I found {first}'s account. "
                             f"For security, could you provide their 4-digit phone banking PIN?")
                else:
                    reply = (f"Got it, {first} — nice to have you with us. "
                             f"Just a quick security check: could you tell me your 4-digit phone banking PIN?")
            else:
                attempts = session["name_attempts"]
                addr = "" if on_behalf else f", {spoken_first}"
                if attempts == 1:
                    reply = (f"Thanks{addr}. I wasn't able to find an account under that name. "
                             f"Could you double-check the spelling, or share {'the' if on_behalf else 'your'} Customer ID if you have it handy?")
                elif attempts == 2:
                    reply = (f"I'm still not finding a match{addr}. "
                             f"{'The' if on_behalf else 'Your'} Customer ID would help me locate "
                             f"{'them' if on_behalf else 'you'} right away — it's usually on the bank card or welcome letter.")
                else:
                    reply = (f"I'm sorry{addr}, I haven't been able to locate an account with those details. "
                             f"Please call us on 1800-123-4567 or visit your nearest branch and we'll get you sorted. Thanks for calling Apex Bank.")
            await say(reply)

        elif stage == "ask_pin":
            digits = _re.sub(r"\D", "", transcript)
            # Only address the caller by name if they've identified themselves — never by the
            # (possibly different) account holder's name when looking up someone else's account.
            caller_first = first_name_of(session.get("caller_name"))
            addr = f", {caller_first}" if caller_first else ""
            close_after = False
            if len(digits) >= 4:
                pin = digits[:4]
                if check_pin(session["customer_data"], pin):
                    session["stage"]        = "verified"
                    session["pin_attempts"] = 0
                    session["verified_since"] = len(conversation)  # index of the first post-verification turn
                    reply = (f"PIN verified — you're all set{addr}. "
                             f"How can I help you today?")
                else:
                    session["pin_attempts"] += 1
                    if session["pin_attempts"] >= 3:
                        reply = (f"I'm sorry{addr}, we've had three unsuccessful PIN attempts. "
                                 f"For your security, please reset your PIN through the Apex Bank app, "
                                 f"call us back on 1800-123-4567, or visit your nearest branch. "
                                 f"Thanks for calling — take care.")
                        session["stage"] = "ask_name"
                        session["customer_data"] = None
                        close_after = True
                    else:
                        remaining = 3 - session["pin_attempts"]
                        reply = (f"That PIN didn't match{addr}. "
                                 f"You have {remaining} tr{'y' if remaining == 1 else 'ies'} remaining — please try again.")
            else:
                reply = f"I just need your 4-digit phone banking PIN{addr}. Go ahead whenever you're ready."
            # Speak the reply BEFORE marking the session closed — end_session() below sets
            # session_closed, and say() refuses to send anything once that flag is set, so
            # the order here matters: otherwise this goodbye would be silently dropped.
            await say(reply)
            if close_after:
                await end_session("pin_attempts_exceeded")

        elif stage == "verified":
            # ── Switch-account intent: "check my wife's / sister's / father's account" ──
            _RELATIONS = (
                r"wife|husband|partner|spouse|girlfriend|boyfriend|"
                r"sister|brother|sibling|"
                r"mother|mom|mum|father|dad|"
                r"son|daughter|child|kid|"
                r"grandfather|grandmother|grandpa|grandma|gran|"
                r"uncle|aunt|nephew|niece|cousin|"
                r"friend|colleague|associate|"
                r"somebody else|someone else|another person|another account|different account"
            )
            _other_match = _re.search(
                rf"\b(my\s+)?({_RELATIONS})('?s)?\b",
                transcript, _re.IGNORECASE
            )
            # Also check "for [Name]" — caller directly names the person
            _for_name_v = _re.search(r"\bfor\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", transcript)

            if _for_name_v:
                # Name provided inline — look it up and start PIN flow for them
                session["on_behalf"] = True
                lookup_text  = _for_name_v.group(1).strip()
                spoken_first = lookup_text.split()[0].title()
                if len(lookup_text.split()) == 1:
                    await say(f"Sure! Could I also get {spoken_first}'s last name to find the right account?")
                    session["stage"] = "ask_name"
                    session["spoken_name"] = spoken_first
                    return
                customer = find_customer(lookup_text)
                if customer:
                    session["customer_name"] = customer.get("name", lookup_text)
                    session["customer_data"] = customer
                    session["stage"]         = "ask_pin"
                    session["pin_attempts"]  = 0
                    first = session["customer_name"].split()[0].title()
                    await say(f"Found {first}'s account. Could you provide their 4-digit phone banking PIN to verify?")
                else:
                    await say(f"I wasn't able to find an account for {lookup_text}. Could you double-check the full name or share their Customer ID?")
                return
            elif _other_match:
                relation = _other_match.group(2).lower()
                relation_display = relation if relation not in ("somebody else","someone else","another person","another account","different account") else "that person"
                await say(f"Of course! Could you give me your {relation_display}'s full name so I can look up their account?")
                session["stage"] = "ask_name"
                session["customer_name"] = None
                session["customer_data"] = None
                session["pin_attempts"]  = 0
                session["name_attempts"] = 0
                session["on_behalf"]     = True
                return

            log.info("STEP 3 ▶ Stage=verified — building LLM prompt + RAG context")
            await ws.send_json({"type": "status", "msg": "Thinking..."})
            # Strip PIN from customer record before sending to LLM
            raw_record = session["customer_data"].get("record", "")
            customer_context = _re.sub(r"(?:Phone Banking )?PIN\s*[:\-]\s*\d{4}", "[PIN REDACTED]", raw_record, flags=_re.IGNORECASE)
            kb_context       = retrieve_kb(transcript)
            combined_context = f"VERIFIED CUSTOMER RECORD:\n{customer_context}"
            if kb_context:
                combined_context += f"\n\nBANK PRODUCT INFORMATION:\n{kb_context}"

            first_name  = first_name_of(session["customer_name"])
            caller_first = first_name_of(session.get("caller_name"))
            sys_prompt = cfg.get("system_prompt", DEFAULT_PROMPT_CONFIG["system_prompt"])
            guardrails = cfg.get("guardrails", [])
            if guardrails:
                sys_prompt += "\n\nGUARDRAILS:\n" + "\n".join(f"- {g}" for g in guardrails)
            if caller_first and caller_first != first_name:
                identity_note = (
                    f"You are speaking with {caller_first}, who is asking about {first_name}'s account. "
                    f"That account is ALREADY verified. Address the caller as {caller_first} — "
                    f"never call them {first_name}, that is the account holder's name, not the caller's."
                )
            else:
                identity_note = f"The customer ({first_name or caller_first}) is ALREADY verified."
            sys_prompt += (
                f"\n\nCONTEXT (verified customer data):\n{combined_context}"
                f"\n\nIMPORTANT: {identity_note} "
                f"Do NOT repeat the greeting, ask for their name or PIN again, mention the PIN, "
                f"or explain any verification logic. Never reveal or repeat any PIN digits. "
                f"Answer their question directly and naturally using only the customer data above. "
                f"Always write currency amounts as 'Rupees X' — never use 'Rs.' or 'INR' as text-to-speech will mispronounce them. "
                f"Never start or end a response with 'Thank you for calling', 'Thank you for contacting', or any similar phrase — just answer the question."
            )

            # History = every turn from the moment this customer was verified, up to (but
            # excluding) the current question — that current question is already the last
            # entry in `conversation` (appended at line ~1090) and gets added explicitly
            # below, so including it here too would duplicate it as two consecutive
            # "Customer:" lines with no reply in between, which reliably confused the model
            # into narrating fake stage directions about "the second request" instead of
            # just answering (seen live in production logs).
            verified_since = session.get("verified_since") or 0
            verified_turns = conversation[verified_since:-1]
            history = ""
            for m in verified_turns[-6:]:
                role    = "Customer" if m["role"] == "user" else "Assistant"
                # Redact any 4-digit sequences from history so LLM never sees the PIN
                content = _re.sub(r'\b\d{4}\b', '[PIN]', m["content"])
                history += f"{role}: {content}\n"
            prompt = f"{sys_prompt}\n\nConversation:\n{history}Customer: {transcript}\nAssistant:"

            _llm_mode = load_provider_config()["llm_mode"]
            log.info("STEP 4 ▶ Sending prompt to LLM (%s)...", "cloud" if _llm_mode == "cloud" else OLLAMA_MODEL)
            tl = time.time()
            try:
                # Without a stop sequence the model sometimes keeps going past its own
                # answer and hallucinates further fake "Customer:"/"Assistant:" turns
                # (seen live: a reply containing "(No answer yet)... (After the second
                # request)..." stage directions read aloud verbatim by TTS). Cut
                # generation the moment it tries to start a new turn.
                reply = await generate_llm_reply(prompt, ["\nCustomer:", "\nAssistant:", "\nUser:"])
            except Exception as e:
                log.error("STEP 4 ▶ LLM error: %s", e)
                reply = fallback
            # Defense in depth: if the model still slipped a fake next turn past the stop
            # sequence, trim it — only the first turn is ever a real answer to this question.
            reply = _re.split(r"\n\s*(?:Customer|Assistant|User)\s*:", reply)[0].strip()
            if not reply:
                reply = fallback
            log.info("STEP 5 ▶ LLM reply (%.2fs): %s", time.time() - tl, reply)
            await say(reply)

    try:
        while not session_closed.is_set():
            msg = await ws.receive()

            # Client closed the socket — exit the loop cleanly
            if msg.get("type") == "websocket.disconnect":
                log.info("Client disconnected (received disconnect message)")
                break

            if "bytes" in msg and msg["bytes"]:
                touch()   # reset inactivity timer on every audio chunk received
                log.info("STEP 0 ▶ WS received %d bytes from client (processing busy=%s)",
                         len(msg["bytes"]), processing.locked())
                if processing.locked():
                    # Barge-in: queue this audio, signal the current turn to skip playback
                    nonlocal_ref = msg["bytes"]
                    barge_in_pending = nonlocal_ref
                    interrupted.set()
                    log.info("Barge-in: queuing new audio, signalling interrupt")
                    await ws.send_json({"type": "status", "msg": "Got it — I'll respond shortly."})
                    continue

                async with processing:
                    await process_audio(msg["bytes"])
                    # After processing, handle any barge-in audio that arrived during the turn
                    if barge_in_pending:
                        pending = barge_in_pending
                        barge_in_pending = None
                        log.info("Processing queued barge-in audio")
                        await process_audio(pending)

            elif "text" in msg and msg["text"]:
                data = json.loads(msg["text"])
                if data.get("type") == "reset":
                    conversation.clear()
                    session = make_session()
                    touch()
                    await ws.send_json({"type": "status", "msg": "Session reset."})
                    reset_greeting = "No problem — let's start fresh. Could I get your name or Customer ID?"
                    await say(reset_greeting, "reply")
                elif data.get("type") == "barge_in":
                    # Client stopped audio playback — signal server to skip remaining PCM
                    interrupted.set()
                    log.info("Barge-in signal received from client")
                elif data.get("type") == "stop_waiting":
                    nudges_disabled = bool(data.get("enabled", True))
                    touch()
                    log.info("Inactivity nudges %s by client", "disabled" if nudges_disabled else "re-enabled")
                    await ws.send_json({
                        "type": "status",
                        "msg": "Take your time — I won't nudge you." if nudges_disabled
                               else "Inactivity check-ins re-enabled."
                    })

    except WebSocketDisconnect:
        log.info("Client disconnected")
    except Exception as e:
        log.error("WS error: %s", e, exc_info=True)
    finally:
        session_closed.set()
        call_recorder.close()
        watcher_task.cancel()
        try:
            await watcher_task
        except asyncio.CancelledError:
            pass
