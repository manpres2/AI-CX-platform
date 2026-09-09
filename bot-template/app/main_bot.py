"""
main_bot.py - Generic Local AI Voice Bot template, with Admin Panel

Template source for the "Create New Bot" feature: a superadmin fills a form
on the launcher, which copies this whole folder into bots/<slug>/, writes a
fresh branding.json/prompt_config.json from the form, and starts it on its
own port. Same local voice stack as the platform's other bots (Whisper STT +
ChromaDB RAG + Ollama LLaMA + Kokoro TTS), but with no baked-in business
domain — the admin configures branding/prompt/knowledge-base after creation,
exactly like the platform's existing bots.

FastAPI WebSocket: Whisper STT + ChromaDB RAG + Ollama LLaMA + Kokoro TTS
Admin panel at: http://localhost:<port>/admin  (login required)
Run from app/ folder: uvicorn main_bot:app --host 0.0.0.0 --port <port>
"""

import asyncio
import io
import json
import logging
import math
import os
import platform
import re
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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, Response
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
# .resolve() anchors these to this file's actual location regardless of the
# process's current working directory — under `uvicorn --reload`, the worker
# subprocess can resolve __file__ relative to a different cwd than expected,
# which silently broke these paths (KB/branding/bgnoise files "not found")
# until this was added.
_this_file = Path(__file__).resolve()

# .env lives at the repo root (C:\AIManpres2\.env). Created bots live nested
# one level deeper than the platform's other apps (C:\AIManpres2\bots\<slug>\
# vs e.g. C:\AIManpres2\techsupport-voice-bot\), so REPO_ROOT needs one more
# .parent than BASE_DIR below, not the two the other apps use.
BASE_DIR     = _this_file.parent.parent
REPO_ROOT    = BASE_DIR.parent.parent
_env_path = REPO_ROOT / ".env"
if _env_path.exists():
    try:
        load_dotenv(_env_path, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(_env_path, encoding="utf-16")

STATIC_DIR   = BASE_DIR / "static_bot"
LOG_DIR      = BASE_DIR / "logs"
KB_DOCS      = BASE_DIR / "kb_docs"
KB_STORE     = BASE_DIR / "kb_store"
PROMPT_FILE  = BASE_DIR / "prompt_config.json"
BRAND_FILE   = BASE_DIR / "branding.json"
BRAND_LOGO   = STATIC_DIR / "brand-logo"

for d in [LOG_DIR, STATIC_DIR, KB_DOCS, KB_STORE]:
    d.mkdir(exist_ok=True)

# NOTE: kept under the "bank_name" key so the shared index.html/admin.html
# branding JS (which reads d.bank_name) works unmodified for this bot too.
# These defaults are only ever used as an ultimate fallback — the "Create New
# Bot" flow always writes a real branding.json immediately after scaffolding.
DEFAULT_BRANDING = {
    "bank_name":  "New AI Assistant",
    "tagline":    "AI Voice Assistant",
    "badge_text": "Local AI Platform",
    "logo_emoji": "🤖",
    "kind":       "voice",   # "voice" (mic in / spoken reply) or "chat" (typed messages only)
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
# Multilingual so a Hindi question still retrieves relevant chunks from an
# English-language knowledge base (verified ~0.92 cosine similarity between
# equivalent English/Hindi sentences) — swapping this requires a KB rebuild
# (admin panel → Knowledge Base → Rebuild) since embeddings aren't comparable
# across models.
EMBED_MODEL   = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION    = "bot_kb"
RAG_TOP_K     = 3

# Admin credentials — shared with the BFSI bot's .env (set ADMIN_USER/ADMIN_PASS there)
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "apexbank2026")

# ── Runtime config: TTS voice + LLM model (editable via admin panel) ───────────
RUNTIME_FILE = BASE_DIR / "runtime_config.json"

AVAILABLE_VOICES = {
    "en": {
        "female": ["af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
                   "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky"],
        "male":   ["am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
                   "am_michael", "am_onyx", "am_puck", "am_santa"],
    },
    "hi": {
        "female": ["hf_alpha", "hf_beta"],
        "male":   ["hm_omega", "hm_psi"],
    },
}

# Kokoro G2P pipeline is keyed by lang_code, derived from the voice's prefix —
# extend this map if more languages are added later.
VOICE_PREFIX_TO_LANG_CODE = {"af": "a", "am": "a", "hf": "h", "hm": "h"}

def lang_code_for_voice(voice: str) -> str:
    return VOICE_PREFIX_TO_LANG_CODE.get(voice.split("_", 1)[0], "a")

DEFAULT_RUNTIME_CONFIG = {
    "kokoro_voice": os.getenv("TECH_KOKORO_VOICE", os.getenv("KOKORO_VOICE", "am_michael")),
    "ollama_model": os.getenv("TECH_OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "qwen3:8b")),
    "convo_language": os.getenv("TECH_CONVO_LANGUAGE", "en"),
    "whisper_model": WHISPER_MODEL,
    "kokoro_repo_id": os.getenv("KOKORO_REPO_ID", "hexgrad/Kokoro-82M"),
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
KOKORO_VOICE   = _runtime_config["kokoro_voice"]
OLLAMA_MODEL   = _runtime_config["ollama_model"]
CONVO_LANGUAGE = _runtime_config["convo_language"]
# The saved choice wins over the env default, so a model picked in the admin
# panel survives a restart.
WHISPER_MODEL  = _runtime_config.get("whisper_model", WHISPER_MODEL)
KOKORO_REPO_ID = _runtime_config.get("kokoro_repo_id") or "hexgrad/Kokoro-82M"

# ── Prompt config (editable via admin panel) ──────────────────────────────────
# Only ever used as an ultimate fallback — the "Create New Bot" flow always
# writes a real prompt_config.json (seeded from the creation form) immediately
# after scaffolding, so these generic defaults rarely if ever take effect.
DEFAULT_PROMPT_CONFIG = {
    "system_prompt": (
        "You're a friendly, easygoing AI voice assistant helping a caller over the phone. "
        "Talk like a helpful person, not a script — casual and warm, not stiff or formal.\n"
        "Keep every answer to 2-3 SHORT sentences maximum — this is a voice call, not text.\n"
        "Never use bullet points, markdown, numbers, or lists — speak naturally, one thought at a time.\n"
        "If you don't know the caller's name yet, ask for it early on in a casual way — something like "
        "'oh, and what's your name?' — not like filling out a form. Once you know it, use their first name "
        "naturally now and then when you reply, but don't overdo it — dropping it in every single sentence "
        "gets weird fast.\n"
        "If you don't yet know what the caller needs, ask a short clarifying question first before offering "
        "help.\n"
        "Stick to what's actually true — use the knowledge base context if it's there, otherwise say you're "
        "not sure rather than guessing.\n"
        "CRITICAL VOICE RULES:\n"
        "- Never start or end a reply with 'Thank you for calling', 'Thank you for contacting', or any variation — they're already on the call.\n"
        "- Get straight to the point. No preamble, no sign-off phrases."
    ),
    "greeting": "Hi there! What's your name, and how can I help you today?",
    "guardrails": [
        "Never ask for or store passwords, PINs, or other sensitive credentials",
        "Never guarantee an outcome will work — frame it as 'let's try this' rather than a promise",
    ],
    "kb_filter": "",
    "rag_top_k": 3,
    # RAG-first: always search the knowledge base before answering, and only let
    # the model fall back on its own general knowledge when nothing retrieved
    # clears rag_min_relevance (cosine similarity, 0-1).
    "rag_first": True,
    "rag_min_relevance": 0.35,
    "fallback_message": "I'm sorry, I'm having trouble helping with that right now. Please try again shortly.",
    # Empty by default so an untouched install keeps the localized (English/Hindi)
    # sign-off from LOCALIZED_STRINGS; a non-empty value here overrides it —
    # see the farewell handling in the voice and chat WS handlers.
    "farewell_message": "",
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
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)
SERVER_START = time.time()

# ── Model loading ─────────────────────────────────────────────────────────────
# Chat-kind bots never touch audio (no mic in, no spoken reply), so skip
# loading Whisper/Kokoro entirely for them — no point spending GPU/VRAM on
# models a chat bot will never call. "kind" is fixed at creation time by the
# launcher's "Create New Bot" form (see BOT_KIND below).
BOT_KIND = load_branding().get("kind", "voice")

# Which Whisper model is live right now — kept in step with the admin panel's
# swaps, and distinct from the configured one while a new one is still loading.
_stt_loaded_name = None

if BOT_KIND == "chat":
    log.info("Bot kind = chat — skipping Whisper/Kokoro model loading (text-only bot)")
    stt_model = None
    _tts_pipelines: dict[str, KPipeline] = {}
else:
    log.info("Loading Whisper (%s)...", WHISPER_MODEL)
    stt_model = whisper.load_model(WHISPER_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    _stt_loaded_name = WHISPER_MODEL
    log.info("Whisper on %s", "CUDA" if torch.cuda.is_available() else "CPU")

    log.info("Loading Kokoro TTS...")
    _tts_pipelines: dict[str, KPipeline] = {"a": KPipeline(lang_code="a", repo_id=KOKORO_REPO_ID)}

def get_tts_pipeline(lang_code: str) -> KPipeline:
    """Kokoro's G2P backend is tied to a lang_code at construction time, so each
    language needs its own pipeline instance. Built lazily and cached — the 'a'
    (English) pipeline above is always warmed up front since it's the default."""
    pipeline = _tts_pipelines.get(lang_code)
    if pipeline is None:
        log.info("Loading Kokoro TTS pipeline for lang_code=%r from %s...", lang_code, KOKORO_REPO_ID)
        pipeline = KPipeline(lang_code=lang_code, repo_id=KOKORO_REPO_ID)
        _tts_pipelines[lang_code] = pipeline
    return pipeline

def reload_tts_pipelines():
    """Drop the cached pipelines so the next spoken line is built from whichever
    Kokoro repo is configured now. The first line after a switch pays the load,
    so the default language is warmed on a thread rather than on that call."""
    _tts_pipelines.clear()
    def _warm():
        try:
            get_tts_pipeline(lang_code_for_voice(KOKORO_VOICE))
            log.info("Kokoro reloaded from %s", KOKORO_REPO_ID)
        except Exception as e:
            log.error("Could not load Kokoro from %s: %s", KOKORO_REPO_ID, e)
    threading.Thread(target=_warm, daemon=True).start()

if BOT_KIND != "chat":
    log.info("Kokoro ready.")

# ── Document ingestion (.txt / .pdf / .docx) ────────────────────────────────
KB_EXTENSIONS = (".txt", ".pdf", ".docx")

def _ocr_image_bytes(img_bytes: bytes) -> str:
    if not OCR_AVAILABLE:
        return ""
    try:
        text = pytesseract.image_to_string(Image.open(_io.BytesIO(img_bytes))).strip()
        return text
    except Exception as e:
        log.debug("OCR skipped for one image: %s", e)
        return ""

def _flatten_table(rows: list[list[str]]) -> str:
    rows = [[(c or "").strip().replace("\n", " ") for c in r] for r in rows if r]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    header, body = rows[0], rows[1:]
    if not body:
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

            words = page.get_text("words")
            kept = [w for w in words if not any(fitz.Rect(w[:4]).intersects(r) for r in table_rects)]
            kept.sort(key=lambda w: (w[5], w[6], w[7]))
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
    """quick=True gives a fast, approximate extraction (skips OCR and PDF table
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
        log.warning("KB store empty — upload docs and rebuild from the admin panel.")
        return False
    try:
        chroma_client = chromadb.PersistentClient(path=str(KB_STORE))
        kb_collection = chroma_client.get_collection(COLLECTION)
        if embedder is None:
            log.info("Loading sentence-transformer (CPU — GPU is reserved for Whisper/Kokoro)...")
            embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
        log.info("RAG ready. %d chunks.", kb_collection.count())
        return True
    except Exception as e:
        log.warning("RAG init failed: %s", e)
        return False

init_rag()

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Local AI Voice Bot")
security = HTTPBasic()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Multi-user admin auth — shared users.db at the repo root. app_key is derived
# from this bot's own folder name (its slug), NOT hardcoded — a freshly
# created bot's folder is named after its slug, so this makes it automatically
# grantable in the permission system with zero per-bot source edits needed.
# See app/auth.py for the shared implementation (duplicated per app).
auth.configure(REPO_ROOT / "users.db", app_key=BASE_DIR.name)
auth.ensure_bootstrap_user(ADMIN_USER, ADMIN_PASS)
verify_admin = auth.verify_admin

# ── Public routes ─────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Place index.html in static_bot/ folder.</h2>")

# ── Platform branding (read-only mirror of the launcher's) ──────────────────
# The launcher owns branding_launcher.json; every other app reads it so one
# rename or logo upload shows up on every page instead of just the launcher.
# Read-only here by design — editing stays in the launcher's admin panel.
PLATFORM_BRAND_FILE = REPO_ROOT / "branding_launcher.json"
PLATFORM_LOGO_DIR   = REPO_ROOT / "launcher" / "static_launcher"
PLATFORM_LOGO_MIME  = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                       "svg": "image/svg+xml", "webp": "image/webp", "gif": "image/gif"}

def _platform_logo_file():
    for ext in ("png", "jpg", "jpeg", "svg", "webp", "gif"):
        p = PLATFORM_LOGO_DIR / f"brand-logo.{ext}"
        if p.exists():
            return p
    return None

@app.get("/api/platform-branding")
async def platform_branding():
    data = {"company_name": "AI CX Platform", "logo_emoji": "\U0001F916"}
    if PLATFORM_BRAND_FILE.exists():
        try:
            data.update(json.loads(PLATFORM_BRAND_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
    data["has_logo_image"] = _platform_logo_file() is not None
    return data

@app.get("/api/platform-branding/logo")
async def platform_branding_logo():
    logo = _platform_logo_file()
    if not logo:
        raise HTTPException(404, "No platform logo uploaded")
    return FileResponse(str(logo),
                        media_type=PLATFORM_LOGO_MIME.get(logo.suffix.lstrip(".").lower(), "image/png"))

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
    log_text = "\n".join(relevant[-150:])
    prompt = (
        "You are a senior backend engineer helping troubleshoot a FastAPI voice-AI "
        "bot server (Whisper STT + local/cloud LLM + Kokoro/cloud TTS + "
        "ChromaDB RAG over a WebSocket). Below are recent log lines from the running "
        "server. Identify what's going wrong, explain the likely root cause in plain "
        "English, and suggest concrete next steps to investigate or fix it. If there "
        "are multiple distinct issues, list each separately. Keep it concise and "
        "actionable — this is read by the person operating the server, not a formal "
        "report.\n\n"
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

# ── Call recordings & transcripts ───────────────────────────────────────────────
# Each full call already has a continuous call_<ts>.wav (CallRecorder, above);
# the websocket handler's `finally` block writes a matching call_<ts>.json
# sidecar with the same turns already transcribed by Whisper for the LLM prompt
# — nothing is re-transcribed here, just persisted and made listable/searchable.

def _call_duration_secs(wav_path: Path) -> float:
    try:
        with wave.open(str(wav_path), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0

def _load_call_transcript(call_id: str) -> dict | None:
    p = LOG_DIR / f"call_{call_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

@app.get("/admin/api/recordings")
async def list_recordings(username: str = Depends(verify_admin)):
    # Keyed off the transcript JSON, not the WAV — a chat session has no
    # audio at all, so requiring a WAV would silently drop every chat
    # transcript from this list. has_audio tells the frontend whether to
    # offer playback.
    jsons = sorted(LOG_DIR.glob("call_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    result = []
    for jf in jsons:
        call_id = jf.stem.split("_", 1)[1]
        meta = _load_call_transcript(call_id) or {}
        turns = meta.get("turns", [])
        first_user_line = next((t.get("content", "") for t in turns if t.get("role") == "user"), "")
        preview = first_user_line[:140] + ("…" if len(first_user_line) > 140 else "")
        wav_path = LOG_DIR / f"call_{call_id}.wav"
        has_audio = wav_path.exists()
        result.append({
            "call_id": call_id,
            "channel": meta.get("channel", "voice"),
            "modified": datetime.fromtimestamp(jf.stat().st_mtime).strftime("%d %b %H:%M:%S"),
            "duration_secs": round(_call_duration_secs(wav_path), 1) if has_audio else None,
            "size_kb": (wav_path.stat().st_size // 1024) if has_audio else (jf.stat().st_size // 1024),
            "caller_name": meta.get("caller_name"),
            "turn_count": len(turns),
            "preview": preview,
            "has_transcript": True,
            "has_audio": has_audio,
        })
    return {"calls": result}

@app.get("/admin/api/recordings/search")
async def search_recordings(q: str, username: str = Depends(verify_admin)):
    q_lower = q.strip().lower()
    if not q_lower:
        return {"calls": []}
    result = []
    for f in sorted(LOG_DIR.glob("call_*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            meta = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        turns = meta.get("turns", [])
        snippet = None
        if meta.get("caller_name") and q_lower in meta["caller_name"].lower():
            snippet = f"Caller name matched: {meta['caller_name']}"
        else:
            for t in turns:
                content = t.get("content", "")
                idx = content.lower().find(q_lower)
                if idx == -1:
                    continue
                start = max(0, idx - 40)
                end = min(len(content), idx + len(q_lower) + 40)
                snippet = ("…" if start > 0 else "") + content[start:end] + ("…" if end < len(content) else "")
                break
        if snippet is None:
            continue
        call_id = str(meta.get("call_id") or f.stem.split("_", 1)[1])
        result.append({
            "call_id": call_id,
            "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%d %b %H:%M:%S"),
            "caller_name": meta.get("caller_name"),
            "turn_count": len(turns),
            "snippet": snippet,
            "has_audio": (LOG_DIR / f"call_{call_id}.wav").exists(),
        })
    return {"calls": result}

@app.post("/admin/api/recordings/clear")
async def clear_all_recordings(username: str = Depends(verify_admin)):
    wavs  = list(LOG_DIR.glob("call_*.wav"))
    jsons = list(LOG_DIR.glob("call_*.json"))
    for f in wavs + jsons:
        f.unlink()
    log.info("All call recordings + transcripts cleared by admin (%d wav, %d transcript files)", len(wavs), len(jsons))
    return {"status": "cleared", "count": len(wavs)}

@app.get("/admin/api/recordings/{call_id}")
async def get_recording_transcript(call_id: str, username: str = Depends(verify_admin)):
    meta = _load_call_transcript(call_id)
    if not meta:
        raise HTTPException(404, "No transcript saved for this call")
    return meta

@app.delete("/admin/api/recordings/{call_id}")
async def delete_recording(call_id: str, username: str = Depends(verify_admin)):
    wav_path  = LOG_DIR / f"call_{call_id}.wav"
    json_path = LOG_DIR / f"call_{call_id}.json"
    if not wav_path.exists() and not json_path.exists():
        raise HTTPException(404, "Recording not found")
    wav_path.unlink(missing_ok=True)
    json_path.unlink(missing_ok=True)
    log.info("Call recording deleted by admin: call_%s", call_id)
    return {"status": "deleted"}

@app.get("/admin/api/callers")
async def list_callers(username: str = Depends(verify_admin)):
    """Cross-call caller memory — who's called before, what they called about, and
    whether it got resolved. Identity is by spoken name only (no phone/caller ID
    on this line), so this is a best-effort match, not a hard guarantee."""
    history = load_caller_history()
    result = []
    for rec in history.values():
        calls = rec.get("calls", [])
        last = calls[-1] if calls else {}
        result.append({
            "name": rec.get("display_name"),
            "call_count": len(calls),
            "last_call_date": last.get("date"),
            "last_issue": last.get("issue"),
            "last_resolved": last.get("resolved"),
            "calls": list(reversed(calls)),
        })
    result.sort(key=lambda r: r.get("last_call_date") or "", reverse=True)
    return {"callers": result}

@app.delete("/admin/api/callers/{name}")
async def delete_caller(name: str, username: str = Depends(verify_admin)):
    history = load_caller_history()
    key = _caller_key(name)
    if key not in history:
        raise HTTPException(404, "Caller not found")
    del history[key]
    save_caller_history(history)
    log.info("Caller history deleted by admin: %s", name)
    return {"status": "deleted"}

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
    return HTMLResponse("<h2>Place admin.html in static_bot/ folder.</h2>")

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

@app.get("/admin/api/stt/models")
async def get_stt_models(username: str = Depends(verify_admin)):
    """Every Whisper model, which are on disk, and which one is live."""
    cfg = load_runtime_config()
    return {
        "models": list_stt_models(),
        "configured": cfg.get("whisper_model", WHISPER_MODEL),
        "loaded": _stt_loaded_name,
        "loading": _stt_load_state["loading"],
        "error": _stt_load_state["error"],
        "cache_dir": str(whisper_cache_dir()),
    }

@app.post("/admin/api/stt/download")
async def download_stt_model(data: dict, username: str = Depends(verify_admin)):
    model = (data.get("model") or "").strip()
    try:
        started = start_stt_download(model)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not started:
        raise HTTPException(409, f'"{model}" is already downloading')
    return {"status": "started", "model": model}

@app.get("/admin/api/stt/download-status")
async def get_stt_download_status(model: str, username: str = Depends(verify_admin)):
    if not model:
        raise HTTPException(400, "model required")
    return stt_download_status(model)

@app.post("/admin/api/stt/apply")
async def apply_stt(data: dict, username: str = Depends(verify_admin)):
    """Save the chosen model and start swapping it in. The reply comes back
    immediately — poll /admin/api/stt/models to see when it is live."""
    model = (data.get("model") or "").strip()
    if model not in whisper._MODELS:
        raise HTTPException(400, f"Unknown Whisper model '{model}'")
    if not stt_model_downloaded(model):
        raise HTTPException(400, f"'{model}' isn't downloaded yet — download it first")
    cfg = load_runtime_config()
    cfg["whisper_model"] = model
    save_runtime_config(cfg)
    if BOT_KIND == "chat":
        # A chat bot never loads speech models, so just remember the choice.
        return {"status": "saved", "model": model, "note": "text-only bot — nothing to load"}
    apply_stt_model(model)
    log.info("STT model switching to %s (admin)", model)
    return {"status": "loading", "model": model}

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
        "convo_language": CONVO_LANGUAGE,
        "kokoro_repo_id": KOKORO_REPO_ID,
        "available_voices": AVAILABLE_VOICES,
        "available_models": installed_models,
    }

@app.post("/admin/api/runtime-config")
async def save_runtime_config_api(data: dict, username: str = Depends(verify_admin)):
    global KOKORO_VOICE, OLLAMA_MODEL, CONVO_LANGUAGE, KOKORO_REPO_ID
    voice = (data.get("kokoro_voice") or KOKORO_VOICE).strip()
    model = (data.get("ollama_model") or OLLAMA_MODEL).strip()
    language = (data.get("convo_language") or CONVO_LANGUAGE).strip()
    if language not in AVAILABLE_VOICES:
        raise HTTPException(400, f"Unknown language {language!r}")
    KOKORO_VOICE = voice
    OLLAMA_MODEL = model
    CONVO_LANGUAGE = language
    repo = (data.get("kokoro_repo_id") or KOKORO_REPO_ID).strip()
    repo_changed = repo != KOKORO_REPO_ID
    KOKORO_REPO_ID = repo
    # Read-modify-write: whisper_model lives in the same file and is set from a
    # different pane, so rebuilding this dict from scratch would silently drop it.
    save_runtime_config({**load_runtime_config(), "kokoro_voice": voice,
                         "ollama_model": model, "convo_language": language,
                         "kokoro_repo_id": repo})
    if repo_changed and BOT_KIND != "chat":
        reload_tts_pipelines()
    log.info("Runtime config updated by admin: voice=%s model=%s language=%s", voice, model, language)
    return {"status": "saved", "kokoro_voice": voice, "ollama_model": model, "convo_language": language}

VOICE_SAMPLE_TEXT = {
    "en": "Hi, thanks for calling. This is a quick preview of this voice.",
    "hi": "नमस्ते, कॉल करने के लिए धन्यवाद। यह इस आवाज़ का एक छोटा नमूना है।",
}

@app.get("/admin/api/voice-sample")
async def voice_sample(voice: str, username: str = Depends(verify_admin)):
    valid_voices = [v for genders in AVAILABLE_VOICES.values() for vs in genders.values() for v in vs]
    if voice not in valid_voices:
        raise HTTPException(400, "Unknown voice")
    lang = lang_code_for_voice(voice)
    sample_text = VOICE_SAMPLE_TEXT.get("hi" if lang == "h" else "en", VOICE_SAMPLE_TEXT["en"])
    loop = asyncio.get_event_loop()
    pcm = await loop.run_in_executor(None, synthesize, sample_text, voice)
    return Response(content=pcm_to_wav_bytes(pcm), media_type="audio/wav")

# ── Pluggable provider config (local ↔ cloud LLM/TTS) ──────────────────────────
def _mask_provider_config(cfg: dict) -> dict:
    """Never send real API keys to the browser — replace each with a has_key flag."""
    out = json.loads(json.dumps(cfg))
    out["llm_cloud"]["has_key"] = bool(out["llm_cloud"].pop("api_key", ""))
    out["stt_cloud"]["has_key"] = bool(out["stt_cloud"].pop("api_key", ""))
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

    cfg["stt_mode"] = data.get("stt_mode", cfg["stt_mode"])
    incoming_stt = data.get("stt_cloud", {})
    cfg["stt_cloud"]["base_url"] = incoming_stt.get("base_url", cfg["stt_cloud"]["base_url"])
    cfg["stt_cloud"]["model"] = incoming_stt.get("model", cfg["stt_cloud"]["model"])
    if incoming_stt.get("api_key"):
        cfg["stt_cloud"]["api_key"] = incoming_stt["api_key"]
    cfg["tts_mode"] = data.get("tts_mode", cfg["tts_mode"])
    cfg["tts_local_engine"] = data.get("tts_local_engine", cfg.get("tts_local_engine", "kokoro"))
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

@app.post("/admin/api/providers/cloud-models")
async def list_cloud_models(data: dict, username: str = Depends(verify_admin)):
    """Ask an OpenAI-compatible provider which models it serves, so the admin can
    pick one instead of having to know its exact id. Falls back to the saved key
    when the form's key box is blank, which it is whenever a key is already
    stored."""
    saved    = load_provider_config()["llm_cloud"]
    base_url = (data.get("base_url") or saved.get("base_url", "")).rstrip("/")
    api_key  = data.get("api_key") or saved.get("api_key", "")
    if not base_url:
        return {"error": "Base URL required"}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            )
            resp.raise_for_status()
            models = [m.get("id", "") for m in resp.json().get("data", []) if m.get("id")]
        return {"models": sorted(models)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

@app.post("/admin/api/providers/test-tts")
async def test_tts(data: dict, username: str = Depends(verify_admin)):
    """Test candidate (not-yet-saved) cloud TTS settings before committing them."""
    engine = data.get("engine", "elevenlabs")
    saved = load_provider_config()["tts_cloud"].get(engine, {})
    cfg = {**saved, **{k: v for k, v in data.items() if k not in ("engine", "text") and v}}
    text = data.get("text") or "Hi, this is a quick preview of this cloud voice."
    fn = _TTS_CLOUD_ENGINES.get(engine)
    if fn is None:
        raise HTTPException(400, f"Unknown TTS provider: {engine}. Restart the bot after updating providers.")
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
    global chroma_client, kb_collection, embedder
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

        # Always reload (not just when unset) so a changed EMBED_MODEL takes effect
        # on rebuild without needing a full server restart.
        embedder = SentenceTransformer(EMBED_MODEL, device="cpu")

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

        chroma_client = client
        kb_collection = collection
        log.info("KB rebuilt: %d chunks from %d files (skipped: %s)",
                  len(all_chunks), len(doc_files), skipped or "none")
        return {"status": "ok", "chunks": len(all_chunks), "files": len(doc_files), "skipped": skipped}
    except Exception as e:
        log.error("KB rebuild failed: %s", e)
        return {"status": "error", "message": str(e)}

@app.post("/admin/api/kb/search")
async def search_kb(data: dict, username: str = Depends(verify_admin)):
    query = data.get("query", "")
    top_k = int(data.get("top_k", 3))
    if not query:
        raise HTTPException(400, "query required")
    if not kb_collection or not embedder:
        return {"error": "RAG not loaded — rebuild KB first"}
    # Scored and cut off exactly like a live turn, so the relevance floor can be
    # tuned from what this preview shows.
    cfg       = load_prompt_config()
    rag_first = cfg.get("rag_first", DEFAULT_PROMPT_CONFIG["rag_first"])
    min_rel   = float(cfg.get("rag_min_relevance", DEFAULT_PROMPT_CONFIG["rag_min_relevance"]))
    hits      = retrieve_kb_hits(query, top_k)
    return {
        "query": query, "rag_first": rag_first, "min_relevance": min_rel,
        "results": [{**h, "used": (not rag_first) or h["score"] >= min_rel} for h in hits],
    }

# ── Branding routes ───────────────────────────────────────────────────────────
@app.get("/admin/api/branding")
async def get_branding(username: str = Depends(verify_admin)):
    data = load_branding()
    data["has_logo_image"] = find_logo_file() is not None
    return data

@app.get("/api/branding")
async def get_branding_public():
    data = load_branding()
    data["has_logo_image"] = find_logo_file() is not None
    return data

@app.post("/admin/api/branding")
async def save_branding_route(data: dict, username: str = Depends(verify_admin)):
    # Merge rather than overwrite — the admin form only ever submits the
    # fields it manages (name/tagline/badge/emoji), so a blind overwrite
    # would silently drop "kind" (set once at bot creation, not editable
    # here) on the very next branding save.
    save_branding({**load_branding(), **data})
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

# ── Speech gate ───────────────────────────────────────────────────────────────
# Whisper never returns an empty transcript for silence. Trained largely on
# captioned video, it emits the caption it expects to see, and its favourites
# ("Thank you.", "Thanks for watching.") are exactly what the farewell detector
# reads as "the caller is done" — so tapping the mic button and letting go ended
# the call outright (logs_tech has 682 bytes of audio, 21ms, transcribed as
# "Thank you for watching." followed by the session closing). These two checks
# cost nothing and run before Whisper is ever asked.
MIC_SAMPLE_RATE = 16000   # what the browser sends, whatever rate TTS runs at
MIN_SPEECH_MS   = 400     # no useful turn fits in less than this
SILENCE_PEAK    = 0.02    # 16-bit PCM normalised to -1..1
SILENCE_RMS     = 0.004

def speech_check(pcm_bytes: bytes) -> tuple[bool, str]:
    """Whether a captured turn plausibly contains speech, and if not, why not.
    Deliberately lenient — this only has to reject silence and stray taps, not
    judge audio quality, so quiet speech still gets through to Whisper."""
    samples = pcm_to_numpy(pcm_bytes)
    ms = len(samples) / MIC_SAMPLE_RATE * 1000
    if ms < MIN_SPEECH_MS:
        return False, f"only {ms:.0f}ms of audio"
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms  = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
    if peak < SILENCE_PEAK and rms < SILENCE_RMS:
        return False, f"no speech in {ms:.0f}ms (peak {peak:.3f}, rms {rms:.4f})"
    return True, ""

# Whatever Whisper falls back on when it has nothing to transcribe. Only ever
# treated as noise when one of these is the *entire* transcript.
_WHISPER_FILLER = {
    "thank you", "thanks", "thanks for watching", "thank you for watching",
    "thanks for watching bye", "you", "bye", "so", "uh", "um", "okay", "ok",
    "please subscribe", "subscribe", "the end", "silence", "music",
}

def _user_turns(conversation: list[dict]) -> int:
    return sum(1 for m in conversation if m.get("role") == "user")

def is_whisper_filler(text: str) -> bool:
    stripped = re.sub(r"[^a-z ]", " ", (text or "").lower())
    return " ".join(stripped.split()) in _WHISPER_FILLER

# Generic — no domain vocabulary bias. A bot with a specific KB (tech support,
# HR, sales, etc.) benefits from a more targeted hint, but that's set via
# admin configuration going forward, not hardcoded per-bot at the source level.
WHISPER_NAME_HINT = (
    "A phone call between an AI assistant and a caller. Ordinary conversational speech."
)

# ── Fixed (non-LLM) canned lines, localized per CONVO_LANGUAGE ─────────────────
LOCALIZED_STRINGS = {
    "en": {
        "repeat": [
            "Sorry, I didn't catch that — could you say that again?",
            "I still didn't get that. Could you speak a little closer to the mic?",
            "I'm not picking up any sound — check that your microphone is on, then try again.",
        ],
        "nudges": [
            "Are you still there? Please go ahead — I'm listening.",
            "I'm still here whenever you're ready. What can I help you with?",
            "I haven't heard from you for a while. I'll be closing this session now.",
        ],
        "timeout": (
            "We haven't heard from you after a few attempts. Your session has now "
            "ended. Please reach out again whenever you're ready — goodbye."
        ),
        "farewell": "Glad I could help! Have a great day. Goodbye!",
        "reset_greeting": (
            "No problem — let's start fresh. What's your name, and how can I help you?"
        ),
    },
    "hi": {
        "repeat": [
            "माफ़ कीजिए, मैं समझ नहीं पाया — क्या आप दोबारा कह सकते हैं?",
            "अभी भी सुनाई नहीं दिया। क्या आप माइक के थोड़ा पास बोल सकते हैं?",
            "मुझे कोई आवाज़ नहीं मिल रही — कृपया जाँच लें कि आपका माइक्रोफ़ोन चालू है, फिर दोबारा कोशिश करें।",
        ],
        "nudges": [
            "क्या आप अभी भी वहाँ हैं? कृपया बोलिए, मैं सुन रहा हूँ।",
            "मैं अभी भी यहाँ हूँ, जब आप तैयार हों बताइए। मैं आपकी कैसे मदद कर सकता हूँ?",
            "काफ़ी समय से आपकी तरफ़ से कोई जवाब नहीं आया। मैं अब यह सेशन बंद कर रहा हूँ।",
        ],
        "timeout": (
            "कई बार कोशिश करने के बाद भी आपकी तरफ़ से कोई जवाब नहीं आया। आपका सेशन अब समाप्त हो "
            "गया है। जब भी आप तैयार हों, फिर से संपर्क करें — अलविदा।"
        ),
        "farewell": "मुझे मदद करके खुशी हुई! आपका दिन शुभ रहे। अलविदा!",
        "reset_greeting": (
            "कोई बात नहीं — चलिए फिर से शुरू करते हैं। आपका नाम क्या है, और मैं आपकी कैसे मदद कर "
            "सकता हूँ?"
        ),
    },
}

def repeat_line(attempt: int) -> str:
    """Ask for the turn again, escalating: by the third miss in a row the caller
    needs a hint about their microphone, not the same apology again."""
    lines = localized("repeat")
    return lines[min(max(attempt, 1), len(lines)) - 1]

def localized(key: str) -> str:
    return LOCALIZED_STRINGS.get(CONVO_LANGUAGE, LOCALIZED_STRINGS["en"]).get(
        key, LOCALIZED_STRINGS["en"][key]
    )

# ── Speech-to-text engine ─────────────────────────────────────────────────────
# The Whisper model is swappable while the bot is running: a new one is loaded
# on a worker thread and only becomes `stt_model` once it is fully ready, so an
# in-flight call keeps transcribing with the old one instead of hitting a
# half-loaded model or a None.
STT_MODEL_SIZES_MB = {
    "tiny.en": 75, "tiny": 75, "base.en": 142, "base": 142,
    "small.en": 484, "small": 484, "medium.en": 1500, "medium": 1500,
    "large-v1": 2900, "large-v2": 2900, "large-v3": 2900, "large": 2900,
    "large-v3-turbo": 1600, "turbo": 1600,
}

_stt_downloads: dict[str, dict] = {}
_stt_load_state = {"loading": None, "error": None}

def whisper_cache_dir() -> Path:
    """Where openai-whisper keeps its weights — the same default it downloads to,
    so a model pulled from the admin panel is the one load_model() then finds."""
    return Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "whisper"

def _stt_model_file(name: str) -> Path:
    url = whisper._MODELS.get(name, "")
    return whisper_cache_dir() / os.path.basename(url) if url else whisper_cache_dir() / f"{name}.pt"

def stt_model_downloaded(name: str) -> bool:
    p = _stt_model_file(name)
    # A partially-downloaded file is still on disk, so require most of the
    # expected size before calling it ready.
    if not p.exists():
        return False
    expected = STT_MODEL_SIZES_MB.get(name, 0) * 1024 * 1024
    return p.stat().st_size >= expected * 0.85 if expected else True

def list_stt_models() -> list[dict]:
    active = load_runtime_config().get("whisper_model", WHISPER_MODEL)
    return [{
        "name": name,
        "size_mb": STT_MODEL_SIZES_MB.get(name, 0),
        "downloaded": stt_model_downloaded(name),
        "active": name == active,
        "loaded": name == _stt_loaded_name,
    } for name in whisper._MODELS]

def _download_stt_model(name: str):
    """Pull the weights via whisper itself (it verifies the checksum), while the
    status endpoint watches the file grow."""
    state = _stt_downloads[name]
    try:
        whisper._download(whisper._MODELS[name], str(whisper_cache_dir()), False)
        state.update(status="done", percent=100)
        log.info("STT model %s downloaded", name)
    except Exception as e:
        state.update(status="error", error=str(e))
        log.error("STT model %s download failed: %s", name, e)

def start_stt_download(name: str) -> bool:
    if name not in whisper._MODELS:
        raise ValueError(f"Unknown Whisper model '{name}'")
    cur = _stt_downloads.get(name)
    if cur and cur.get("status") == "downloading":
        return False
    whisper_cache_dir().mkdir(parents=True, exist_ok=True)
    _stt_downloads[name] = {"status": "downloading", "percent": 0, "error": None}
    threading.Thread(target=_download_stt_model, args=(name,), daemon=True).start()
    log.info("Downloading STT model %s...", name)
    return True

def stt_download_status(name: str) -> dict:
    state = dict(_stt_downloads.get(name) or {"status": "idle", "percent": 0, "error": None})
    total_mb = STT_MODEL_SIZES_MB.get(name, 0)
    path = _stt_model_file(name)
    have_mb = round(path.stat().st_size / (1024 * 1024), 1) if path.exists() else 0
    if state["status"] == "downloading" and total_mb:
        state["percent"] = min(99, int(have_mb / total_mb * 100))
    state.update(model=name, downloaded_mb=have_mb, total_mb=total_mb,
                 downloaded=stt_model_downloaded(name))
    return state

def _load_stt_model(name: str):
    global stt_model, _stt_loaded_name
    try:
        log.info("Loading Whisper model %s...", name)
        model = whisper.load_model(name, device="cuda" if torch.cuda.is_available() else "cpu")
        stt_model = model
        _stt_loaded_name = name
        _stt_load_state.update(loading=None, error=None)
        log.info("Whisper model %s is now live", name)
    except Exception as e:
        _stt_load_state.update(loading=None, error=str(e))
        log.error("Could not load Whisper model %s: %s", name, e)

def apply_stt_model(name: str):
    """Swap the live Whisper model. Returns immediately — the load runs on a
    thread and the current model keeps serving until the new one is ready."""
    if name not in whisper._MODELS:
        raise ValueError(f"Unknown Whisper model '{name}'")
    if name == _stt_loaded_name or _stt_load_state["loading"] == name:
        return
    _stt_load_state.update(loading=name, error=None)
    threading.Thread(target=_load_stt_model, args=(name,), daemon=True).start()

async def transcribe_cloud(pcm_bytes: bytes, cfg: dict) -> str:
    """Any OpenAI-compatible /audio/transcriptions endpoint — OpenAI and Groq
    both serve Whisper this way, which is worth having when the GPU is busy."""
    base_url = (cfg.get("base_url") or "https://api.openai.com/v1").rstrip("/")
    data = {"model": cfg.get("model") or "whisper-1"}
    # The bank bot is English-only and defines no conversation language.
    lang = globals().get("CONVO_LANGUAGE", "en")
    if lang:
        data["language"] = lang
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{base_url}/audio/transcriptions",
            headers={"Authorization": f"Bearer {cfg.get('api_key', '')}"},
            files={"file": ("audio.wav", pcm_to_wav_bytes(pcm_bytes), "audio/wav")},
            data=data,
        )
        resp.raise_for_status()
        return (resp.json().get("text") or "").strip()

async def transcribe(pcm_bytes: bytes) -> str:
    provider = load_provider_config()
    if provider.get("stt_mode") == "cloud":
        try:
            return await transcribe_cloud(pcm_bytes, provider.get("stt_cloud", {}))
        except Exception as e:
            # Falling back beats dropping the caller's turn on the floor.
            log.error("Cloud STT failed (%s) — falling back to the local model", e)
    audio = pcm_to_numpy(pcm_bytes)
    loop  = asyncio.get_event_loop()
    # The English tech-term hint biases Whisper toward English vocabulary, so it's
    # only useful (and only passed) when the conversation is actually in English.
    kwargs = {"language": CONVO_LANGUAGE, "fp16": torch.cuda.is_available()}
    if CONVO_LANGUAGE == "en":
        kwargs["initial_prompt"] = WHISPER_NAME_HINT
    result = await loop.run_in_executor(None, lambda: stt_model.transcribe(audio, **kwargs))
    return result["text"].strip()

def _cosine(a, b) -> float:
    """Cosine similarity of two vectors, clamped to 0-1."""
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))

def retrieve_kb_hits(query: str, top_k: int = None) -> list[dict]:
    """Top-K knowledge-base chunks for a question, best match first, each scored
    0-1 by cosine similarity.

    The score is recomputed here from the returned vectors rather than read off
    Chroma's distance: the collection is built in Chroma's default squared-L2
    space over un-normalised embeddings, so its distances have no fixed range to
    threshold against, while cosine does."""
    cfg = load_prompt_config()
    k   = top_k or cfg.get("rag_top_k", RAG_TOP_K)
    kb_filter = cfg.get("kb_filter", "").strip()
    if not kb_collection or not embedder or not query.strip():
        return []
    try:
        qvec    = embedder.encode([query]).tolist()
        where   = {"source": {"$contains": kb_filter}} if kb_filter else None
        results = kb_collection.query(
            query_embeddings=qvec, n_results=k,
            where=where if where else None,
            include=["documents", "metadatas", "embeddings"],
        )
        docs  = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        embs  = results.get("embeddings")
        # Chroma hands embeddings back as numpy arrays, so index them
        # positionally rather than testing them for truthiness.
        vecs  = embs[0] if embs is not None and len(embs) else []
        hits  = []
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            hits.append({
                "text":   doc,
                "source": (meta or {}).get("source", "?"),
                "score":  round(_cosine(qvec[0], vecs[i]), 3) if i < len(vecs) else 0.0,
            })
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits
    except Exception as e:
        log.warning("RAG query failed: %s", e)
        return []

def kb_lookup(query: str) -> dict:
    """Knowledge-base lookup for one question, ready to drop into a prompt.

    With RAG-first on (admin panel → System Prompt), the knowledge base is always
    searched before the model is allowed to answer: chunks scoring at or above
    rag_min_relevance become authoritative context, and only when nothing clears
    that bar is the model told to fall back on its own general knowledge — and to
    say that's what it's doing. With RAG-first off, whatever comes back is passed
    along as optional context, exactly as before.

    Returns {"context", "instruction", "section", "grounded", "hits"}; "section"
    is context+instruction pre-joined for the common case of appending one block
    to a system prompt."""
    cfg       = load_prompt_config()
    rag_first = cfg.get("rag_first", DEFAULT_PROMPT_CONFIG["rag_first"])
    min_rel   = float(cfg.get("rag_min_relevance", DEFAULT_PROMPT_CONFIG["rag_min_relevance"]))
    hits      = retrieve_kb_hits(query)
    best      = hits[0]["score"] if hits else 0.0

    if not rag_first:
        used, instruction, grounded = hits, "", bool(hits)
    else:
        used     = [h for h in hits if h["score"] >= min_rel]
        grounded = bool(used)
        if used:
            log.info("RAG-first ▶ %d of %d chunk(s) cleared relevance %.2f (best %.2f, from %s) "
                     "— answering from the knowledge base",
                     len(used), len(hits), min_rel, best,
                     ", ".join(sorted({h["source"] for h in used})))
            instruction = (
                "RAG-FIRST MODE: Answer from the knowledge base context above. It outranks anything you "
                "think you know, so don't contradict it and don't layer outside facts on top of it. If it "
                "only covers part of what was asked, answer that part and say plainly that you don't have "
                "the rest on file."
            )
        else:
            log.info("RAG-first ▶ nothing cleared relevance %.2f (best %.2f of %d chunk(s)) "
                     "— falling back to the model's own knowledge", min_rel, best, len(hits))
            instruction = (
                "RAG-FIRST MODE: Nothing in the knowledge base matched this question, so unless the "
                "answer is already in the context above, answer from your own general knowledge — and "
                "make it clear you're speaking generally rather than from our documented material. Don't "
                "state specifics (figures, policies, procedures) you can't stand behind."
            )

    context = "\n\n".join(h["text"] for h in used)
    section = ""
    if context:
        section += f"\n\nKNOWLEDGE BASE CONTEXT (known issues/procedures):\n{context}"
    if instruction:
        section += f"\n\n{instruction}"
    return {"context": context, "instruction": instruction, "section": section,
            "grounded": grounded, "hits": used}

def time_of_day_greeting() -> str:
    """"Good morning/afternoon/evening" for the caller's very first hello once
    a name is known. "Hello" outside all three bands rather than "good night",
    since the latter reads as a sign-off on a call that has just started."""
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good morning"
    if 12 <= hour < 17:
        return "Good afternoon"
    if 17 <= hour < 22:
        return "Good evening"
    return "Hello"

_NAME_PATTERNS = [
    # "my name is X [Y]" is unambiguous enough to allow a two-word capture.
    re.compile(r"\bmy name(?:'s| is)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)", re.IGNORECASE),
    # "I'm X" / "it's X" / "this is X" are risky — far more often followed by a verb
    # phrase ("I'm having trouble...") than a name, so only take a single word and
    # only when it's immediately followed by a sentence boundary or one of a few
    # known connectors, not an arbitrary continuation.
    re.compile(r"\b(?:i'?m|it'?s|this is)\s+([A-Za-z]+)(?=[.,!?]|\s+and\b|\s+here\b|\s+calling\b|$)", re.IGNORECASE),
    re.compile(r"^([A-Za-z]+)\s+here\b", re.IGNORECASE),
    # Hindi: "मेरा नाम X है" ("my name is X") — same unambiguous shape as the
    # English "my name is X" pattern above.
    re.compile(r"मेरा नाम\s+([ऀ-ॿ]+(?:\s+[ऀ-ॿ]+)?)\s*है"),
]
# Trailing words trimmed off a match rather than rejecting it outright, e.g.
# "my name is Priya and my laptop..." → "Priya" (drop the dangling "and").
_TRAILING_CONNECTORS = {"and", "but", "so", "calling", "here", "who", "which", "that"}
# If any *remaining* word is one of these, the match is almost certainly a verb
# phrase caught by the pattern above, not an actual name — reject it entirely.
_NAME_STOPWORDS = {
    "having", "trying", "using", "not", "still", "just", "calling", "getting",
    "seeing", "facing", "experiencing", "unable", "looking", "working", "done",
    "good", "fine", "okay", "ok", "ready", "sorry", "back", "there", "gonna",
    "going", "about", "on", "with", "so", "very", "really", "also", "sure",
    "afraid", "wondering", "hoping", "glad", "happy", "excited", "frozen",
    "stuck", "confused", "lost", "new", "here", "trouble",
}

# A subset of the farewell phrases below that specifically claim the issue got
# fixed, not just "goodbye" — used to infer resolved status for caller history.
_RESOLVED_SIGNAL = re.compile(
    r"\b(problem solved|that fixed it|it'?s? working now|i('m| am) (all set|fixed))\b",
    re.IGNORECASE
)
_RESOLVED_SIGNAL_HI = re.compile(r"(समस्या (हल|ठीक) हो गई|काम कर रहा है)")

def extract_caller_name(transcript: str) -> str | None:
    """Lightweight heuristic name pickup — no separate LLM round-trip, just enough
    to catch someone saying 'hi, it's Priya' or 'my name is Rahul Kumar' naturally
    at the top of a call. Deliberately conservative: better to miss a real name and
    ask again than to confidently address someone by a misfired verb phrase."""
    for pat in _NAME_PATTERNS:
        m = pat.search(transcript)
        if not m:
            continue
        words = m.group(1).strip().split()
        while words and words[-1].lower() in _TRAILING_CONNECTORS:
            words.pop()
        if not words or any(w.lower() in _NAME_STOPWORDS for w in words):
            continue
        return " ".join(words).title()
    return None

def synthesize(text: str, voice: str | None = None) -> bytes:
    voice = voice or KOKORO_VOICE
    pipeline = get_tts_pipeline(lang_code_for_voice(voice))
    chunks = [a for _, _, a in pipeline(text, voice=voice) if a is not None]
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

# ── Caller history (cross-call memory) ──────────────────────────────────────
# There's no phone number / caller ID on this WebSocket-based line — the only
# identifier we have is the name the caller gives us in speech. So identity
# here is a best-effort match on normalized name, not a hard guarantee: two
# different callers who share a first name will be treated as the same person.
CALLER_HISTORY_FILE = BASE_DIR / "caller_history.json"

def _caller_key(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())

def load_caller_history() -> dict:
    if CALLER_HISTORY_FILE.exists():
        try:
            return json.loads(CALLER_HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Could not load caller_history.json: %s", e)
    return {}

def save_caller_history(history: dict):
    CALLER_HISTORY_FILE.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

def get_caller_record(name: str) -> dict | None:
    return load_caller_history().get(_caller_key(name))

def record_caller_call(name: str, call_id: int, issue: str, resolved: bool | None):
    history = load_caller_history()
    rec = history.setdefault(_caller_key(name), {"display_name": name, "calls": []})
    rec["display_name"] = name  # keep the most recently used casing
    rec["calls"].append({
        "call_id": call_id,
        "date": datetime.now().isoformat(),
        "issue": issue,
        "resolved": resolved,
    })
    save_caller_history(history)
    log.info("Recorded call for caller %r: issue=%r resolved=%s", name, issue, resolved)

# ── Turn-level conversation discipline ──────────────────────────────────────
# Appended to the system prompt on every turn, after the admin's editable text
# and the guardrails, because both failures these rules address were bad enough
# to warrant not being switch-off-able from the panel:
#   * the bot opening on the caller's *previous* issue before hearing the
#     current one, and
#   * the bot asking how a troubleshooting step went when it had never given
#     that step on this call.
THIS_CALL_ONLY_RULES = (
    "\n\nTHIS CALL ONLY — non-negotiable:\n"
    "- Everything under \"Conversation so far on THIS call\" is the entire history of what has "
    "actually been said between you and this caller right now. Nothing outside it has happened.\n"
    "- Only ask how a step went if YOU gave that exact step in this conversation. If you cannot "
    "point to yourself saying it above, you never said it — do not ask about its result, and do "
    "not say things like \"did that work?\", \"what happened when you tried that?\" or \"any luck "
    "with the restart?\" out of nowhere.\n"
    "- Never assume the caller has already tried, checked, restarted, unplugged or reinstalled "
    "anything unless they said so above.\n"
    "- Open by dealing with the problem the caller is describing on this call. Do not lead with, "
    "or steer back to, anything from an earlier call."
)

def this_call_issue(conversation: list[dict], limit: int = 4) -> str:
    """What this call is about, in the caller's own words — the first few things
    they said. Used to judge whether an earlier call is even related."""
    said = [m["content"] for m in conversation if m["role"] == "user" and m.get("content")]
    return " ".join(said[:limit]).strip()

# Cosine similarity above which a past call counts as "the same sort of problem"
# and may be referenced. Tuned to be deliberately reluctant: a false negative
# just means a normal fresh-issue call, a false positive is the exact failure
# the caller complained about.
PAST_CALL_RELEVANCE = 0.55

def caller_history_section(caller_name: str, past_calls: list[dict], current_issue: str) -> str:
    """The past-calls part of the system prompt.

    Every call is treated as a new, unrelated problem by default. A prior call is
    only surfaced to the model when the caller's current words actually look like
    the same problem — checked against the one-line issue summaries stored in
    caller_history.json. When nothing matches, the model is told plainly that the
    history exists but must stay out of the conversation.

    Unlike techsupport-voice-bot, this bot has no call-transcript RAG store to use
    as a second opinion — this checks only the one-line summaries."""
    n = len(past_calls)
    base = (f"\n\nThis caller has reached out {n} time(s) before, so greet them warmly as "
            f"someone you've spoken to before — a brief \"good to hear from you again\" is plenty.")

    if not current_issue:
        # Name known, but they haven't said what's wrong yet. Nothing to match
        # against, so say nothing about the past — asking about an old issue here
        # is exactly the behaviour we're removing.
        return base + (
            "\n\nYou do not yet know what they are calling about today. Ask what's going on now. "
            "Do NOT bring up, hint at, or ask about any previous call."
        )

    related, best = [], 0.0
    if embedder:
        try:
            summaries = [c.get("issue") or "" for c in past_calls]
            if any(summaries):
                vecs = embedder.encode([current_issue] + summaries)
                for c, v in zip(past_calls, vecs[1:]):
                    score = _cosine(list(vecs[0]), list(v))
                    best = max(best, score)
                    if score >= PAST_CALL_RELEVANCE:
                        related.append({"issue": c.get("issue"), "date": c.get("date"),
                                        "resolved": c.get("resolved"), "score": round(score, 3)})
        except Exception as e:
            log.warning("Past-issue similarity check failed: %s", e)

    if not related:
        log.info("Caller history ▶ %s: %d prior call(s), none related to %r "
                 "(best %.2f < %.2f) — treating as a brand-new issue",
                 caller_name, n, current_issue[:60], best, PAST_CALL_RELEVANCE)
        return base + (
            "\n\nWhat they are calling about today is UNRELATED to any of those earlier calls. "
            "Treat this as a completely new issue: do not mention, summarise, or ask about a "
            "previous call, and do not carry over any assumption from one. Work only from what "
            "they tell you on this call."
        )

    related.sort(key=lambda r: r["score"], reverse=True)
    top = related[0]
    status = ("it was resolved" if top.get("resolved") is True else
              "it was not resolved" if top.get("resolved") is False else
              "the outcome was never confirmed")
    log.info("Caller history ▶ %s: prior call %r matches today's issue (%.2f) — allowed as context",
             caller_name, top.get("issue"), top["score"])
    return base + (
        f"\n\nToday's problem looks like the same one they called about on "
        f"{(top.get('date') or '')[:10]}: \"{top.get('issue')}\" — {status}. "
        f"You may acknowledge that briefly once (\"looks like this one's back\") and use it to skip "
        f"steps they already went through. Still let them describe what's happening now, and do not "
        f"assume the earlier steps were repeated unless they say so."
    )

# ── Pluggable LLM / TTS providers (local ↔ cloud, admin-configurable) ──────────
# Kept in a separate, git-ignored file since cloud mode stores API keys — unlike
# runtime_config.json (voice/model name only, no secrets), this must never
# be committed. "local" mode is untouched: it still just uses KOKORO_VOICE/
# OLLAMA_MODEL exactly as before this feature existed.
PROVIDER_FILE = BASE_DIR / "provider_config.json"
DEFAULT_PROVIDER_CONFIG = {
    "llm_mode": "local",   # "local" | "cloud"
    "llm_cloud": {"base_url": "https://api.openai.com/v1", "api_key": "", "model": ""},
    "stt_mode": "local",   # "local" | "cloud"
    "stt_cloud": {"base_url": "https://api.openai.com/v1", "api_key": "", "model": "whisper-1"},
    "tts_mode": "local",   # "local" | "cloud"
    "tts_local_engine": "kokoro",
    "tts_cloud_engine": "elevenlabs",   # "elevenlabs" | "openai" | "veena" | "qwen3"
    "tts_cloud": {
        "elevenlabs": {"api_key": "", "voice_id": ""},
        "openai":     {"api_key": "", "voice": "alloy", "base_url": "https://api.openai.com/v1"},
        "veena":      {"endpoint_url": "", "api_key": "", "speaker": "kavya"},
        "qwen3":      {"endpoint_url": "http://127.0.0.1:8020/tts", "api_key": "", "language": "Auto",
                        "ref_audio": "", "ref_text": ""},
    },
}

def load_provider_config() -> dict:
    cfg = json.loads(json.dumps(DEFAULT_PROVIDER_CONFIG))  # deep copy of defaults
    if PROVIDER_FILE.exists():
        try:
            saved = json.loads(PROVIDER_FILE.read_text(encoding="utf-8"))
            cfg["llm_mode"] = saved.get("llm_mode", cfg["llm_mode"])
            cfg["llm_cloud"].update(saved.get("llm_cloud", {}))
            cfg["stt_mode"] = saved.get("stt_mode", cfg["stt_mode"])
            cfg["stt_cloud"].update(saved.get("stt_cloud", {}))
            cfg["tts_mode"] = saved.get("tts_mode", cfg["tts_mode"])
            cfg["tts_local_engine"] = saved.get("tts_local_engine", "kokoro")
            if saved.get("tts_mode") == "cloud" and saved.get("tts_cloud_engine") == "qwen3":
                cfg["tts_mode"] = "local"
                cfg["tts_local_engine"] = "qwen3"
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
    resp = httpx.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{cfg.get('voice_id', '')}",
        params={"output_format": "pcm_24000"},
        headers={"xi-api-key": cfg.get("api_key", ""), "Content-Type": "application/json"},
        json={"text": text},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.content

def synthesize_openai_tts(text: str, cfg: dict) -> bytes:
    base_url = (cfg.get("base_url") or "https://api.openai.com/v1").rstrip("/")
    resp = httpx.post(
        f"{base_url}/audio/speech",
        headers={"Authorization": f"Bearer {cfg.get('api_key', '')}"},
        json={"model": cfg.get("model") or "tts-1", "input": text, "voice": cfg.get("voice", "alloy"),
              "response_format": "pcm"},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.content

def synthesize_veena(text: str, cfg: dict) -> bytes:
    """Calls a self-hosted Veena TTS endpoint (see modal_veena_tts.py) — returns
    raw 16-bit PCM mono @ 24kHz, same as the other cloud engines. Timeout is
    longer than elevenlabs/openai to absorb Modal cold starts after idle."""
    resp = httpx.post(
        cfg.get("endpoint_url", "").rstrip("/"),
        json={"text": text, "speaker": cfg.get("speaker", "kavya"), "api_key": cfg.get("api_key", "")},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.content

def synthesize_qwen3(text: str, cfg: dict) -> bytes:
    """Call the companion Qwen3-TTS voice-cloning service. It returns raw signed
    16-bit mono PCM at 24 kHz, which is the format expected by the voice clients."""
    text = _normalize_for_speech(text)
    endpoint = (cfg.get("endpoint_url") or "http://127.0.0.1:8020/tts").rstrip("/")
    ref_audio = (cfg.get("ref_audio") or "").strip()
    voice_mode = cfg.get("voice_mode") or ("clone" if ref_audio else "preset")
    if voice_mode == "clone" and not ref_audio:
        raise ValueError("Qwen3-TTS reference audio is required")
    resp = httpx.post(
        endpoint,
        headers={"Authorization": f"Bearer {cfg.get('api_key', '')}"} if cfg.get("api_key") else {},
        json={"text": text, "language": cfg.get("language") or "Auto",
              "voice_mode": voice_mode, "speaker": cfg.get("speaker") or "Ryan",
              "ref_audio": ref_audio, "ref_text": (cfg.get("ref_text") or "").strip()},
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.content

_TTS_CLOUD_ENGINES = {"elevenlabs": synthesize_elevenlabs, "openai": synthesize_openai_tts,
                      "veena": synthesize_veena, "qwen3": synthesize_qwen3}

async def synthesize_active(text: str) -> bytes:
    """Dispatches to the active TTS provider — local Kokoro (existing synthesize(),
    unchanged) or the configured cloud engine. Runs the blocking call in a thread,
    same as how synthesize() was already invoked via run_in_executor before this
    dispatcher existed."""
    cfg = load_provider_config()
    loop = asyncio.get_event_loop()
    if cfg["tts_mode"] == "local" and cfg.get("tts_local_engine") == "qwen3":
        return await loop.run_in_executor(None, synthesize_qwen3, text, cfg["tts_cloud"].get("qwen3", {}))
    if cfg["tts_mode"] == "cloud":
        engine = cfg.get("tts_cloud_engine", "elevenlabs")
        fn = _TTS_CLOUD_ENGINES.get(engine)
        if fn is None:
            raise ValueError(f"Unknown TTS provider: {engine}")
        engine_cfg = cfg["tts_cloud"].get(engine, {})
        return await loop.run_in_executor(None, fn, text, engine_cfg)
    return await loop.run_in_executor(None, synthesize, text)

# ── LangGraph turn graph ─────────────────────────────────────────────────────
# Replaces the old inline if/elif turn logic (name pickup → farewell check →
# RAG lookup → prompt build → LLM call) with an explicit graph, so the shape
# of a conversational turn is visible as nodes/edges instead of buried in one
# long function body. Compiled once at import; every turn is a fresh
# `ainvoke()` with no checkpointer — voice_ws/chat_ws's own closures already
# carry conversation/caller state across turns exactly as they did before.
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

_TG_FAREWELL_RE = re.compile(
    r"\b(bye|goodbye|good ?bye|see you|take care|that'?s? ?(it|all)|"
    r"thank(s| you)( so much| very much)?|cheers|have a (good|great|nice) (day|one)|"
    r"no (more )?questions?|i('m| am) (done|good|all set|okay now|fixed)|all good|"
    r"nothing else|that will be all|end (the )?(call|session)|it'?s? working now|"
    r"problem solved|that fixed it)\b",
    re.IGNORECASE,
)
_TG_FAREWELL_RE_HI = re.compile(
    r"(धन्यवाद|शुक्रिया|अलविदा|बाय बाय|ठीक है बस|समस्या (हल|ठीक) हो गई|"
    r"काम कर रहा है|और कुछ नहीं|बस इतना ही|बहुत बढ़िया)"
)


class TurnState(TypedDict, total=False):
    transcript: str
    conversation: list[dict]          # history BEFORE this turn — this turn's
                                       # own transcript is NOT in it yet; the
                                       # caller appends it after seeing `action`
    caller_name: str | None
    caller_past_calls: list[dict]
    resolved_flag: bool | None
    channel: Literal["voice", "chat"]
    is_first_user_turn: bool          # True if no user turn has landed yet this call
    cfg: dict
    # node-local / output fields
    name_just_learned: bool
    farewell_matched: bool
    kb: dict
    prompt: str
    action: Literal["ignore_noise", "farewell", "reply"]
    reply_text: str


def _tg_pickup_name(state: TurnState) -> dict:
    if state.get("caller_name"):
        return {}
    found = extract_caller_name(state["transcript"])
    if not found:
        return {}
    rec = get_caller_record(found)
    past_calls = rec["calls"] if rec else []
    log.info("Picked up caller name: %s (%d prior call(s) on file)", found, len(past_calls))
    return {"caller_name": found, "caller_past_calls": past_calls, "name_just_learned": True}


def _tg_detect_farewell(state: TurnState) -> dict:
    transcript = state["transcript"]
    matched = bool(_TG_FAREWELL_RE.search(transcript) or _TG_FAREWELL_RE_HI.search(transcript))
    return {"farewell_matched": matched}


def _tg_route_after_farewell_check(state: TurnState) -> str:
    if state.get("farewell_matched"):
        if (state["channel"] == "voice" and is_whisper_filler(state["transcript"])
                and state.get("is_first_user_turn")):
            # A bare "Thank you." as the very first thing on the call is Whisper
            # filler far more often than a real goodbye, and hanging up on it
            # can't be undone. Treat it as noise and wait for a real turn.
            return "ignore_noise"
        return "farewell"
    return "rag_lookup"


def _tg_ignore_noise(state: TurnState) -> dict:
    log.info("Ignoring farewell-looking opening turn %r — treating as noise", state["transcript"])
    return {"action": "ignore_noise"}


def _tg_farewell(state: TurnState) -> dict:
    transcript = state["transcript"]
    cfg = state["cfg"]
    resolved = state.get("resolved_flag")
    if _RESOLVED_SIGNAL.search(transcript) or _RESOLVED_SIGNAL_HI.search(transcript):
        resolved = True
    reply_text = (cfg.get("farewell_message") or "").strip() or localized("farewell")
    return {"action": "farewell", "reply_text": reply_text, "resolved_flag": resolved}


def _tg_rag_lookup(state: TurnState) -> dict:
    log.info("STEP 3 ▶ Building LLM prompt + RAG context")
    return {"kb": kb_lookup(state["transcript"])}


def _tg_build_prompt(state: TurnState) -> dict:
    cfg = state["cfg"]
    kb = state["kb"]
    caller_name = state.get("caller_name")
    conversation = state["conversation"]
    transcript = state["transcript"]
    # Voice and chat share this node, but their existing prompt copy differs in
    # a few nouns — preserved verbatim per channel rather than unified.
    is_voice = state["channel"] == "voice"
    noun = "caller" if is_voice else "person"
    session_word = "call" if is_voice else "session"
    hindi_style = "spoken" if is_voice else "written"

    sys_prompt = cfg.get("system_prompt", DEFAULT_PROMPT_CONFIG["system_prompt"])
    guardrails = cfg.get("guardrails", [])
    if guardrails:
        sys_prompt += "\n\nGUARDRAILS:\n" + "\n".join(f"- {g}" for g in guardrails)
    sys_prompt += kb["section"]
    if caller_name:
        sys_prompt += (
            f"\n\nThe {noun}'s name is {caller_name} — you already have it, don't ask again. "
            f"Use their first name naturally now and then when you reply, not in every single sentence."
        )
        if state.get("name_just_learned"):
            sys_prompt += (
                f"\n\nThey just told you their name for the first time this {session_word}. Open your very next "
                f"reply with a short, warm \"{time_of_day_greeting()}, {caller_name.split()[0]}!\" (or a "
                f"close natural variant) before anything else, then continue straight into helping them."
            )
        if state.get("caller_past_calls"):
            # this_call_issue expects the current transcript included — conversation
            # here is pre-append, so add it as a local, non-mutating extra turn.
            sys_prompt += caller_history_section(
                caller_name, state["caller_past_calls"],
                this_call_issue(conversation + [{"role": "user", "content": transcript}]),
            )
    else:
        sys_prompt += (
            f"\n\nYou still don't have the {noun}'s name. Getting it comes before troubleshooting: if "
            "they jumped straight into describing the problem without giving it, this reply must ask "
            "for their name before or alongside anything else you say — a quick, casual \"and what's "
            "your name?\" or \"before we dig in, who am I speaking with?\" is enough. Don't let the "
            "conversation move into device details, model numbers, or troubleshooting steps while this "
            "is still unanswered — one missed chance to ask is fine, but don't let it go two replies "
            "in a row without asking again."
        )
    if CONVO_LANGUAGE == "hi":
        sys_prompt += (
            "\n\nIMPORTANT: Respond ONLY in Hindi, written in the Devanagari script — regardless of "
            f"the language the instructions above are written in, and even if the {noun} mixes in some "
            f"English words. Keep it natural, {hindi_style} Hindi, not a stiff word-for-word translation."
        )
    sys_prompt += THIS_CALL_ONLY_RULES

    # A wider window than the 7 turns this used to carry: the model was asking
    # how steps went that it had never actually given, because the steps it
    # *had* given had already scrolled out of the prompt. `conversation` is
    # pre-append here (unlike the old code's post-append list + [-20:-1]), so
    # the equivalent slice is the last 19 turns with no exclusion needed.
    history = ""
    for m in conversation[-19:]:
        role = "Customer" if m["role"] == "user" else "Assistant"
        history += f"{role}: {m['content']}\n"
    prompt = f"{sys_prompt}\n\nConversation so far on THIS call:\n{history}Customer: {transcript}\nAssistant:"
    return {"prompt": prompt}


async def _tg_call_llm(state: TurnState) -> dict:
    cfg = state["cfg"]
    fallback = cfg.get("fallback_message", DEFAULT_PROMPT_CONFIG["fallback_message"])
    _llm_mode = load_provider_config()["llm_mode"]
    log.info("STEP 4 ▶ Sending prompt to LLM (%s)...", "cloud" if _llm_mode == "cloud" else OLLAMA_MODEL)
    tl = time.time()
    try:
        reply = await generate_llm_reply(state["prompt"], ["\nCustomer:", "\nAssistant:", "\nUser:"])
    except Exception as e:
        log.error("STEP 4 ▶ LLM error: %s", e)
        reply = fallback
    reply = re.split(r"\n\s*(?:Customer|Assistant|User)\s*:", reply)[0].strip()
    if not reply:
        reply = fallback
    log.info("STEP 5 ▶ LLM reply (%.2fs): %s", time.time() - tl, reply)
    return {"action": "reply", "reply_text": reply}


_turn_graph_builder = StateGraph(TurnState)
_turn_graph_builder.add_node("pickup_name", _tg_pickup_name)
_turn_graph_builder.add_node("detect_farewell", _tg_detect_farewell)
_turn_graph_builder.add_node("ignore_noise", _tg_ignore_noise)
_turn_graph_builder.add_node("farewell", _tg_farewell)
_turn_graph_builder.add_node("rag_lookup", _tg_rag_lookup)
_turn_graph_builder.add_node("build_prompt", _tg_build_prompt)
_turn_graph_builder.add_node("call_llm", _tg_call_llm)
_turn_graph_builder.set_entry_point("pickup_name")
_turn_graph_builder.add_edge("pickup_name", "detect_farewell")
_turn_graph_builder.add_conditional_edges("detect_farewell", _tg_route_after_farewell_check, {
    "ignore_noise": "ignore_noise", "farewell": "farewell", "rag_lookup": "rag_lookup",
})
_turn_graph_builder.add_edge("rag_lookup", "build_prompt")
_turn_graph_builder.add_edge("build_prompt", "call_llm")
_turn_graph_builder.add_edge("ignore_noise", END)
_turn_graph_builder.add_edge("farewell", END)
_turn_graph_builder.add_edge("call_llm", END)
turn_graph = _turn_graph_builder.compile()

# ── WebSocket ─────────────────────────────────────────────────────────────────
INACTIVITY_PROMPT_SECS = 20

@app.websocket("/ws/voice")
async def voice_ws(ws: WebSocket):
    if BOT_KIND == "chat":
        await ws.close(code=1008, reason="This bot is text-chat only — use /ws/chat")
        return
    await ws.accept()
    log.info("Client connected")

    call_ts        = int(time.time())
    call_recorder  = CallRecorder(LOG_DIR / f"call_{call_ts}.wav")
    call_started_at = datetime.now().isoformat()

    cfg        = load_prompt_config()
    conversation: list[dict] = []
    caller_name: str | None = None   # picked up from speech once mentioned; no formal ask-name step
    caller_past_calls: list[dict] = []   # this caller's prior calls, looked up once the name is known
    resolved_flag: bool | None = None    # whether *this* call's issue got resolved, inferred at farewell
    processing = asyncio.Lock()
    loop       = asyncio.get_event_loop()
    missed_turns = 0        # consecutive turns with no speech in them

    # ── Inactivity tracking ───────────────────────────────────────────────────
    last_activity   = time.time()
    nudge_count     = 0
    session_closed  = asyncio.Event()
    nudges_disabled = False

    # ── Barge-in state ────────────────────────────────────────────────────────
    interrupted       = asyncio.Event()
    barge_in_pending  = None

    def touch(reset_nudges: bool = True):
        nonlocal last_activity, nudge_count
        last_activity = time.time()
        if reset_nudges:
            nudge_count = 0

    async def say(text: str, msg_type: str = "reply", _nudge: bool = False) -> float:
        """Speaks a line and returns its audio duration in seconds (0 if skipped
        by a barge-in or a dead connection) — callers that need to wait for
        playback to actually finish on the client (farewell, timeout) use this
        instead of guessing a fixed delay."""
        if session_closed.is_set() or ws.client_state != WebSocketState.CONNECTED:
            return 0.0
        duration = 0.0
        try:
            log.info("🤖 BOT  said: %s", text)
            await ws.send_json({"type": msg_type, "text": text})
            ts = time.time()
            pcm = await synthesize_active(text)
            save_wav(pcm, msg_type)
            call_recorder.write(pcm, source_rate=SAMPLE_RATE)
            if interrupted.is_set():
                log.info("TTS done (%.2fs) but BARGE-IN active — skipping audio playback", time.time() - ts)
                await ws.send_json({"type": "barge_in_ack"})
            else:
                duration = len(pcm) / 2 / SAMPLE_RATE  # int16 mono PCM
                log.info("TTS synthesized (%.2fs): %.1f KB (%.1fs of audio) → sending to client",
                         time.time() - ts, len(pcm) / 1024, duration)
                await ws.send_bytes(pcm)
            await ws.send_json({"type": "turn_complete"})
        except (WebSocketDisconnect, RuntimeError):
            session_closed.set()
            return 0.0
        conversation.append({"role": "assistant", "content": text})
        touch(reset_nudges=not _nudge)
        return duration

    async def end_session(reason: str):
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
        nonlocal nudge_count
        MAX_NUDGES = 3
        nudge_messages = localized("nudges")

        while not session_closed.is_set():
            await asyncio.sleep(5)
            if nudges_disabled:
                continue
            idle = time.time() - last_activity
            if processing.locked():
                continue
            if idle >= INACTIVITY_PROMPT_SECS and nudge_count < MAX_NUDGES:
                prompt = nudge_messages[nudge_count]
                nudge_count += 1
                log.info("Inactivity nudge %d/%d sent (idle %.0fs)", nudge_count, MAX_NUDGES, idle)
                await say(prompt, "reply", _nudge=True)

                if nudge_count >= MAX_NUDGES:
                    await asyncio.sleep(INACTIVITY_PROMPT_SECS)
                    if not session_closed.is_set():
                        log.info("Session closed after %d nudges with no response.", MAX_NUDGES)
                        audio_secs = await say(localized("timeout"), "reply", _nudge=True)
                        await asyncio.sleep(audio_secs + 1.0)
                        await end_session("inactivity_timeout")
                        return

    watcher_task = asyncio.create_task(inactivity_watcher())

    # ── Greeting ──────────────────────────────────────────────────────────────
    greeting = cfg.get("greeting", DEFAULT_PROMPT_CONFIG["greeting"])
    await ws.send_json({"type": "status", "msg": "Preparing greeting..."})
    pcm = await synthesize_active(greeting)
    save_wav(pcm, "greeting")
    call_recorder.write(pcm, source_rate=SAMPLE_RATE)
    await ws.send_bytes(pcm)
    await ws.send_json({"type": "turn_complete"})
    conversation.append({"role": "assistant", "content": greeting})

    async def process_audio(raw: bytes):
        nonlocal barge_in_pending, caller_name, caller_past_calls, resolved_flag, missed_turns
        interrupted.clear()
        call_recorder.write(raw, source_rate=16000)  # caller's mic audio, always 16kHz over the wire
        kb = len(raw) / 1024
        log.info("STEP 1 ▶ Audio received from client: %.1f KB (%d bytes)", kb, len(raw))
        has_speech, why = speech_check(raw)
        if not has_speech:
            # Never send this to Whisper: it would invent a caption, and the one
            # it invents most often reads as a goodbye.
            log.info("STEP 1 ▶ Ignoring turn — %s", why)
            missed_turns += 1
            await say(repeat_line(missed_turns))
            await ws.send_json({"type": "turn_complete"})
            return
        await ws.send_json({"type": "status", "msg": f"Received {kb:.1f} KB — transcribing..."})
        t0 = time.time()
        transcript = await transcribe(raw)
        dt = time.time() - t0
        if not transcript:
            log.info("STEP 2 ▶ Whisper returned EMPTY transcript (%.2fs) — likely silence/too short", dt)
            missed_turns += 1
            await say(repeat_line(missed_turns))
            await ws.send_json({"type": "turn_complete"})
            return
        missed_turns = 0
        log.info("STEP 2 ▶ Whisper transcript (%.2fs): %r", dt, transcript)
        await ws.send_json({"type": "transcript", "text": transcript})
        await ws.send_json({"type": "status", "msg": "Thinking..."})

        result = await turn_graph.ainvoke({
            "transcript": transcript,
            "conversation": conversation,
            "caller_name": caller_name,
            "caller_past_calls": caller_past_calls,
            "resolved_flag": resolved_flag,
            "channel": "voice",
            "is_first_user_turn": _user_turns(conversation) == 0,
            "cfg": cfg,
        })

        if result["action"] == "ignore_noise":
            missed_turns += 1
            await say(repeat_line(missed_turns))
            await ws.send_json({"type": "turn_complete"})
            return

        conversation.append({"role": "user", "content": transcript})
        caller_name = result.get("caller_name", caller_name)
        caller_past_calls = result.get("caller_past_calls", caller_past_calls)

        if result["action"] == "farewell":
            resolved_flag = result.get("resolved_flag", resolved_flag)
            audio_secs = await say(result["reply_text"])
            # Wait for the farewell line to actually finish playing on the client, plus a
            # 1s grace period, before closing — a flat 0.5s regardless of message length
            # was cutting longer sign-offs off mid-sentence.
            wait_secs = audio_secs + 1.0
            log.info("Farewell detected — closing session in %.1fs (%.1fs audio + 1s grace).",
                     wait_secs, audio_secs)
            await asyncio.sleep(wait_secs)
            await end_session("farewell")
            return

        await say(result["reply_text"])

    try:
        while not session_closed.is_set():
            msg = await ws.receive()

            if msg.get("type") == "websocket.disconnect":
                log.info("Client disconnected (received disconnect message)")
                break

            if "bytes" in msg and msg["bytes"]:
                touch()
                log.info("STEP 0 ▶ WS received %d bytes from client (processing busy=%s)",
                         len(msg["bytes"]), processing.locked())
                if processing.locked():
                    barge_in_pending = msg["bytes"]
                    interrupted.set()
                    log.info("Barge-in: queuing new audio, signalling interrupt")
                    await ws.send_json({"type": "status", "msg": "Got it — I'll respond shortly."})
                    continue

                async with processing:
                    await process_audio(msg["bytes"])
                    if barge_in_pending:
                        pending = barge_in_pending
                        barge_in_pending = None
                        log.info("Processing queued barge-in audio")
                        await process_audio(pending)

            elif "text" in msg and msg["text"]:
                data = json.loads(msg["text"])
                if data.get("type") == "reset":
                    conversation.clear()
                    caller_name = None
                    caller_past_calls = []
                    resolved_flag = None
                    touch()
                    await ws.send_json({"type": "status", "msg": "Session reset."})
                    await say(localized("reset_greeting"), "reply")
                elif data.get("type") == "barge_in":
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
        try:
            (LOG_DIR / f"call_{call_ts}.json").write_text(
                json.dumps({
                    "call_id": call_ts,
                    "started": call_started_at,
                    "ended": datetime.now().isoformat(),
                    "caller_name": caller_name,
                    "turns": conversation,
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            log.warning("Failed to save call transcript: %s", e)
        if caller_name and any(m["role"] == "user" for m in conversation):
            issue_summary = None
            try:
                convo_text = "\n".join(
                    f"{'Customer' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                    for m in conversation
                )
                summary_prompt = (
                    "Summarize in under 12 words what technical issue the caller was dealing "
                    "with in this support call. Respond with the issue only — no preamble, no "
                    "quotes, no trailing period.\n\n" + convo_text
                )
                issue_summary = (await generate_llm_reply(summary_prompt)).strip().strip('"').rstrip(".")
            except Exception as e:
                log.warning("Issue summarization failed: %s", e)
            if not issue_summary:
                first_user = next((m["content"] for m in conversation if m["role"] == "user"), "")
                issue_summary = first_user[:80] or "Unspecified issue"
            try:
                record_caller_call(caller_name, call_ts, issue_summary, resolved_flag)
            except Exception as e:
                log.warning("Failed to save caller history: %s", e)
        watcher_task.cancel()
        try:
            await watcher_task
        except asyncio.CancelledError:
            pass


# ── Chat (text-only bots, and available as a lighter option on voice bots too) ─
# Mirrors voice_ws's prompt-building (KB retrieval, guardrails, caller-name
# memory, farewell detection, language, transcript persistence) so a chat
# bot's admin panel behaves the same as a voice bot's wherever that's still
# relevant — it just skips everything audio-specific (STT, TTS, barge-in,
# inactivity nudges have no clean text-chat equivalent, so they're dropped
# rather than half-ported).
@app.websocket("/ws/chat")
async def chat_ws(ws: WebSocket):
    await ws.accept()
    log.info("Chat client connected")

    call_ts = int(time.time())
    call_started_at = datetime.now().isoformat()

    cfg = load_prompt_config()
    conversation: list[dict] = []
    caller_name: str | None = None
    caller_past_calls: list[dict] = []
    resolved_flag: bool | None = None

    greeting = cfg.get("greeting", DEFAULT_PROMPT_CONFIG["greeting"])
    await ws.send_json({"type": "reply", "text": greeting})
    conversation.append({"role": "assistant", "content": greeting})

    async def process_message(text: str) -> bool:
        """Returns True if the session should close after this turn (farewell)."""
        nonlocal caller_name, caller_past_calls, resolved_flag

        await ws.send_json({"type": "status", "msg": "Thinking..."})
        result = await turn_graph.ainvoke({
            "transcript": text,
            "conversation": conversation,
            "caller_name": caller_name,
            "caller_past_calls": caller_past_calls,
            "resolved_flag": resolved_flag,
            "channel": "chat",
            "is_first_user_turn": _user_turns(conversation) == 0,
            "cfg": cfg,
        })

        conversation.append({"role": "user", "content": text})
        caller_name = result.get("caller_name", caller_name)
        caller_past_calls = result.get("caller_past_calls", caller_past_calls)

        if result["action"] == "farewell":
            resolved_flag = result.get("resolved_flag", resolved_flag)
            conversation.append({"role": "assistant", "content": result["reply_text"]})
            await ws.send_json({"type": "reply", "text": result["reply_text"]})
            await ws.send_json({"type": "session_ended", "reason": "farewell"})
            log.info("Farewell detected — closing chat session.")
            return True

        conversation.append({"role": "assistant", "content": result["reply_text"]})
        await ws.send_json({"type": "reply", "text": result["reply_text"]})
        return False

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                log.info("Chat client disconnected (received disconnect message)")
                break
            if "text" not in msg or not msg["text"]:
                continue
            data = json.loads(msg["text"])
            if data.get("type") == "message":
                text = (data.get("text") or "").strip()
                if not text:
                    continue
                should_close = await process_message(text)
                if should_close:
                    await asyncio.sleep(0.3)
                    try:
                        await ws.close()
                    except RuntimeError:
                        pass
                    break
            elif data.get("type") == "reset":
                conversation.clear()
                caller_name = None
                caller_past_calls = []
                resolved_flag = None
                reset_greeting = localized("reset_greeting")
                conversation.append({"role": "assistant", "content": reset_greeting})
                await ws.send_json({"type": "reply", "text": reset_greeting})
    except WebSocketDisconnect:
        log.info("Chat client disconnected")
    except Exception as e:
        log.error("Chat WS error: %s", e, exc_info=True)
    finally:
        try:
            (LOG_DIR / f"call_{call_ts}.json").write_text(
                json.dumps({
                    "call_id": call_ts,
                    "channel": "chat",
                    "started": call_started_at,
                    "ended": datetime.now().isoformat(),
                    "caller_name": caller_name,
                    "turns": conversation,
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            log.warning("Failed to save chat transcript: %s", e)
        if caller_name and any(m["role"] == "user" for m in conversation):
            issue_summary = None
            try:
                convo_text = "\n".join(
                    f"{'Customer' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                    for m in conversation
                )
                summary_prompt = (
                    "Summarize in under 12 words what the person was asking about in this chat. "
                    "Respond with the topic only — no preamble, no quotes, no trailing period.\n\n" + convo_text
                )
                issue_summary = (await generate_llm_reply(summary_prompt)).strip().strip('"').rstrip(".")
            except Exception as e:
                log.warning("Issue summarization failed: %s", e)
            if not issue_summary:
                first_user = next((m["content"] for m in conversation if m["role"] == "user"), "")
                issue_summary = first_user[:80] or "Unspecified topic"
            try:
                record_caller_call(caller_name, call_ts, issue_summary, resolved_flag)
            except Exception as e:
                log.warning("Failed to save caller history: %s", e)
