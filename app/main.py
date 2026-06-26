"""
main.py - Voice AI BFSI Demo with Admin Panel
FastAPI WebSocket: Whisper STT + ChromaDB RAG + Ollama LLaMA + Kokoro TTS
Admin panel at: http://localhost:8000/admin  (login required)
Run from app/ folder: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import logging
import os
import platform
import shutil
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import whisper
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
import httpx
from kokoro import KPipeline
import secrets

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

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
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
KOKORO_VOICE  = os.getenv("KOKORO_VOICE", "af_heart")
SAMPLE_RATE   = 24000
EMBED_MODEL   = "all-MiniLM-L6-v2"
COLLECTION    = "apex_bank_bfsi"
RAG_TOP_K     = 3

# Admin credentials — set in .env or fall back to defaults
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "apexbank2026")

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

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    ok_user = secrets.compare_digest(credentials.username, ADMIN_USER)
    ok_pass = secrets.compare_digest(credentials.password, ADMIN_PASS)
    if not (ok_user and ok_pass):
        raise HTTPException(status_code=401, detail="Unauthorized",
                            headers={"WWW-Authenticate": "Basic"})
    return credentials.username

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

@app.get("/admin/api/kb/files")
async def list_kb_files(username: str = Depends(verify_admin)):
    files = []
    for f in sorted(KB_DOCS.glob("*.txt")):
        files.append({
            "name": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
            "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%d %b %Y %H:%M"),
            "lines": len(f.read_text(encoding="utf-8", errors="ignore").splitlines()),
        })
    return {"files": files, "count": len(files)}

@app.post("/admin/api/kb/upload")
async def upload_kb_file(file: UploadFile = File(...), username: str = Depends(verify_admin)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(400, "Only .txt files allowed")
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
    return {"content": p.read_text(encoding="utf-8", errors="ignore")}

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

        if embedder is None:
            embedder = SentenceTransformer(EMBED_MODEL)

        doc_files = list(KB_DOCS.glob("*.txt"))
        if not doc_files:
            return {"status": "error", "message": "No .txt files in kb_docs/"}

        all_chunks, all_ids, all_metas = [], [], []
        for doc_path in doc_files:
            text = doc_path.read_text(encoding="utf-8", errors="ignore")
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

        embeddings = embedder.encode(all_chunks, show_progress_bar=False).tolist()
        collection.add(documents=all_chunks, embeddings=embeddings,
                       ids=all_ids, metadatas=all_metas)

        # Reload global references
        chroma_client = client
        kb_collection = collection
        log.info("KB rebuilt: %d chunks from %d files", len(all_chunks), len(doc_files))
        return {"status": "ok", "chunks": len(all_chunks), "files": len(doc_files)}
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

# ── Core helpers ──────────────────────────────────────────────────────────────
def pcm_to_numpy(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

WHISPER_NAME_HINT = (
    "Apex Bank customer service call. Customer name is Manpreet Singh. "
    "Topics include savings accounts, fixed deposits, home loans, personal loans, "
    "credit cards, KYC, net banking, insurance, and mutual funds."
)

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

def synthesize(text: str) -> bytes:
    chunks = [a for _, _, a in tts_pipeline(text, voice=KOKORO_VOICE) if a is not None]
    if not chunks:
        return b""
    combined = np.clip(np.concatenate(chunks), -1.0, 1.0)
    return (combined * 32767).astype(np.int16).tobytes()

def save_wav(pcm: bytes, label: str = "out"):
    ts = int(time.time())
    p  = LOG_DIR / f"{label}_{ts}.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)

# ── WebSocket ─────────────────────────────────────────────────────────────────
# ── Session state helper ─────────────────────────────────────────────────────
def make_session():
    """Fresh session state for each WebSocket connection."""
    return {
        "stage":         "ask_name",  # ask_name | ask_pin | verified
        "customer_name": None,        # name from KB record (verified)
        "spoken_name":   None,        # name the caller gave us (may not match KB)
        "customer_id":   None,
        "pin_attempts":  0,
        "name_attempts": 0,
        "customer_data": None,
    }

import re as _re

def find_customer(name_or_id: str) -> dict | None:
    """Search ChromaDB for a customer record matching name or ID."""
    if not kb_collection or not embedder:
        return None
    try:
        # Try the full lookup first, then fallback to individual words (first/last name)
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

        needle_words = [w for w in name_or_id.lower().split() if len(w) > 2]
        for doc in all_docs:
            doc_lower = doc.lower()
            # Require ALL name words to appear in the doc (prevents wrong-surname matches)
            if needle_words and all(w in doc_lower for w in needle_words):
                m = _re.search(r"(?:User Profile|Customer Name|Name)\s*[:\-]\s*([A-Za-z]+(?:\s+[A-Za-z]+)?)", doc, _re.IGNORECASE)
                real_name = m.group(1).strip() if m else name_or_id.strip().title()
                log.info("Customer matched: %s → %s", name_or_id, real_name)
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


INACTIVITY_PROMPT_SECS = 20   # seconds of silence before each "still here?" nudge

@app.websocket("/ws/voice")
async def voice_ws(ws: WebSocket):
    await ws.accept()
    log.info("Client connected")

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

    # ── Barge-in state ────────────────────────────────────────────────────────
    interrupted       = asyncio.Event()   # set when user speaks over bot
    barge_in_pending  = None              # raw PCM bytes queued during processing

    def touch(reset_nudges: bool = True):
        nonlocal last_activity, nudge_count
        last_activity = time.time()
        if reset_nudges:
            nudge_count = 0   # reset only when customer actually speaks

    async def say(text: str, msg_type: str = "reply", _nudge: bool = False):
        await ws.send_json({"type": msg_type, "text": text})
        pcm = await loop.run_in_executor(None, synthesize, text)
        save_wav(pcm, msg_type)
        if interrupted.is_set():
            await ws.send_json({"type": "barge_in_ack"})
        else:
            await ws.send_bytes(pcm)
        await ws.send_json({"type": "turn_complete"})
        conversation.append({"role": "assistant", "content": text})
        # After a nudge, restart the silence clock but keep the nudge count
        touch(reset_nudges=not _nudge)

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
            idle = time.time() - last_activity
            if processing.locked():
                continue  # bot is working — don't count that as customer silence
            if idle >= INACTIVITY_PROMPT_SECS and nudge_count < MAX_NUDGES:
                stage = session["stage"]
                stage_key = stage if stage in ("ask_name", "ask_pin") else "verified"
                prompt = nudge_messages[nudge_count][stage_key]
                nudge_count += 1
                log.info("Inactivity nudge %d/%d sent (idle %.0fs)", nudge_count, MAX_NUDGES, idle)
                await say(prompt, "reply", _nudge=True)

                if nudge_count >= MAX_NUDGES:
                    # All nudges exhausted — wait one more interval then close
                    await asyncio.sleep(INACTIVITY_PROMPT_SECS)
                    if not session_closed.is_set():
                        log.info("Session closed after %d nudges with no response.", MAX_NUDGES)
                        await say(
                            "We haven't been able to reach you after several attempts. "
                            "Your session has now ended for security. "
                            "Thank you for calling Apex Bank — please call us back when you're ready. Goodbye.",
                            "reply", _nudge=True
                        )
                        session_closed.set()
                        await ws.close()
                        return

    watcher_task = asyncio.create_task(inactivity_watcher())

    # ── Greeting ──────────────────────────────────────────────────────────────
    greeting = "Hi there, thanks for calling Apex Bank! I'm your AI assistant. Could I get your name or Customer ID to get started?"
    await ws.send_json({"type": "status", "msg": "Preparing greeting..."})
    pcm = await loop.run_in_executor(None, synthesize, greeting)
    save_wav(pcm, "greeting")
    await ws.send_bytes(pcm)
    await ws.send_json({"type": "turn_complete"})
    conversation.append({"role": "assistant", "content": greeting})

    async def process_audio(raw: bytes):
        """Transcribe + stage machine + reply. Must be called inside the processing lock."""
        nonlocal barge_in_pending
        interrupted.clear()
        await ws.send_json({"type": "status", "msg": "Transcribing..."})
        transcript = await transcribe(raw)
        if not transcript:
            await ws.send_json({"type": "status", "msg": "Couldn't hear clearly — please try again."})
            await ws.send_json({"type": "turn_complete"})
            return
        # Redact any 4-digit sequences from the transcript log when in PIN stage
        log_transcript = _re.sub(r'\b\d{4}\b', '****', transcript) if session.get("stage") == "ask_pin" else transcript
        log.info("Transcript: %s", log_transcript)
        await ws.send_json({"type": "transcript", "text": transcript})
        conversation.append({"role": "user", "content": transcript})

        # ── Greeting detection (runs before stage machine, only at ask_name) ───
        if session["stage"] == "ask_name":
            _salutation_only = _re.fullmatch(
                r"(hi+|hey+|hello+|howdy|good\s*(morning|afternoon|evening)|"
                r"how are you( doing)?|how'?s? it going|what'?s up)[.!?,\s]*",
                transcript.strip(), _re.IGNORECASE
            )
            if _salutation_only:
                await say("Hey! Good to hear from you. Could you share your full name or Customer ID so I can pull up your account?")
                return

        # ── Farewell detection (runs before stage machine) ────────────────────
        _farewell = _re.search(
            r"\b(bye|goodbye|good ?bye|see you|take care|that'?s? ?(it|all)|"
            r"thank(s| you)( so much| very much)?|cheers|have a (good|great|nice) (day|one)|"
            r"no (more )?questions?|i('m| am) (done|good|all set|okay now)|all good|"
            r"nothing else|that will be all|end (the )?(call|session))\b",
            transcript, _re.IGNORECASE
        )
        if _farewell:
            name = (session.get("customer_name") or session.get("spoken_name") or "").split()
            first = name[0].title() if name else ""
            farewell_reply = (
                f"It was a pleasure helping you{', ' + first if first else ''}! "
                f"Have a wonderful day. Goodbye!"
            )
            await say(farewell_reply)
            log.info("Farewell detected — closing session.")
            session_closed.set()
            await asyncio.sleep(0.5)
            await ws.close()
            return

        stage = session["stage"]

        if stage == "ask_name":
            await ws.send_json({"type": "status", "msg": "Looking up customer..."})

            # ── "On behalf of / for somebody else" intent ─────────────────────
            _behalf_match = _re.search(
                r"\b(for somebody|for someone|for another|on behalf of|"
                r"checking for|balance for|account for|calling for)\b",
                transcript, _re.IGNORECASE
            )
            # Also catch "for [Name]" directly — e.g. "I need the account for Jasmeet Chauhan"
            _for_name = _re.search(
                r"\bfor\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", transcript
            )
            if _for_name:
                # They've given the name directly — use it as the lookup
                lookup_text = _for_name.group(1).strip()
                spoken_first = lookup_text.split()[0].title()
                session["spoken_name"] = spoken_first
                session["name_attempts"] = session.get("name_attempts", 0) + 1
                only_first_name = len(lookup_text.split()) == 1
                if only_first_name:
                    await say(f"Sure! Could I also get {spoken_first}'s last name to find the right account?")
                    return
                customer = find_customer(lookup_text)
                if customer:
                    session["customer_name"] = customer.get("name", lookup_text)
                    session["customer_data"] = customer
                    session["stage"]         = "ask_pin"
                    session["name_attempts"] = 0
                    first = session["customer_name"].split()[0].title()
                    await say(f"Got it — I found {first}'s account. For security, could you provide their 4-digit phone banking PIN?")
                else:
                    await say(f"I wasn't able to find an account for {lookup_text}. Could you double-check the full name or share their Customer ID?")
                return
            elif _behalf_match:
                await say("Of course! Could you give me the full name or Customer ID of the account holder you'd like to check?")
                return

            # Extract name from natural speech like "my name is X" or "I'm X"
            _name_match = _re.search(
                r"(?:my name is|i(?:'m| am)|this is|it['']s|name['\s]*s?)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
                transcript, _re.IGNORECASE
            )
            # Words that are never names — greetings, filler, affirmations, farewells
            _NON_NAMES = {
                "hi","hey","hello","bye","goodbye","yes","no","okay","ok","sure","thanks",
                "thank","please","sorry","right","alright","yep","nope","hmm","um","uh",
                "what","how","when","why","who","can","could","would","should","is","are",
                "it","the","a","an","and","or","but","so","good","fine","great","nice",
                "just","really","actually","maybe","perhaps","well","now","then","there",
            }
            # Also treat a short transcript that looks purely like a name (1-3 alpha words)
            _plain_name = _re.fullmatch(r"([A-Za-z]+(?:\s+[A-Za-z]+){0,2})[.!?,]*", transcript.strip())

            if _name_match:
                lookup_text = _name_match.group(1).strip()
            elif _plain_name:
                words = _plain_name.group(1).strip().split()
                # Only treat as a name if all words are >= 3 chars and none are filler
                if all(len(w) >= 3 and w.lower() not in _NON_NAMES for w in words):
                    lookup_text = _plain_name.group(1).strip()
                else:
                    lookup_text = None
            else:
                lookup_text = None  # no name detected — don't attempt a lookup

            if not lookup_text:
                # Nothing that looks like a name — respond naturally
                greet_words = {"hi","hey","hello"}
                is_greeting = any(w in greet_words for w in transcript.lower().split())
                if is_greeting:
                    await say("Hey there! To get started, could you tell me your full name or Customer ID?")
                else:
                    await say("I didn't quite catch your name. Could you tell me your full name or Customer ID?")
                return

            spoken_first = lookup_text.split()[0].title()
            session["spoken_name"] = spoken_first
            session["name_attempts"] = session.get("name_attempts", 0) + 1

            only_first_name = len(lookup_text.split()) == 1

            # Always ask for last name if only one word given — don't attempt a DB lookup yet
            if only_first_name:
                reply = (f"Nice to meet you, {spoken_first}! I just need your last name too — "
                         f"could you give me your full name so I can find your account?")
                await say(reply)
                return

            customer = find_customer(lookup_text)
            if customer:
                session["customer_name"] = customer.get("name", lookup_text)
                session["customer_data"] = customer
                session["stage"]         = "ask_pin"
                session["name_attempts"] = 0
                first = session["customer_name"].split()[0].title()
                reply = (f"Got it, {first} — nice to have you with us. "
                         f"Just a quick security check: could you tell me your 4-digit phone banking PIN?")
            else:
                attempts = session["name_attempts"]
                if attempts == 1:
                    reply = (f"Thanks, {spoken_first}. I wasn't able to find an account under that name. "
                             f"Could you double-check the spelling, or share your Customer ID if you have it handy?")
                elif attempts == 2:
                    reply = (f"I'm still not finding a match, {spoken_first}. "
                             f"Your Customer ID would help me locate you right away — it's usually on your bank card or welcome letter.")
                else:
                    reply = (f"I'm sorry, {spoken_first}, I haven't been able to locate an account with those details. "
                             f"Please call us on 1800-123-4567 or visit your nearest branch and we'll get you sorted. Thanks for calling Apex Bank.")
            await say(reply)

        elif stage == "ask_pin":
            digits = _re.sub(r"\D", "", transcript)
            first  = (session["customer_name"] or session.get("spoken_name") or "").split()[0].title()
            if len(digits) >= 4:
                pin = digits[:4]
                if check_pin(session["customer_data"], pin):
                    session["stage"]        = "verified"
                    session["pin_attempts"] = 0
                    reply = (f"PIN verified — you're all set, {first}. "
                             f"How can I help you today?")
                else:
                    session["pin_attempts"] += 1
                    if session["pin_attempts"] >= 3:
                        reply = (f"I'm sorry {first}, we've had three unsuccessful PIN attempts. "
                                 f"For your security, please reset your PIN through the Apex Bank app, "
                                 f"call us back on 1800-123-4567, or visit your nearest branch. "
                                 f"Thanks for calling — take care.")
                        session["stage"] = "ask_name"
                        session["customer_data"] = None
                        session_closed.set()
                    else:
                        remaining = 3 - session["pin_attempts"]
                        reply = (f"That PIN didn't match, {first}. "
                                 f"You have {remaining} tr{'y' if remaining == 1 else 'ies'} remaining — please try again.")
            else:
                reply = f"I just need your 4-digit phone banking PIN, {first}. Go ahead whenever you're ready."
            await say(reply)

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
                return

            await ws.send_json({"type": "status", "msg": "Thinking..."})
            # Strip PIN from customer record before sending to LLM
            raw_record = session["customer_data"].get("record", "")
            customer_context = _re.sub(r"(?:Phone Banking )?PIN\s*[:\-]\s*\d{4}", "[PIN REDACTED]", raw_record, flags=_re.IGNORECASE)
            kb_context       = retrieve_kb(transcript)
            combined_context = f"VERIFIED CUSTOMER RECORD:\n{customer_context}"
            if kb_context:
                combined_context += f"\n\nBANK PRODUCT INFORMATION:\n{kb_context}"

            first_name = (session["customer_name"] or "").strip().split()[0].title()
            sys_prompt = cfg.get("system_prompt", DEFAULT_PROMPT_CONFIG["system_prompt"])
            guardrails = cfg.get("guardrails", [])
            if guardrails:
                sys_prompt += "\n\nGUARDRAILS:\n" + "\n".join(f"- {g}" for g in guardrails)
            sys_prompt += (
                f"\n\nCONTEXT (verified customer data):\n{combined_context}"
                f"\n\nIMPORTANT: The customer ({first_name}) is ALREADY verified. "
                f"Do NOT repeat the greeting, ask for their name or PIN again, mention the PIN, "
                f"or explain any verification logic. Never reveal or repeat any PIN digits. "
                f"Answer their question directly and naturally using only the customer data above. "
                f"Always write currency amounts as 'Rupees X' — never use 'Rs.' or 'INR' as text-to-speech will mispronounce them. "
                f"Never start or end a response with 'Thank you for calling', 'Thank you for contacting', or any similar phrase — just answer the question."
            )

            verified_turns = [m for m in conversation if m.get("role") == "user" or
                              (m.get("role") == "assistant" and
                               "verified" in m.get("content", "").lower())]
            history = ""
            for m in verified_turns[-6:]:
                role    = "Customer" if m["role"] == "user" else "Assistant"
                # Redact any 4-digit sequences from history so LLM never sees the PIN
                content = _re.sub(r'\b\d{4}\b', '[PIN]', m["content"])
                history += f"{role}: {content}\n"
            prompt = f"{sys_prompt}\n\nConversation:\n{history}Customer: {transcript}\nAssistant:"

            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(
                        OLLAMA_URL,
                        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
                    )
                    resp.raise_for_status()
                    reply = resp.json().get("response", "").strip()
            except Exception as e:
                log.error("LLM error: %s", e)
                reply = fallback
            if not reply:
                reply = fallback
            log.info("Reply: %s", reply)
            await say(reply)

    try:
        while not session_closed.is_set():
            msg = await ws.receive()

            if "bytes" in msg and msg["bytes"]:
                touch()   # reset inactivity timer on every audio chunk received
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

    except WebSocketDisconnect:
        log.info("Client disconnected")
    except Exception as e:
        log.error("WS error: %s", e, exc_info=True)
    finally:
        session_closed.set()
        watcher_task.cancel()
        try:
            await watcher_task
        except asyncio.CancelledError:
            pass
