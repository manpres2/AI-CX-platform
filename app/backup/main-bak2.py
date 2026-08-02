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

BASE_DIR   = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"
LOG_DIR    = BASE_DIR / "logs"
KB_DOCS    = BASE_DIR / "kb_docs"
KB_STORE   = BASE_DIR / "kb_store"
PROMPT_FILE = BASE_DIR / "prompt_config.json"

for d in [LOG_DIR, STATIC_DIR, KB_DOCS, KB_STORE]:
    d.mkdir(exist_ok=True)

OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
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
        "Always be factual — only state what is in the context provided."
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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

# ── Core helpers ──────────────────────────────────────────────────────────────
def pcm_to_numpy(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

async def transcribe(pcm_bytes: bytes) -> str:
    audio = pcm_to_numpy(pcm_bytes)
    loop  = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: stt_model.transcribe(audio, language="en",
                                            fp16=torch.cuda.is_available()))
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

async def llm_generate(conversation: list[dict], query: str) -> str:
    cfg         = load_prompt_config()
    system_msg  = cfg.get("system_prompt", DEFAULT_PROMPT_CONFIG["system_prompt"])
    guardrails  = cfg.get("guardrails", [])
    if guardrails:
        system_msg += "\n\nGUARDRAILS — strictly follow these rules:\n"
        system_msg += "\n".join(f"- {g}" for g in guardrails)

    context = retrieve_kb(query)
    if context:
        system_msg += f"\n\nRelevant bank information:\n{context}"

    history = ""
    for msg in conversation[-6:]:
        role     = "Customer" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    prompt = f"{system_msg}\n\nConversation:\n{history}Assistant:"

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

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
@app.websocket("/ws/voice")
async def voice_ws(ws: WebSocket):
    await ws.accept()
    log.info("Client connected")
    conversation: list[dict] = []
    cfg      = load_prompt_config()
    greeting = cfg.get("greeting", DEFAULT_PROMPT_CONFIG["greeting"])
    fallback = cfg.get("fallback_message", DEFAULT_PROMPT_CONFIG["fallback_message"])
    processing = asyncio.Lock()   # prevents overlapping turns

    await ws.send_json({"type": "status", "msg": "Preparing greeting..."})
    loop = asyncio.get_event_loop()
    pcm  = await loop.run_in_executor(None, synthesize, greeting)
    save_wav(pcm, "greeting")
    await ws.send_bytes(pcm)
    conversation.append({"role": "assistant", "content": greeting})

    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"]:
                # Drop the frame silently if we're already mid-turn
                if processing.locked():
                    log.info("Ignoring audio — previous turn still processing")
                    await ws.send_json({"type": "status", "msg": "Please wait — still processing previous response."})
                    continue

                async with processing:
                    raw = msg["bytes"]
                    await ws.send_json({"type": "status", "msg": "Transcribing..."})
                    transcript = await transcribe(raw)
                    if not transcript:
                        await ws.send_json({"type": "status", "msg": "Couldn't hear clearly — please try again."})
                        continue
                    log.info("Transcript: %s", transcript)
                    await ws.send_json({"type": "transcript", "text": transcript})
                    conversation.append({"role": "user", "content": transcript})

                    await ws.send_json({"type": "status", "msg": "Thinking..."})
                    try:
                        reply = await llm_generate(conversation, transcript)
                    except Exception as e:
                        log.error("LLM error: %s", e)
                        reply = fallback
                    if not reply:
                        reply = fallback
                    log.info("Reply: %s", reply)
                    await ws.send_json({"type": "reply", "text": reply})
                    conversation.append({"role": "assistant", "content": reply})

                    await ws.send_json({"type": "status", "msg": "Generating speech..."})
                    pcm = await loop.run_in_executor(None, synthesize, reply)
                    save_wav(pcm, "reply")
                    await ws.send_bytes(pcm)
                    # Signal browser the turn is fully complete
                    await ws.send_json({"type": "turn_complete"})

            elif "text" in msg and msg["text"]:
                data = json.loads(msg["text"])
                if data.get("type") == "reset":
                    conversation.clear()
                    cfg      = load_prompt_config()
                    greeting = cfg.get("greeting", DEFAULT_PROMPT_CONFIG["greeting"])
                    await ws.send_json({"type": "status", "msg": "Conversation reset."})

    except WebSocketDisconnect:
        log.info("Client disconnected")
    except Exception as e:
        log.error("WS error: %s", e, exc_info=True)
