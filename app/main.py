"""
main.py — Voice AI BFSI Demo (with RAG)
FastAPI WebSocket server: Whisper STT → ChromaDB RAG → Ollama LLaMA → Kokoro TTS
Place at: C:/AIManpres2/app/main.py
Run from app/ folder: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import logging
import time
import wave
from pathlib import Path

import numpy as np
import torch
import whisper
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import httpx
from kokoro import KPipeline

# RAG imports (optional — gracefully degrades if KB not built yet)
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
# Load .env with encoding fallback (Windows Notepad saves UTF-16 by default)
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    try:
        load_dotenv(_env_path, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(_env_path, encoding="utf-16")

BASE_DIR   = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"
LOG_DIR    = BASE_DIR / "logs"
KB_STORE   = BASE_DIR / "kb_store"
LOG_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

OLLAMA_URL    = "http://localhost:8080/v1/chat/completions"
OLLAMA_MODEL  = "llama3.1:8b"       # Change if you pulled a different model
WHISPER_MODEL = "base"               # "small" for better accuracy (slower)
KOKORO_VOICE  = "af_heart"
SAMPLE_RATE   = 24000
EMBED_MODEL   = "all-MiniLM-L6-v2"
COLLECTION    = "apex_bank_bfsi"
RAG_TOP_K     = 3                    # Number of KB chunks to retrieve

BFSI_SYSTEM_PROMPT = """You are a helpful voice assistant for Apex Bank, a fictional Indian bank.
You assist customers on a phone IVR system.
Keep every answer to 2-3 SHORT sentences maximum — this is a voice call, not text.
Never use bullet points, markdown, or numbered lists.
Speak in a warm, professional, conversational tone.
If the customer asks something outside banking, politely say you can only help with banking queries.
Always be factual — only state what is in the context provided."""

GREETING = "Thank you for calling Apex Bank. This is your AI assistant. How may I help you today?"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Model loading ─────────────────────────────────────────────────────────────
log.info("Loading Whisper (%s)…", WHISPER_MODEL)
stt_model = whisper.load_model(WHISPER_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
log.info("Whisper on %s", "CUDA" if torch.cuda.is_available() else "CPU")

log.info("Loading Kokoro TTS…")
tts_pipeline = KPipeline(lang_code="a")
log.info("Kokoro ready.")

# ── RAG setup ─────────────────────────────────────────────────────────────────
chroma_client = None
kb_collection = None
embedder      = None

if RAG_AVAILABLE and KB_STORE.exists() and any(KB_STORE.iterdir()):
    try:
        log.info("Loading ChromaDB from %s…", KB_STORE)
        chroma_client = chromadb.PersistentClient(path=str(KB_STORE))
        kb_collection = chroma_client.get_collection(COLLECTION)
        log.info("Loading sentence-transformer (%s)…", EMBED_MODEL)
        embedder = SentenceTransformer(EMBED_MODEL)
        log.info("RAG ready. Collection has %d chunks.", kb_collection.count())
    except Exception as e:
        log.warning("RAG init failed (%s) — running without KB.", e)
else:
    log.warning("KB store not found or empty — run app/index_kb.py first. Running without RAG.")

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="BFSI Voice AI Demo")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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


# ── Helpers ───────────────────────────────────────────────────────────────────

def pcm_to_numpy(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


async def transcribe(pcm_bytes: bytes) -> str:
    audio = pcm_to_numpy(pcm_bytes)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: stt_model.transcribe(audio, language="en", fp16=torch.cuda.is_available())
    )
    return result["text"].strip()


def retrieve_kb(query: str, top_k: int = RAG_TOP_K) -> str:
    """Query ChromaDB and return relevant context as a single string."""
    if not kb_collection or not embedder:
        return ""
    try:
        qvec = embedder.encode([query]).tolist()
        results = kb_collection.query(query_embeddings=qvec, n_results=top_k)
        docs = results.get("documents", [[]])[0]
        return "\n\n".join(docs)
    except Exception as e:
        log.warning("RAG query failed: %s", e)
        return ""


async def llm_generate(conversation: list[dict], query: str) -> str:
    """Retrieve KB context, call llama.cpp OpenAI-compatible API."""
    context = retrieve_kb(query)
    system_msg = BFSI_SYSTEM_PROMPT
    if context:
        system_msg += f"\n\nRelevant bank information:\n{context}"

    messages = [{"role": "system", "content": system_msg}]
    for msg in conversation[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


def synthesize(text: str) -> bytes:
    chunks = []
    for _, _, audio in tts_pipeline(text, voice=KOKORO_VOICE, speed=1.0):
        if audio is not None:
            chunks.append(audio)
    if not chunks:
        return b""
    combined = np.clip(np.concatenate(chunks), -1.0, 1.0)
    return (combined * 32767).astype(np.int16).tobytes()


def save_wav(pcm: bytes, label: str = "out"):
    ts = int(time.time())
    p = LOG_DIR / f"{label}_{ts}.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/voice")
async def voice_ws(ws: WebSocket):
    await ws.accept()
    log.info("Client connected")
    conversation: list[dict] = []

    # Greeting
    await ws.send_json({"type": "status", "msg": "Preparing greeting…"})
    loop = asyncio.get_event_loop()
    pcm = await loop.run_in_executor(None, synthesize, GREETING)
    save_wav(pcm, "greeting")
    await ws.send_bytes(pcm)
    conversation.append({"role": "assistant", "content": GREETING})

    try:
        while True:
            msg = await ws.receive()

            if "bytes" in msg and msg["bytes"]:
                raw = msg["bytes"]
                log.info("Audio received: %d bytes", len(raw))

                # STT
                await ws.send_json({"type": "status", "msg": "Transcribing…"})
                transcript = await transcribe(raw)
                if not transcript:
                    await ws.send_json({"type": "status", "msg": "Couldn't hear clearly — please try again."})
                    continue
                log.info("Transcript: %s", transcript)
                await ws.send_json({"type": "transcript", "text": transcript})
                conversation.append({"role": "user", "content": transcript})

                # LLM + RAG
                await ws.send_json({"type": "status", "msg": "Thinking…"})
                try:
                    reply = await llm_generate(conversation, transcript)
                except Exception as e:
                    log.error("LLM error: %s", e)
                    reply = "I'm sorry, I'm having trouble right now. Please hold for a moment."
                if not reply:
                    reply = "I'm sorry, could you please repeat that?"
                log.info("Reply: %s", reply)
                await ws.send_json({"type": "reply", "text": reply})
                conversation.append({"role": "assistant", "content": reply})

                # TTS
                await ws.send_json({"type": "status", "msg": "Generating speech…"})
                pcm = await loop.run_in_executor(None, synthesize, reply)
                save_wav(pcm, "reply")
                await ws.send_bytes(pcm)

            elif "text" in msg and msg["text"]:
                data = json.loads(msg["text"])
                if data.get("type") == "reset":
                    conversation.clear()
                    await ws.send_json({"type": "status", "msg": "Conversation reset."})

    except WebSocketDisconnect:
        log.info("Client disconnected")
    except Exception as e:
        log.error("WS error: %s", e, exc_info=True)
