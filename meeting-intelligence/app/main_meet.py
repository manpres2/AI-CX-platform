"""
main_meet.py - Local AI Meeting Intelligence Platform (Phase 1: pipeline first)
FastAPI: Whisper STT + pyannote diarization + Ollama/cloud LLM extraction + ChromaDB search
Admin panel at: http://localhost:8002/admin  (login required)
Run from app/ folder: uvicorn main_meet:app --host 0.0.0.0 --port 8002

Sibling to the BFSI bank bot (main.py, port 8000) and tech-support bot
(main_tech.py, port 8001). Same repo conventions (HTTPBasic admin auth,
JSON-file config, BASE_DIR resolution), but no live call/voice loop — this
app processes uploaded/pointed-to meeting recordings, not live audio, so its
whole product surface is the admin dashboard (no public index.html).

Phase 1 scope: upload -> transcribe -> diarize -> extract tasks/decisions ->
store -> search -> dashboard. Live Microsoft Teams capture and outbound
reminder sending are deferred (see Live_meeting_capture.md §13).
"""

import json
import logging
import os
import platform
import re
import secrets
import shutil
import sys
import time
import uuid
import wave
from datetime import datetime
from pathlib import Path

import httpx
import torch
import whisper
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

import db
import diarization
import extraction

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
_this_file = Path(__file__).resolve()
_env_path = _this_file.parent.parent.parent / ".env"
if _env_path.exists():
    try:
        load_dotenv(_env_path, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(_env_path, encoding="utf-16")

# whisper shells out to a binary literally named "ffmpeg"/"ffmpeg.exe" on PATH;
# this machine has no system ffmpeg install. The imageio-ffmpeg pip package
# bundles a static binary, but under a versioned filename (e.g.
# ffmpeg-win-x86_64-v7.1.exe) that Windows won't resolve as "ffmpeg" even with
# its directory on PATH — so copy it once to bin/ffmpeg.exe and put that on PATH.
try:
    import imageio_ffmpeg
    _bundled_ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    _bin_dir = _this_file.parent.parent / "bin"
    _bin_dir.mkdir(exist_ok=True)
    _local_ffmpeg = _bin_dir / "ffmpeg.exe"
    if not _local_ffmpeg.exists():
        shutil.copy2(_bundled_ffmpeg, _local_ffmpeg)
    os.environ["PATH"] = str(_bin_dir) + os.pathsep + os.environ.get("PATH", "")
except ImportError:
    pass

BASE_DIR       = _this_file.parent.parent
STATIC_DIR     = BASE_DIR / "static_meet"
UPLOADS_DIR    = BASE_DIR / "uploads_meet"
LOGS_DIR       = BASE_DIR / "logs_meet"
TRANSCRIPTS_STORE = BASE_DIR / "transcripts_store_meet"
RUNTIME_FILE   = BASE_DIR / "runtime_config_meet.json"

for d in [STATIC_DIR, UPLOADS_DIR, LOGS_DIR, TRANSCRIPTS_STORE]:
    d.mkdir(exist_ok=True)

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "apexbank2026")

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "meeting_transcripts"

DEFAULT_RUNTIME_CONFIG = {
    "whisper_model": "small",
    "whisper_device": "cpu",
    "diarization_device": "cpu",
}

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = BASE_DIR / "server_meet.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger("meeting-intelligence")
SERVER_START = time.time()


def load_runtime_config() -> dict:
    cfg = dict(DEFAULT_RUNTIME_CONFIG)
    if RUNTIME_FILE.exists():
        try:
            cfg.update(json.loads(RUNTIME_FILE.read_text(encoding="utf-8")))
        except Exception as e:
            log.warning("Could not load runtime_config_meet.json: %s", e)
    return cfg


def save_runtime_config(cfg: dict):
    RUNTIME_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


_runtime_config = load_runtime_config()

# ── Whisper (lazy load) ──────────────────────────────────────────────────────
_whisper_model = None
_whisper_model_name = None


def get_whisper():
    global _whisper_model, _whisper_model_name
    cfg = load_runtime_config()
    name = cfg.get("whisper_model", "small")
    device = cfg.get("whisper_device", "cpu")
    if _whisper_model is None or _whisper_model_name != name:
        log.info("Loading Whisper (%s) on %s...", name, device)
        _whisper_model = whisper.load_model(name, device=device)
        _whisper_model_name = name
        log.info("Whisper ready.")
    return _whisper_model


# ── ChromaDB + embeddings (lazy) ─────────────────────────────────────────────
_chroma_client = None
_meeting_collection = None
_embedder = None


def init_rag():
    global _chroma_client, _meeting_collection, _embedder
    if not RAG_AVAILABLE:
        log.warning("chromadb/sentence-transformers not available — search disabled.")
        return
    try:
        log.info("Loading embedding model (%s)...", EMBED_MODEL)
        _embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
        _chroma_client = chromadb.PersistentClient(path=str(TRANSCRIPTS_STORE))
        _meeting_collection = _chroma_client.get_or_create_collection(COLLECTION_NAME)
        log.info("RAG ready. %d chunks indexed.", _meeting_collection.count())
    except Exception as e:
        log.error("Failed to init RAG: %s", e)


def chunk_segments(segments: list[dict], window: int = 6, overlap: int = 2) -> list[dict]:
    """Groups consecutive transcript segments into overlapping windows so each
    chunk keeps enough context for embedding, while preserving speaker/timestamp
    metadata (unlike plain word-window chunking, which loses per-chunk speaker
    attribution) — same word-window spirit as app/index_kb.py's chunk_text()."""
    chunks = []
    i = 0
    while i < len(segments):
        group = segments[i:i + window]
        if not group:
            break
        text = " ".join(f"{s['speaker']}: {s['text']}" for s in group)
        chunks.append({
            "text": text,
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "speakers": sorted({s["speaker"] for s in group}),
        })
        i += window - overlap
    return chunks


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Meeting Intelligence Platform")
security = HTTPBasic()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
async def _startup():
    db.init_db()
    init_rag()


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    ok_user = secrets.compare_digest(credentials.username, ADMIN_USER)
    ok_pass = secrets.compare_digest(credentials.password, ADMIN_PASS)
    if not (ok_user and ok_pass):
        raise HTTPException(status_code=401, detail="Unauthorized",
                             headers={"WWW-Authenticate": "Basic"})
    return credentials.username


@app.get("/")
async def root():
    return HTMLResponse("<meta http-equiv='refresh' content='0; url=/admin'>")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "whisper": _whisper_model_name or load_runtime_config().get("whisper_model"),
        "meetings": db.count_meetings(),
        "rag": _meeting_collection.count() if _meeting_collection else "not loaded",
        "cuda": torch.cuda.is_available(),
    }


@app.get("/api/sysinfo")
async def sysinfo():
    uptime_s = int(time.time() - SERVER_START)
    h, m, s = uptime_s // 3600, (uptime_s % 3600) // 60, uptime_s % 60
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "meeting_count": db.count_meetings(),
        "open_tasks": db.count_open_tasks(),
        "overdue_tasks": db.count_overdue_tasks(),
        "uptime": f"{h}h {m}m {s}s",
    }


# ── Admin: page ──────────────────────────────────────────────────────────────
@app.get("/admin")
async def admin_panel(username: str = Depends(verify_admin)):
    admin_html = STATIC_DIR / "admin.html"
    if admin_html.exists():
        return HTMLResponse(admin_html.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Place admin.html in static_meet/ folder.</h2>")


# ── Admin: meetings ──────────────────────────────────────────────────────────
@app.get("/admin/api/meetings")
async def list_meetings(username: str = Depends(verify_admin)):
    return {"meetings": db.list_meetings()}


@app.get("/admin/api/meetings/{meeting_id}")
async def get_meeting(meeting_id: int, username: str = Depends(verify_admin)):
    meeting = db.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(404, "Meeting not found")
    transcript = []
    if meeting.get("transcript_json_path"):
        p = Path(meeting["transcript_json_path"])
        if p.exists():
            transcript = json.loads(p.read_text(encoding="utf-8")).get("segments", [])
    return {
        "meeting": meeting,
        "transcript": transcript,
        "participants": db.list_participants(meeting_id),
        "tasks": db.list_tasks(meeting_id),
        "decisions": db.list_decisions(meeting_id),
    }


@app.delete("/admin/api/meetings/{meeting_id}")
async def delete_meeting(meeting_id: int, username: str = Depends(verify_admin)):
    meeting = db.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(404, "Meeting not found")
    db.delete_meeting(meeting_id)
    if meeting.get("transcript_json_path"):
        Path(meeting["transcript_json_path"]).unlink(missing_ok=True)
    if meeting.get("source_filename"):
        (UPLOADS_DIR / meeting["source_filename"]).unlink(missing_ok=True)
    try:
        if _meeting_collection:
            existing = _meeting_collection.get(where={"meeting_id": meeting_id})
            if existing and existing.get("ids"):
                _meeting_collection.delete(ids=existing["ids"])
    except Exception as e:
        log.warning("Could not clean up Chroma entries for meeting %s: %s", meeting_id, e)
    return {"status": "deleted"}


@app.post("/admin/api/meetings/upload")
async def upload_meeting(file: UploadFile = File(...), username: str = Depends(verify_admin)):
    safe_name = f"{uuid.uuid4().hex}_{file.filename}"
    dest = UPLOADS_DIR / safe_name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    cfg = load_runtime_config()
    meeting_id = db.insert_meeting(
        title=file.filename, source_filename=safe_name,
        uploaded_at=datetime.now().isoformat(), whisper_model=cfg.get("whisper_model", "small"),
    )
    log.info("Meeting %d uploaded: %s", meeting_id, file.filename)
    return {"meeting_id": meeting_id, "status": "uploaded"}


def _wav_duration_secs(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


@app.post("/admin/api/meetings/{meeting_id}/process")
async def process_meeting(meeting_id: int, username: str = Depends(verify_admin)):
    meeting = db.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(404, "Meeting not found")
    if meeting["status"] == "processing":
        raise HTTPException(409, "Already processing")

    db.update_meeting_status(meeting_id, "processing")
    src_path = UPLOADS_DIR / meeting["source_filename"]
    cfg = load_runtime_config()

    try:
        # STEP 1 — transcribe
        log.info("Meeting %d: transcribing %s", meeting_id, src_path.name)
        model = get_whisper()
        result = model.transcribe(str(src_path), verbose=False)
        whisper_segments = result.get("segments", [])
        if not whisper_segments:
            raise RuntimeError("Whisper returned no segments — empty or unsupported audio.")

        # STEP 2 — diarize + align
        speaker_turns = []
        try:
            log.info("Meeting %d: diarizing", meeting_id)
            speaker_turns = diarization.diarize(src_path, device=cfg.get("diarization_device", "cpu"))
        except diarization.DiarizationUnavailable as e:
            log.warning("Meeting %d: diarization unavailable (%s) — using single 'SPEAKER_00' fallback", meeting_id, e)

        def label_for(seg_start: float, seg_end: float) -> str:
            if not speaker_turns:
                return "SPEAKER_00"
            best_label, best_overlap = "SPEAKER_00", 0.0
            for turn in speaker_turns:
                overlap = min(seg_end, turn["end"]) - max(seg_start, turn["start"])
                if overlap > best_overlap:
                    best_overlap, best_label = overlap, turn["speaker"]
            return best_label

        segments = [{
            "speaker": label_for(seg["start"], seg["end"]),
            "start": seg["start"], "end": seg["end"], "text": seg["text"].strip(),
        } for seg in whisper_segments]

        # STEP 3 — write transcript sidecar
        transcript_path = LOGS_DIR / f"meeting_{meeting_id}.json"
        transcript_path.write_text(json.dumps({
            "meeting_id": meeting_id, "title": meeting["title"],
            "segments": segments, "created_at": datetime.now().isoformat(),
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        # STEP 4 — participants (talk time / turn count per speaker)
        talk = {}
        for s in segments:
            entry = talk.setdefault(s["speaker"], {"secs": 0.0, "turns": 0})
            entry["secs"] += max(0.0, s["end"] - s["start"])
            entry["turns"] += 1
        for speaker, stats in talk.items():
            db.insert_participant(meeting_id, speaker, stats["secs"], stats["turns"])

        # STEP 5 — chunk + embed for search
        if _meeting_collection and _embedder:
            chunks = chunk_segments(segments)
            if chunks:
                texts = [c["text"] for c in chunks]
                ids = [f"meeting{meeting_id}_{i:04d}" for i in range(len(chunks))]
                metas = [{"meeting_id": meeting_id, "start": c["start"], "end": c["end"],
                          "speakers": ",".join(c["speakers"])} for c in chunks]
                embeddings = _embedder.encode(texts, show_progress_bar=False).tolist()
                _meeting_collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metas)

        # STEP 6 — LLM extraction
        log.info("Meeting %d: extracting tasks/decisions", meeting_id)
        transcript_text = "\n".join(
            f"[{int(s['start'] // 60):02d}:{int(s['start'] % 60):02d}] {s['speaker']}: {s['text']}"
            for s in segments
        )
        extracted = await extraction.extract_tasks_decisions(transcript_text)
        for t in extracted.get("tasks", []):
            db.insert_task(meeting_id, t.get("task", "").strip() or "Unspecified task",
                            t.get("owner"), t.get("deadline"), t.get("priority", "medium"), None)
        for dec in extracted.get("decisions", []):
            db.insert_decision(meeting_id, dec.get("decision", "").strip() or "Unspecified decision",
                                dec.get("timestamp_secs"), None)
        if extracted.get("_parse_error"):
            log.warning("Meeting %d: extraction parse error: %s", meeting_id, extracted["_parse_error"])

        # STEP 7 — finish
        duration = segments[-1]["end"] if segments else _wav_duration_secs(src_path)
        db.finish_meeting(meeting_id, duration, len(talk), str(transcript_path), datetime.now().isoformat())
        log.info("Meeting %d: done (%d segments, %d speakers, %d tasks, %d decisions)",
                  meeting_id, len(segments), len(talk), len(extracted.get("tasks", [])), len(extracted.get("decisions", [])))
        return {"status": "done"}

    except Exception as e:
        log.error("Meeting %d: processing failed: %s", meeting_id, e, exc_info=True)
        db.update_meeting_status(meeting_id, "failed", str(e))
        raise HTTPException(500, f"Processing failed: {e}")


# ── Admin: participants ──────────────────────────────────────────────────────
@app.post("/admin/api/participants/{participant_id}/rename")
async def rename_participant(participant_id: int, data: dict, username: str = Depends(verify_admin)):
    name = (data.get("display_name") or "").strip()
    if not name:
        raise HTTPException(400, "display_name required")
    db.rename_participant(participant_id, name)
    return {"status": "renamed"}


# ── Admin: tasks ─────────────────────────────────────────────────────────────
@app.get("/admin/api/tasks/due")
async def tasks_due(username: str = Depends(verify_admin)):
    return {"tasks": db.list_tasks_due()}


@app.post("/admin/api/tasks/{task_id}/status")
async def set_task_status(task_id: int, data: dict, username: str = Depends(verify_admin)):
    status = data.get("status", "open")
    if status not in ("open", "done"):
        raise HTTPException(400, "status must be 'open' or 'done'")
    db.set_task_status(task_id, status)
    return {"status": "updated"}


# ── Admin: search ────────────────────────────────────────────────────────────
@app.get("/admin/api/search")
async def search(q: str, top_k: int = 5, username: str = Depends(verify_admin)):
    if not _meeting_collection or not _embedder or not q.strip():
        return {"results": []}
    try:
        qvec = _embedder.encode([q]).tolist()
        res = _meeting_collection.query(query_embeddings=qvec, n_results=top_k)
        results = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0] if res.get("distances") else [None] * len(docs)
        for doc, meta, dist in zip(docs, metas, dists):
            meeting = db.get_meeting(meta.get("meeting_id"))
            results.append({
                "meeting_id": meta.get("meeting_id"),
                "meeting_title": meeting["title"] if meeting else "Unknown meeting",
                "snippet": doc[:400],
                "start": meta.get("start"),
                "speakers": meta.get("speakers"),
                "score": round(1 - dist, 3) if dist is not None else None,
            })
        return {"results": results}
    except Exception as e:
        log.warning("Search failed: %s", e)
        return {"results": [], "error": str(e)}


# ── Admin: runtime config (whisper/diarization device+model) ────────────────
@app.get("/admin/api/runtime-config")
async def get_runtime_config(username: str = Depends(verify_admin)):
    return load_runtime_config()


@app.post("/admin/api/runtime-config")
async def save_runtime_config_api(data: dict, username: str = Depends(verify_admin)):
    cfg = load_runtime_config()
    cfg.update({k: v for k, v in data.items() if k in DEFAULT_RUNTIME_CONFIG})
    save_runtime_config(cfg)
    log.info("Runtime config updated: %s", cfg)
    return {"status": "saved", **cfg}


# ── Admin: LLM provider settings (independent from the two voice bots) ──────
def _mask_provider_config(cfg: dict) -> dict:
    out = json.loads(json.dumps(cfg))
    out["llm_cloud"]["has_key"] = bool(out["llm_cloud"].pop("api_key", ""))
    return out


@app.get("/admin/api/providers")
async def get_providers(username: str = Depends(verify_admin)):
    cfg = _mask_provider_config(extraction.load_provider_config())
    cfg["available_local_models"] = await extraction.list_ollama_models()
    return cfg


@app.post("/admin/api/providers")
async def save_providers(data: dict, username: str = Depends(verify_admin)):
    cfg = extraction.load_provider_config()
    cfg["llm_mode"] = data.get("llm_mode", cfg["llm_mode"])
    incoming_local = data.get("llm_local", {})
    if incoming_local.get("ollama_model"):
        cfg["llm_local"]["ollama_model"] = incoming_local["ollama_model"]
    incoming_cloud = data.get("llm_cloud", {})
    cfg["llm_cloud"]["base_url"] = incoming_cloud.get("base_url", cfg["llm_cloud"]["base_url"])
    cfg["llm_cloud"]["model"] = incoming_cloud.get("model", cfg["llm_cloud"]["model"])
    if incoming_cloud.get("api_key"):
        cfg["llm_cloud"]["api_key"] = incoming_cloud["api_key"]
    extraction.save_provider_config(cfg)
    log.info("Provider config updated: llm_mode=%s", cfg["llm_mode"])
    return {"status": "saved"}


@app.post("/admin/api/providers/test-llm")
async def test_llm(data: dict, username: str = Depends(verify_admin)):
    mode = data.get("llm_mode") or extraction.load_provider_config()["llm_mode"]
    try:
        if mode == "cloud":
            saved = extraction.load_provider_config()["llm_cloud"]
            cfg = {
                "base_url": data.get("base_url") or saved.get("base_url", ""),
                "model": data.get("model") or saved.get("model", ""),
                "api_key": data.get("api_key") or saved.get("api_key", ""),
            }
            reply = await extraction.generate_cloud_reply("Say OK if you can hear me.", cfg)
        else:
            model = data.get("ollama_model") or extraction.load_provider_config()["llm_local"]["ollama_model"]
            reply = await extraction.generate_local_reply("Say OK if you can hear me.", model)
        return {"reply": reply}
    except Exception as e:
        return {"error": str(e)}
