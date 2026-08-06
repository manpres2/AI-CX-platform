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
import shutil
import sys
import time
import uuid
import wave
from datetime import datetime
from pathlib import Path

import docx
import fitz
import httpx
import torch
import whisper
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

import auth
import db
import diarization
import emailer
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

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

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

# Multi-user admin auth — shared users.db at the repo root (one level up from
# BASE_DIR here). See app/auth.py for the shared implementation (duplicated per app).
auth.configure(BASE_DIR.parent / "users.db", app_key="meet")
auth.ensure_bootstrap_user(ADMIN_USER, ADMIN_PASS)
verify_admin = auth.verify_admin


@app.on_event("startup")
async def _startup():
    db.init_db()
    init_rag()


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


def _enrich_task_owner(task: dict) -> dict:
    """Resolves a task's raw (LLM-extracted) owner string to the participant's
    real display_name — falling back to the raw string if no participant
    matches — and surfaces whether an email is on file so the frontend can
    gate the "Remind" button on it instead of failing after the click."""
    participant = db.find_participant_by_name(task["meeting_id"], task["owner"]) if task.get("owner") else None
    task["owner_display"] = (participant["display_name"] or participant["speaker_label"]) if participant else task.get("owner")
    task["owner_email"] = participant["email"] if participant else None
    return task


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
        "tasks": [_enrich_task_owner(t) for t in db.list_tasks(meeting_id)],
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
        source_type="audio",
    )
    log.info("Meeting %d uploaded (audio): %s", meeting_id, file.filename)
    return {"meeting_id": meeting_id, "status": "uploaded"}


def _wav_duration_secs(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


async def finish_processing(meeting_id: int, title: str, segments: list[dict], fallback_duration: float = 0.0):
    """The generic tail of meeting processing — write transcript sidecar,
    derive participants, chunk+embed for search, extract tasks/decisions,
    mark done. Works on any `segments` list regardless of source, so both
    the audio pipeline (after transcribe+diarize) and the transcript-upload
    path (after parsing) call this same function."""
    # write transcript sidecar
    transcript_path = LOGS_DIR / f"meeting_{meeting_id}.json"
    transcript_path.write_text(json.dumps({
        "meeting_id": meeting_id, "title": title,
        "segments": segments, "created_at": datetime.now().isoformat(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # participants (talk time / turn count per speaker)
    talk = {}
    for s in segments:
        entry = talk.setdefault(s["speaker"], {"secs": 0.0, "turns": 0})
        entry["secs"] += max(0.0, s["end"] - s["start"])
        entry["turns"] += 1
    participant_ids = {
        speaker: db.insert_participant(meeting_id, speaker, stats["secs"], stats["turns"])
        for speaker, stats in talk.items()
    }

    # auto-fill a participant's email if they stated one themselves in the
    # transcript (e.g. an intro line) — never overwrites an admin-entered one
    emailed_speakers = set()
    for s in segments:
        speaker = s["speaker"]
        if speaker in emailed_speakers:
            continue
        m = _EMAIL_RE.search(s["text"])
        if m:
            db.set_participant_email_if_empty(participant_ids[speaker], m.group(0))
            emailed_speakers.add(speaker)

    # chunk + embed for search
    if _meeting_collection and _embedder:
        chunks = chunk_segments(segments)
        if chunks:
            texts = [c["text"] for c in chunks]
            ids = [f"meeting{meeting_id}_{i:04d}" for i in range(len(chunks))]
            metas = [{"meeting_id": meeting_id, "start": c["start"], "end": c["end"],
                      "speakers": ",".join(c["speakers"])} for c in chunks]
            embeddings = _embedder.encode(texts, show_progress_bar=False).tolist()
            _meeting_collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metas)

    # LLM extraction
    log.info("Meeting %d: extracting tasks/decisions", meeting_id)
    transcript_text = "\n".join(
        f"[{int(s['start'] // 60):02d}:{int(s['start'] % 60):02d}] {s['speaker']}: {s['text']}"
        for s in segments
    )
    def _norm(v):
        """Small local models occasionally emit the literal string "null"
        instead of JSON null for an unknown field — treat it the same way."""
        return None if v is None or str(v).strip().lower() in ("", "null", "none") else v

    extracted = await extraction.extract_tasks_decisions(transcript_text, speakers=sorted(talk.keys()))
    for t in extracted.get("tasks", []):
        db.insert_task(meeting_id, t.get("task", "").strip() or "Unspecified task",
                        _norm(t.get("owner")), _norm(t.get("deadline")), t.get("priority", "medium"), None)
    for dec in extracted.get("decisions", []):
        db.insert_decision(meeting_id, dec.get("decision", "").strip() or "Unspecified decision",
                            dec.get("timestamp_secs"), None)
    if extracted.get("_parse_error"):
        log.warning("Meeting %d: extraction parse error: %s", meeting_id, extracted["_parse_error"])

    # finish
    duration = segments[-1]["end"] if segments else fallback_duration
    db.finish_meeting(meeting_id, duration, len(talk), str(transcript_path), datetime.now().isoformat())
    log.info("Meeting %d: done (%d segments, %d speakers, %d tasks, %d decisions)",
              meeting_id, len(segments), len(talk), len(extracted.get("tasks", [])), len(extracted.get("decisions", [])))


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

        await finish_processing(meeting_id, meeting["title"], segments, fallback_duration=_wav_duration_secs(src_path))
        return {"status": "done"}

    except Exception as e:
        log.error("Meeting %d: processing failed: %s", meeting_id, e, exc_info=True)
        db.update_meeting_status(meeting_id, "failed", str(e))
        raise HTTPException(500, f"Processing failed: {e}")


# ── Transcript upload (bypasses Whisper/diarization entirely) ───────────────
_TIMESTAMP_RE = r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})"
_CUE_LINE_RE = re.compile(_TIMESTAMP_RE + r"\s*-->\s*" + _TIMESTAMP_RE)
_SPEAKER_PREFIX_RE = re.compile(r"^([A-Za-z][\w .'-]{0,40}):\s*(.*)$", re.DOTALL)
_TEAMS_TURN_RE = re.compile(
    r"^\s*([A-Za-z][\w .'-]*?)\s{2,}(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\s*\n(.+)$", re.DOTALL
)


def _cue_bounds(m: re.Match) -> tuple[float, float]:
    def to_secs(h, mi, s, ms):
        return (int(h) if h else 0) * 3600 + int(mi) * 60 + int(s) + int(ms) / 1000.0
    return to_secs(*m.group(1, 2, 3, 4)), to_secs(*m.group(5, 6, 7, 8))


def _parse_vtt(raw: str) -> list[dict]:
    voice_re = re.compile(r"<v\s+([^>]+)>(.*?)(?:</v>|$)", re.DOTALL)
    segments = []
    for block in re.split(r"\n\s*\n", raw):
        m = _CUE_LINE_RE.search(block)
        if not m:
            continue
        start, end = _cue_bounds(m)
        text_block = block[m.end():].strip()
        if not text_block:
            continue
        vm = voice_re.search(text_block)
        if vm:
            speaker, text = vm.group(1).strip(), re.sub(r"<[^>]+>", "", vm.group(2)).strip()
        else:
            speaker, text = "Unknown", re.sub(r"<[^>]+>", "", text_block).strip()
        if text:
            segments.append({"speaker": speaker, "start": start, "end": end, "text": text})
    return segments


def _parse_srt(raw: str) -> list[dict]:
    segments = []
    for block in re.split(r"\n\s*\n", raw):
        m = _CUE_LINE_RE.search(block)
        if not m:
            continue
        start, end = _cue_bounds(m)
        text = block[m.end():].strip()
        if not text:
            continue
        speaker = "Unknown"
        sm = _SPEAKER_PREFIX_RE.match(text)
        if sm:
            speaker, text = sm.group(1).strip(), sm.group(2).strip()
        if text:
            segments.append({"speaker": speaker, "start": start, "end": end, "text": text})
    return segments


def _parse_teams_transcript(paragraphs: list[str]) -> list[dict] | None:
    """Microsoft Teams' "Download transcript" (.docx) export shapes every
    speaking turn as "{Speaker}   {[H:]MM:SS}\\n{utterance}" — elapsed time
    since the meeting started, not a clock time. The generic colon-based
    speaker-prefix regex mistakes the colon *inside* that timestamp for a
    "Name:" delimiter (matching "Alice   0" as the name from "Alice   0:17"),
    fragmenting one real speaker into a distinct fake one per turn. Detected
    and parsed separately here, with real (not proportional-dummy) timestamps.
    Returns None if the text doesn't actually look like this format, so the
    caller can fall back to generic paragraph parsing."""
    turns = []
    for p in paragraphs:
        m = _TEAMS_TURN_RE.match(p)
        if not m:
            continue
        speaker = re.sub(r"\s+", " ", m.group(1)).strip()
        hours = int(m.group(2)) if m.group(2) else 0
        start = float(hours * 3600 + int(m.group(3)) * 60 + int(m.group(4)))
        body = re.sub(r"\s+", " ", m.group(5)).strip()
        if speaker and body:
            turns.append({"speaker": speaker, "start": start, "text": body})
    if len(turns) < 3 or len(turns) < len(paragraphs) * 0.5:
        return None
    segments = []
    for i, turn in enumerate(turns):
        end = turns[i + 1]["start"] if i + 1 < len(turns) else turn["start"] + max(2.0, len(turn["text"].split()) / 2.5)
        segments.append({"speaker": turn["speaker"], "start": turn["start"], "end": max(end, turn["start"] + 0.1), "text": turn["text"]})
    return segments


def _parse_plain_text(raw: str) -> list[dict]:
    """No real timestamps exist for a plain-text transcript — assigns
    proportional dummy timestamps (~150wpm reading pace) purely so the
    transcript viewer and chunking still work sensibly."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if not paragraphs and raw.strip():
        paragraphs = [raw.strip()]
    teams_segments = _parse_teams_transcript(paragraphs)
    if teams_segments is not None:
        return teams_segments
    segments, t = [], 0.0
    for p in paragraphs:
        speaker, text = "Unknown", p
        sm = _SPEAKER_PREFIX_RE.match(p)
        if sm:
            speaker, text = sm.group(1).strip(), sm.group(2).strip()
        duration = max(2.0, len(text.split()) / 2.5)
        segments.append({"speaker": speaker, "start": t, "end": t + duration, "text": text})
        t += duration
    return segments


TRANSCRIPT_EXTENSIONS = {".txt", ".vtt", ".srt", ".pdf", ".docx"}


def _extract_pdf_text(path: Path) -> str:
    with fitz.open(path) as doc:
        return "\n\n".join(page.get_text() for page in doc)


def _extract_docx_text(path: Path) -> str:
    document = docx.Document(path)
    return "\n\n".join(p.text for p in document.paragraphs if p.text.strip())


def parse_transcript_file(path: Path) -> list[dict]:
    """Parses an already-existing text transcript into the same segments
    shape the audio pipeline produces (speaker/start/end/text), so it feeds
    the same finish_processing() pipeline. Supports WebVTT (with optional
    <v Speaker> voice tags) and SRT (with optional 'Name:' line prefixes),
    PDF and Word (.docx) documents (text is extracted then treated as plain
    paragraphs), and plain text (also the fallback for anything else)."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _parse_plain_text(_extract_pdf_text(path))
    if suffix == ".docx":
        return _parse_plain_text(_extract_docx_text(path))

    raw = path.read_text(encoding="utf-8", errors="ignore")
    if raw.lstrip().upper().startswith("WEBVTT"):
        return _parse_vtt(raw)
    if re.match(r"^\s*1\s*\n\s*\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->", raw):
        return _parse_srt(raw)
    return _parse_plain_text(raw)


@app.post("/admin/api/meetings/upload-transcript")
async def upload_transcript(file: UploadFile = File(...), username: str = Depends(verify_admin)):
    if Path(file.filename or "").suffix.lower() not in TRANSCRIPT_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type. Allowed: {', '.join(sorted(TRANSCRIPT_EXTENSIONS))}")
    safe_name = f"{uuid.uuid4().hex}_{file.filename}"
    dest = UPLOADS_DIR / safe_name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    meeting_id = db.insert_meeting(
        title=file.filename, source_filename=safe_name,
        uploaded_at=datetime.now().isoformat(), whisper_model=None,
        source_type="transcript",
    )
    log.info("Meeting %d uploaded (transcript): %s", meeting_id, file.filename)

    db.update_meeting_status(meeting_id, "processing")
    try:
        segments = parse_transcript_file(dest)
        if not segments:
            raise RuntimeError("Could not parse any text from the uploaded transcript.")
        meeting = db.get_meeting(meeting_id)
        await finish_processing(meeting_id, meeting["title"], segments)
        return {"meeting_id": meeting_id, "status": "done"}
    except Exception as e:
        log.error("Meeting %d: transcript processing failed: %s", meeting_id, e, exc_info=True)
        db.update_meeting_status(meeting_id, "failed", str(e))
        raise HTTPException(500, f"Processing failed: {e}")


# ── Admin: participants ──────────────────────────────────────────────────────
@app.post("/admin/api/participants/{participant_id}")
async def update_participant(participant_id: int, data: dict, username: str = Depends(verify_admin)):
    name = (data.get("display_name") or "").strip() or None
    email = (data.get("email") or "").strip() or None
    db.update_participant(participant_id, name, email)
    return {"status": "updated"}


# ── Admin: tasks ─────────────────────────────────────────────────────────────
@app.get("/admin/api/tasks/due")
async def tasks_due(username: str = Depends(verify_admin)):
    return {"tasks": [_enrich_task_owner(t) for t in db.list_tasks_due()]}


@app.post("/admin/api/tasks/{task_id}/status")
async def set_task_status(task_id: int, data: dict, username: str = Depends(verify_admin)):
    status = data.get("status", "open")
    if status not in ("open", "done"):
        raise HTTPException(400, "status must be 'open' or 'done'")
    db.set_task_status(task_id, status)
    return {"status": "updated"}


@app.post("/admin/api/tasks/{task_id}/send-reminder")
async def send_task_reminder(task_id: int, username: str = Depends(verify_admin)):
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if not task.get("owner"):
        raise HTTPException(400, "This task has no owner to notify")
    participant = db.find_participant_by_name(task["meeting_id"], task["owner"])
    if not participant or not participant.get("email"):
        raise HTTPException(
            400,
            f"No email on file for \"{task['owner']}\" — add one in the participant list first."
        )
    meeting = db.get_meeting(task["meeting_id"])
    deadline = f" (due {task['deadline_date']})" if task.get("deadline_date") else ""
    subject = f"Reminder: {task['task_text'][:80]}"
    body = (
        f"Hi {participant.get('display_name') or task['owner']},\n\n"
        f"This is a reminder about an open action item from \"{meeting['title']}\":\n\n"
        f"  {task['task_text']}{deadline}\n\n"
        f"— Sent from Meeting Intelligence"
    )
    try:
        emailer.send_email(participant["email"], subject, body)
    except Exception as e:
        raise HTTPException(500, f"Could not send reminder: {e}")
    log.info("Reminder sent for task %d to %s", task_id, participant["email"])
    return {"status": "sent", "to": participant["email"]}


# ── Admin: email/SMTP settings ───────────────────────────────────────────────
def _mask_email_config(cfg: dict) -> dict:
    out = dict(cfg)
    out["has_password"] = bool(out.pop("password", ""))
    return out


@app.get("/admin/api/email-config")
async def get_email_config(username: str = Depends(verify_admin)):
    return _mask_email_config(emailer.load_config())


@app.post("/admin/api/email-config")
async def save_email_config(data: dict, username: str = Depends(verify_admin)):
    cfg = emailer.load_config()
    for key in ("smtp_host", "username", "from_address", "from_name"):
        if key in data:
            cfg[key] = data[key]
    if "smtp_port" in data:
        cfg["smtp_port"] = int(data["smtp_port"] or 587)
    if "use_tls" in data:
        cfg["use_tls"] = bool(data["use_tls"])
    if data.get("password"):
        cfg["password"] = data["password"]
    emailer.save_config(cfg)
    log.info("Email config updated: host=%s user=%s", cfg["smtp_host"], cfg["username"])
    return {"status": "saved"}


@app.post("/admin/api/email-config/test")
async def test_email_config(data: dict, username: str = Depends(verify_admin)):
    to_addr = (data.get("to") or "").strip()
    if not to_addr:
        raise HTTPException(400, "A test recipient address is required")
    try:
        emailer.send_email(to_addr, "Meeting Intelligence — test email",
                            "This is a test email from the Meeting Intelligence Email Settings pane.")
    except Exception as e:
        return {"error": str(e)}
    return {"status": "sent"}


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
