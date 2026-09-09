"""
main_launcher.py - Single front-door page for the local AI platform.
Tile grid linking out (new tab) to the other apps: Apex Bank Bot (8000),
TechCare Support Bot (8001), Meeting Intelligence (8002), Unified Ops Portal
(8003), plus a live status overview of all of them. The whole page requires
superadmin login (via the shared users.db all apps share) — this is the
platform's front door, not a public-facing page.

Run from app/ folder: uvicorn main_launcher:app --host 0.0.0.0 --port 8004
"""

import asyncio
import json
import logging
import os
import re
import shutil
import socket
import sqlite3
import subprocess
from datetime import datetime
from html import escape
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response

import auth

_this_file = Path(__file__).resolve()
_env_path = _this_file.parent.parent.parent / ".env"
if _env_path.exists():
    try:
        load_dotenv(_env_path, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(_env_path, encoding="utf-16")

BASE_DIR = _this_file.parent.parent
REPO_ROOT = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static_launcher"
BRAND_FILE = REPO_ROOT / "branding_launcher.json"
ACCESS_LOG_DB = REPO_ROOT / "access_log.db"

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "apexbank2026")

DEFAULT_BRANDING = {"company_name": "Local AI Platform", "logo_emoji": "🤖"}

# ── Model library ───────────────────────────────────────────────────────────
# One folder per engine role under models/, so an admin can drop in a new voice
# or a new embedding model without touching any app's config. Nothing here is
# ever executed — files are downloaded and listed, and the individual apps pick
# them up by path when they are pointed at one.
MODELS_DIR = REPO_ROOT / "models"
# Every bot's RAG embedding model — a source constant today, not yet an
# admin-configurable setting, so there is nowhere at runtime to read it from.
# bot-template's default covers every dynamically created bot unless someone
# hand-edits that bot's main_bot.py after the fact.
BOT_EMBED_MODELS = {
    "bank": "all-MiniLM-L6-v2",
    "tech": "paraphrase-multilingual-MiniLM-L12-v2",
    "_registry_default": "paraphrase-multilingual-MiniLM-L12-v2",
}

MODEL_CATEGORIES = {
    "llm":        {"label": "Language Models (LLM)",
                   "hint": "GGUF file URL, a Hugging Face repo id, or an Ollama model name (e.g. llama3.1:8b)."},
    "tts":        {"label": "Text-to-Speech Engines",
                   "hint": "Voice model file URL or a Hugging Face repo id (e.g. hexgrad/Kokoro-82M)."},
    "stt":        {"label": "Speech-to-Text Engines",
                   "hint": "Whisper checkpoint URL or a Hugging Face repo id (e.g. openai/whisper-small)."},
    "embeddings": {"label": "RAG Embedding Models",
                   "hint": "Sentence-transformer Hugging Face repo id (e.g. sentence-transformers/all-MiniLM-L6-v2)."},
}

# Download jobs run in the background and are polled by the page. Kept in memory
# on purpose: a half-finished download does not survive a launcher restart, and
# pretending otherwise would just show the admin a job that is not running.
_model_jobs: dict[str, dict] = {}
_model_jobs_lock = asyncio.Lock()

# The launcher is the one place that starts and stops everything else, so its
# own log is the only record of who turned what off — it previously kept none.
LOG_FILE = BASE_DIR / "server_launcher.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger("launcher")
log.info("Launcher starting (port 8004)")

APPS = {
    "bank": {"label": "Apex Bank Bot", "base": "http://localhost:8000"},
    "tech": {"label": "TechCare Support Bot", "base": "http://localhost:8001"},
    "meet": {
        "label": "Meeting Intelligence", "base": "http://localhost:8002",
        "port": 8002, "module": "main_meet:app",
        "cwd": REPO_ROOT / "meeting-intelligence" / "app", "reload": False,
    },
    "portal": {
        "label": "Unified Ops Portal", "base": "http://localhost:8003",
        "port": 8003, "module": "main_portal:app",
        "cwd": REPO_ROOT / "portal" / "app", "reload": False,
    },
    "studio": {
        "label": "AI Studio", "base": "http://localhost:8005",
        "port": 8005, "module": "main_studio:app",
        "cwd": REPO_ROOT / "ai-studio" / "app", "reload": False,
    },
}

BUILTIN_BOTS = {
    "bank": {
        "label": "Apex Bank Bot", "port": 8000, "icon": "🏦", "kind": "voice",
        "module": "main:app", "cwd": REPO_ROOT / "app", "reload": True,
    },
    "tech": {
        "label": "TechCare Support Bot", "port": 8001, "icon": "💻", "kind": "voice",
        "module": "main_tech:app", "cwd": REPO_ROOT / "techsupport-voice-bot" / "app", "reload": False,
    },
}

# ── Dynamic bot provisioning ("Create New Bot") ─────────────────────────────
BOT_TEMPLATE_DIR = REPO_ROOT / "bot-template"
BOTS_DIR = REPO_ROOT / "bots"
REGISTRY_FILE = REPO_ROOT / "bots_registry.json"
UVICORN_EXE = REPO_ROOT / "venv" / "Scripts" / "uvicorn.exe"

SLUG_RE = re.compile(r"^[a-z][a-z0-9-]{2,30}$")
PORT_MIN, PORT_MAX = 8010, 8099

app = FastAPI(title="Local AI Platform Launcher")

# Launcher isn't part of the per-app permission matrix (its settings are
# platform-wide, not one app's content) — only require_superadmin is used,
# gating the whole page (this is the platform's front door, not one app's
# admin panel). See app/auth.py for the shared implementation (duplicated
# per app, same as every other cross-cutting concern in this repo).
auth.configure(REPO_ROOT / "users.db", app_key=None)
auth.ensure_bootstrap_user(ADMIN_USER, ADMIN_PASS)
require_superadmin = auth.require_superadmin


def load_branding() -> dict:
    if BRAND_FILE.exists():
        try:
            data = json.loads(BRAND_FILE.read_text(encoding="utf-8"))
            return {**DEFAULT_BRANDING, **data}
        except Exception:
            pass
    return dict(DEFAULT_BRANDING)


def save_branding(data: dict):
    BRAND_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


LOGO_EXTS = ("png", "jpg", "jpeg", "svg", "webp", "gif")
LOGO_MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
             "svg": "image/svg+xml", "webp": "image/webp", "gif": "image/gif"}


def find_logo_file() -> Path | None:
    for ext in LOGO_EXTS:
        p = STATIC_DIR / f"brand-logo.{ext}"
        if p.exists():
            return p
    return None


# ── Access log (who's opening the shared link) ──────────────────────────────
def init_access_log():
    with sqlite3.connect(ACCESS_LOG_DB) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS access_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            ip TEXT,
            username TEXT,
            app_key TEXT,
            path TEXT,
            user_agent TEXT
        )""")


init_access_log()


def _client_ip(request: Request) -> str:
    # ngrok (and any reverse proxy) sets X-Forwarded-For with the real
    # visitor's IP — request.client.host alone would just show the tunnel's
    # own local forwarding address.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def log_access(request: Request, username: str, app_key: str, path: str):
    with sqlite3.connect(ACCESS_LOG_DB) as con:
        con.execute(
            "INSERT INTO access_log (ts, ip, username, app_key, path, user_agent) VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), _client_ip(request), username,
             app_key, path, request.headers.get("user-agent", "")),
        )


@app.get("/")
async def root(request: Request, username: str = Depends(require_superadmin)):
    log_access(request, username, "launcher", "/")
    index = STATIC_DIR / "index.html"
    if index.exists():
        html = index.read_text(encoding="utf-8").replace("%%USERNAME%%", escape(username))
        return HTMLResponse(html)
    return HTMLResponse("<h2>Place index.html in static_launcher/ folder.</h2>")


@app.get("/admin/api/access-log")
async def get_access_log(username: str = Depends(require_superadmin)):
    with sqlite3.connect(ACCESS_LOG_DB) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM access_log ORDER BY id DESC LIMIT 200").fetchall()
        total = con.execute("SELECT COUNT(*) FROM access_log").fetchone()[0]
        unique_ips = con.execute("SELECT COUNT(DISTINCT ip) FROM access_log").fetchone()[0]
    return {
        "entries": [dict(r) for r in rows],
        "total": total,
        "unique_visitors": unique_ips,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/admin/api/branding")
async def get_branding(username: str = Depends(require_superadmin)):
    data = load_branding()
    data["has_logo_image"] = find_logo_file() is not None
    return data


@app.post("/admin/api/branding")
async def save_branding_route(data: dict, username: str = Depends(require_superadmin)):
    company_name = (data.get("company_name") or DEFAULT_BRANDING["company_name"]).strip()
    logo_emoji = (data.get("logo_emoji") or DEFAULT_BRANDING["logo_emoji"]).strip()
    save_branding({"company_name": company_name, "logo_emoji": logo_emoji})
    return {"status": "saved", "company_name": company_name, "logo_emoji": logo_emoji}


@app.get("/admin/api/branding/logo")
async def get_logo(username: str = Depends(require_superadmin)):
    logo = find_logo_file()
    if not logo:
        raise HTTPException(404, "No logo uploaded")
    return FileResponse(str(logo), media_type=LOGO_MIME.get(logo.suffix.lstrip("."), "image/png"))


@app.post("/admin/api/branding/logo")
async def upload_logo(file: UploadFile = File(...), username: str = Depends(require_superadmin)):
    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in LOGO_EXTS:
        raise HTTPException(400, "Unsupported image type — use PNG, JPG, SVG, WEBP, or GIF")
    for old in STATIC_DIR.glob("brand-logo.*"):
        old.unlink()
    dest = STATIC_DIR / f"brand-logo.{ext}"
    dest.write_bytes(await file.read())
    return {"status": "uploaded", "filename": dest.name}


@app.delete("/admin/api/branding/logo")
async def delete_logo(username: str = Depends(require_superadmin)):
    for f in STATIC_DIR.glob("brand-logo.*"):
        f.unlink()
    return {"status": "removed"}


@app.get("/admin/api/overview")
async def overview(username: str = Depends(require_superadmin)):
    results = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for key, info in APPS.items():
            base = info["base"]
            extra = {"controllable": "port" in info, "port": info.get("port")}
            try:
                health_resp = (await client.get(f"{base}/health")).json()
                sysinfo_resp = (await client.get(f"{base}/api/sysinfo")).json()
                results[key] = {
                    "label": info["label"], "base": base, "status": "up",
                    "health": health_resp, "sysinfo": sysinfo_resp, **extra,
                }
            except Exception as e:
                results[key] = {"label": info["label"], "base": base, "status": "down", "error": str(e), **extra}
    return results


# ── Dynamic bot provisioning ("Create New Bot") ─────────────────────────────
def load_registry() -> dict:
    if REGISTRY_FILE.exists():
        try:
            return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_registry(reg: dict):
    REGISTRY_FILE.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")


def _port_reachable(port: int) -> bool:
    """True if something is actually listening on this port right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _next_free_port(registry: dict) -> int:
    used = {info["port"] for info in registry.values()}
    for port in range(PORT_MIN, PORT_MAX + 1):
        if port not in used and not _port_reachable(port):
            return port
    raise HTTPException(500, "No free ports available in the 8010-8099 range")


def _kill_port(port: int):
    """Find whatever PID is listening on this port and kill it — the same
    netstat-based approach every existing stop_*.bat already uses (more
    robust than persisting a PID, which can go stale across restarts)."""
    try:
        out = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:
        return
    for line in out.splitlines():
        if f":{port} " in line and "LISTENING" in line:
            pid = line.split()[-1]
            subprocess.run(["taskkill", "/F", "/T", "/PID", pid], capture_output=True)


def _write_bat_pair(slug: str, port: int, whisper_model: str):
    start_bat = REPO_ROOT / f"start_{slug}.bat"
    stop_bat = REPO_ROOT / f"stop_{slug}.bat"
    start_bat.write_text(
        "@echo off\r\n"
        f"title {slug}\r\n"
        "cd /d \"%~dp0\"\r\n"
        "call venv\\Scripts\\activate\r\n"
        f"set WHISPER_MODEL={whisper_model}\r\n"
        f"cd bots\\{slug}\\app\r\n"
        f"uvicorn main_bot:app --host 0.0.0.0 --port {port}\r\n",
        encoding="utf-8",
    )
    stop_bat.write_text(
        "@echo off\r\n"
        f"title Stop {slug}\r\n"
        "set FOUND=0\r\n"
        f"for /f \"tokens=5\" %%P in ('netstat -ano ^| findstr \":{port} \" ^| findstr LISTENING') do (\r\n"
        "    set FOUND=1\r\n"
        "    taskkill /F /T /PID %%P >nul 2>&1\r\n"
        ")\r\n"
        f"if \"%FOUND%\"==\"0\" (echo No server found on port {port}.) else (echo Stopped.)\r\n"
        "pause\r\n",
        encoding="utf-8",
    )


def _spawn_detached(args: list, cwd: Path, env: dict, window: bool = False):
    """Launch a process that truly survives the launcher, including a
    `taskkill /T` on the launcher's own PID. DETACHED_PROCESS alone isn't
    enough on Windows — taskkill /T walks recorded parent-PID chains, and a
    process created directly by this one is still found that way even when
    detached. Routing through `cmd /c start "" /B ...` makes the immediate
    parent a cmd.exe that exits right after launching, so by the time anyone
    tree-kills the launcher there's no live parent link left to walk."""
    # `window=True` drops the /B so the process gets its own console — used for
    # the platform shutdown, where the batch file's output (and its pause) is
    # the only thing left to look at once this server is gone.
    prefix = ["cmd", "/c", "start", ""] + ([] if window else ["/B"])
    subprocess.Popen(
        prefix + args,
        cwd=str(cwd), env=env,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
    )


def _spawn_bot_process(slug: str, port: int, whisper_model: str = "small"):
    bot_app_dir = BOTS_DIR / slug / "app"
    env = {**os.environ, "WHISPER_MODEL": whisper_model}
    _spawn_detached(
        [str(UVICORN_EXE), "main_bot:app", "--host", "0.0.0.0", "--port", str(port)],
        bot_app_dir, env,
    )


def _spawn_builtin_process(slug: str):
    info = BUILTIN_BOTS[slug]
    args = [str(UVICORN_EXE), info["module"], "--host", "0.0.0.0", "--port", str(info["port"])]
    if info.get("reload"):
        args.append("--reload")
    _spawn_detached(args, info["cwd"], os.environ.copy())


def _probe_base(base: str) -> str:
    """Probe 127.0.0.1 rather than localhost. Resolving localhost costs real time
    here — an up bot answers in 7ms on the literal address against 267ms on the
    name — and these probes run on every listing."""
    return base.replace("//localhost:", "//127.0.0.1:")


async def _bot_status(client: httpx.AsyncClient, base: str) -> dict:
    """Live status for one bot: up/down, uptime (from its own /api/sysinfo,
    same field every app already exposes), and the error if it's down."""
    base = _probe_base(base)
    try:
        health_resp = await client.get(f"{base}/health")
        health_resp.raise_for_status()
        uptime = None
        try:
            sysinfo_resp = await client.get(f"{base}/api/sysinfo")
            uptime = sysinfo_resp.json().get("uptime")
        except Exception:
            pass
        return {"status": "up", "uptime": uptime, "error": None}
    except Exception as e:
        return {"status": "down", "uptime": None, "error": str(e)}


# A freshly-created (or freshly-started) bot takes a while to come up —
# Whisper and Kokoro alone are most of a minute on a cold GPU. Rather than
# showing a dead "Down" tile the whole time, the launcher reads that bot's own
# server.log and maps the startup lines it already writes to a real percentage,
# so the progress bar reflects actual work done rather than a guessed timer.
_STARTUP_MILESTONES = [
    ("Loading sentence-transformer", 82),
    ("RAG ready.", 92),
    ("KB store empty", 92),
    ("RAG init failed", 92),
    ("Kokoro ready.", 75),
    ("Loading Kokoro TTS...", 55),
    ("Whisper on ", 45),
]
# Both of these are written at import time, so the last occurrence of either
# marks the start of the most recent run — everything before it belongs to a
# previous one and must be ignored, or a restarted bot would report the old
# run's finished progress immediately.
_STARTUP_BEGIN = ("Loading Whisper (", "Bot kind = chat")


def _bot_log_file(slug: str) -> Path | None:
    if slug in BUILTIN_BOTS:
        cwd = BUILTIN_BOTS[slug]["cwd"]
        for name in ("logs", "logs_tech"):
            candidate = cwd.parent / name / "server.log"
            if candidate.exists():
                return candidate
        return None
    return BOTS_DIR / slug / "logs" / "server.log"


def _startup_progress(slug: str) -> dict:
    log_file = _bot_log_file(slug)
    if not log_file or not log_file.exists():
        return {"percent": 5, "stage": "Starting process…"}
    try:
        lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return {"percent": 5, "stage": "Starting process…"}

    begin = max((i for i, ln in enumerate(lines) if any(m in ln for m in _STARTUP_BEGIN)), default=None)
    if begin is None:
        return {"percent": 5, "stage": "Starting process…"}
    recent = lines[begin:]

    # A chat bot loads neither Whisper nor Kokoro, so "Loading models…" would
    # be a lie for it — it's only ever waiting on the knowledge-base index.
    is_chat = "Bot kind = chat" in lines[begin]
    best, stage = 20, "Preparing…" if is_chat else "Loading models…"
    for marker, pct in _STARTUP_MILESTONES:
        if any(marker in ln for ln in recent) and pct > best:
            best, stage = pct, marker.rstrip(". ")
    friendly = {
        "Whisper on": "Speech recognition ready",
        "Loading Kokoro TTS": "Loading speech synthesis…",
        "Kokoro ready": "Speech synthesis ready",
        "Loading sentence-transformer": "Loading knowledge-base index…",
        "RAG ready": "Knowledge base ready",
        "KB store empty": "Knowledge base ready (empty)",
        "RAG init failed": "Knowledge base unavailable",
    }
    return {"percent": best, "stage": friendly.get(stage, stage)}


@app.get("/admin/api/bots/{slug}/startup")
async def bot_startup(slug: str, username: str = Depends(require_superadmin)):
    if slug not in BUILTIN_BOTS and slug not in load_registry():
        raise HTTPException(404, f"Unknown bot '{slug}'")
    port = BUILTIN_BOTS[slug]["port"] if slug in BUILTIN_BOTS else load_registry()[slug]["port"]
    async with httpx.AsyncClient(timeout=3.0) as client:
        st = await _bot_status(client, f"http://localhost:{port}")
    if st["status"] == "up":
        return {"status": "up", "percent": 100, "stage": "Live"}
    return {"status": "starting", **_startup_progress(slug)}


def _bot_entries() -> list[dict]:
    """Every registered bot, from disk only — no network, so this is instant."""
    entries = []
    for slug, info in BUILTIN_BOTS.items():
        entries.append({
            "slug": slug, "label": info["label"], "icon": info["icon"],
            "kind": info.get("kind", "voice"), "port": info["port"],
            "base": f"http://localhost:{info['port']}", "builtin": True,
        })
    for slug, info in load_registry().items():
        entries.append({
            "slug": slug, "label": info["label"],
            "icon": "💬" if info.get("kind") == "chat" else "🤖",
            "base": f"http://localhost:{info['port']}", "builtin": False, **info,
        })
    return entries


async def _statuses_for(entries: list[dict]) -> list[dict]:
    """Probe every bot at once. Sequentially this cost ~2s per bot that is down
    (a refused connection is not instant here), which is exactly the wait the
    bots list used to sit through before it could render anything."""
    async with httpx.AsyncClient(timeout=3.0) as client:
        return list(await asyncio.gather(
            *(_bot_status(client, e["base"]) for e in entries)
        ))


@app.post("/admin/api/platform/shutdown")
async def shutdown_platform(username: str = Depends(require_superadmin)):
    """Stop every service on the platform by running stop_all.bat — the same
    script as the desktop shortcut, so there is one definition of "stop
    everything" rather than two that can drift apart.

    This kills the Launcher too, so the response is sent before the script gets
    that far: the batch is spawned detached (a `taskkill /T` on this process
    would otherwise take the script down with it) and stops port 8004 last."""
    script = REPO_ROOT / "stop_all.bat"
    if not script.exists():
        raise HTTPException(500, "stop_all.bat is missing from the repo root")
    log.warning("PLATFORM SHUTDOWN requested by %s — running stop_all.bat", username)
    _spawn_detached(["cmd", "/c", str(script)], REPO_ROOT, dict(os.environ), window=True)
    return {"status": "stopping", "script": str(script)}


# ── Model library API ───────────────────────────────────────────────────────

def _model_dir(category: str) -> Path:
    if category not in MODEL_CATEGORIES:
        raise HTTPException(400, f"Unknown model category: {category}")
    d = MODELS_DIR / category
    d.mkdir(parents=True, exist_ok=True)
    return d


def _dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _safe_entry_name(name: str) -> str:
    """Whatever the admin typed, reduced to a single harmless folder name — a
    URL's last path segment or an HF repo id must never climb out of models/."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "").strip().strip("."))
    name = name.strip("-.") or "model"
    return name[:80]


def _list_models() -> dict:
    out = {}
    for cat in MODEL_CATEGORIES:
        d = MODELS_DIR / cat
        entries = []
        if d.exists():
            for f in sorted(d.iterdir(), key=lambda f: f.name.lower()):
                if f.name.startswith("."):
                    continue
                entries.append({
                    "name": f.name,
                    "kind": "folder" if f.is_dir() else "file",
                    "size_mb": round(_dir_size(f) / 1048576, 1),
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%d %b %Y %H:%M"),
                    "path": str(f),
                })
        out[cat] = {**MODEL_CATEGORIES[cat], "path": str(d), "entries": entries}
    return out


async def _bot_health(port: int) -> dict | None:
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            r = await client.get(f"http://localhost:{port}/health")
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return None


async def _bots_in_use() -> dict:
    """{category: [{"value": model_name, "bots": [bot_label, ...]}]} for every
    engine actually loaded by a bot that is up right now. Read straight from
    each bot's own /health — a public, unauthenticated route every bot already
    exposes — rather than trusting registry/config files that may be stale if
    the bot was hand-restarted with different settings."""
    targets = [(slug, info["label"], info["port"]) for slug, info in BUILTIN_BOTS.items()]
    targets += [(slug, slug, info["port"]) for slug, info in load_registry().items()]

    results = await asyncio.gather(*(_bot_health(port) for _, _, port in targets))

    by_cat: dict[str, dict[str, set]] = {"llm": {}, "tts": {}, "stt": {}, "embeddings": {}}
    for (slug, label, _port), health in zip(targets, results):
        if not health:
            continue
        embed = BOT_EMBED_MODELS.get(slug, BOT_EMBED_MODELS["_registry_default"])
        for cat, value in (("llm", health.get("ollama_model")),
                          ("tts", health.get("tts")),
                          ("stt", health.get("whisper")),
                          ("embeddings", embed)):
            if not value:
                continue
            by_cat[cat].setdefault(value, set()).add(label)

    return {cat: [{"value": v, "bots": sorted(labels)} for v, labels in vals.items()]
            for cat, vals in by_cat.items()}


@app.get("/admin/api/models")
async def list_models(username: str = Depends(require_superadmin)):
    async with _model_jobs_lock:
        jobs = sorted(_model_jobs.values(), key=lambda j: j["started"], reverse=True)[:20]
    return {"categories": _list_models(), "jobs": jobs, "in_use": await _bots_in_use()}


@app.delete("/admin/api/models/{category}/{name}")
async def delete_model(category: str, name: str, username: str = Depends(require_superadmin)):
    d = _model_dir(category)
    target = (d / name).resolve()
    if d.resolve() not in target.parents:
        raise HTTPException(400, "Refusing to delete outside the model folder")
    if not target.exists():
        raise HTTPException(404, "Model not found")
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    log.info("Model deleted by %s: %s/%s", username, category, name)
    return {"status": "deleted"}


async def _run_model_download(job_id: str, category: str, source: str, name: str):
    """Fetch one model. Three shapes are accepted because the ecosystem has
    three: a plain file URL, a Hugging Face repo id, and an Ollama model name."""

    async def progress(**kw):
        async with _model_jobs_lock:
            _model_jobs[job_id].update(kw)

    dest_dir = _model_dir(category)
    try:
        if source.startswith(("http://", "https://")):
            filename = _safe_entry_name(name or source.rstrip("/").split("/")[-1].split("?")[0])
            dest = dest_dir / filename
            tmp = dest.with_suffix(dest.suffix + ".part")
            await progress(status="downloading", detail=f"Fetching {filename}")
            async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
                async with client.stream("GET", source) as resp:
                    resp.raise_for_status()
                    total = int(resp.headers.get("content-length") or 0)
                    done = 0
                    with open(tmp, "wb") as fh:
                        async for chunk in resp.aiter_bytes(1 << 20):
                            fh.write(chunk)
                            done += len(chunk)
                            await progress(
                                downloaded_mb=round(done / 1048576, 1),
                                total_mb=round(total / 1048576, 1) if total else 0,
                                percent=round(done * 100 / total, 1) if total else None,
                            )
            tmp.replace(dest)
            await progress(status="done", detail=f"Saved to models/{category}/{filename}",
                           percent=100)

        elif re.fullmatch(r"[A-Za-z0-9._-]+(:[A-Za-z0-9._-]+)?", source) and ":" in source:
            # Ollama model names are the only source shape carrying a tag.
            await progress(status="downloading", detail=f"ollama pull {source}")
            proc = await asyncio.create_subprocess_exec(
                "ollama", "pull", source,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode(errors="ignore").strip()
                if text:
                    await progress(detail=text[:200])
            if await proc.wait() != 0:
                raise RuntimeError("ollama pull failed — is Ollama installed and running?")
            await progress(status="done", percent=100,
                           detail=f"{source} pulled into Ollama (not stored under models/)")

        elif re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", source):
            from huggingface_hub import snapshot_download
            folder = _safe_entry_name(name or source.split("/")[-1])
            await progress(status="downloading", detail=f"Pulling {source} from Hugging Face")
            await asyncio.to_thread(
                snapshot_download, repo_id=source,
                local_dir=str(dest_dir / folder), local_dir_use_symlinks=False)
            await progress(status="done", percent=100,
                           detail=f"Saved to models/{category}/{folder}")
        else:
            raise ValueError(
                "Not recognised. Use an http(s) file URL, a Hugging Face repo id "
                "(owner/name), or an Ollama model name with a tag (name:tag).")

    except asyncio.CancelledError:
        await progress(status="cancelled", detail="Cancelled")
        raise
    except Exception as e:
        log.warning("Model download failed (%s): %s", source, e)
        await progress(status="error", detail=str(e)[:300])
    finally:
        async with _model_jobs_lock:
            _model_jobs[job_id]["finished"] = datetime.now().isoformat()
            _model_jobs[job_id].pop("_task", None)


@app.post("/admin/api/models/download")
async def download_model(data: dict, username: str = Depends(require_superadmin)):
    category = (data.get("category") or "").strip()
    source = (data.get("source") or "").strip()
    name = (data.get("name") or "").strip()
    if category not in MODEL_CATEGORIES:
        raise HTTPException(400, "Pick a model category")
    if not source:
        raise HTTPException(400, "Enter a URL, Hugging Face repo id, or Ollama model name")
    job_id = f"{category}-{int(datetime.now().timestamp() * 1000)}"
    job = {"id": job_id, "category": category, "source": source, "name": name,
           "status": "queued", "detail": "Queued", "percent": None,
           "downloaded_mb": 0, "total_mb": 0,
           "started": datetime.now().isoformat(), "finished": None}
    async with _model_jobs_lock:
        _model_jobs[job_id] = job
    task = asyncio.create_task(_run_model_download(job_id, category, source, name))
    async with _model_jobs_lock:
        _model_jobs[job_id]["_task"] = task
    log.info("Model download started by %s: %s -> %s", username, source, category)
    return {"status": "started", "job_id": job_id}


@app.post("/admin/api/models/jobs/{job_id}/cancel")
async def cancel_model_job(job_id: str, username: str = Depends(require_superadmin)):
    async with _model_jobs_lock:
        job = _model_jobs.get(job_id)
        task = job.get("_task") if job else None
    if not job:
        raise HTTPException(404, "No such job")
    if task:
        task.cancel()
    return {"status": "cancelling"}


@app.post("/admin/api/models/jobs/clear")
async def clear_model_jobs(username: str = Depends(require_superadmin)):
    """Drop finished jobs from the list. Anything still running stays."""
    async with _model_jobs_lock:
        for jid in [j for j, v in _model_jobs.items() if v.get("finished")]:
            del _model_jobs[jid]
    return {"status": "cleared"}


@app.get("/admin/api/bots")
async def list_bots(status: bool = True, username: str = Depends(require_superadmin)):
    """`status=false` skips the probing entirely, so the page can paint its
    tiles straight away and ask for status separately."""
    entries = _bot_entries()
    if not status:
        return {"bots": entries}
    for entry, st in zip(entries, await _statuses_for(entries)):
        entry.update(st)
    return {"bots": entries}


@app.get("/admin/api/bots/status")
async def bots_status(username: str = Depends(require_superadmin)):
    """Just the live status of each bot, keyed by slug."""
    entries = _bot_entries()
    sts = await _statuses_for(entries)
    return {"statuses": {e["slug"]: st for e, st in zip(entries, sts)}}


@app.post("/admin/api/bots")
async def create_bot(data: dict, username: str = Depends(require_superadmin)):
    label = (data.get("label") or "").strip()
    slug = (data.get("slug") or "").strip().lower()
    if not label:
        raise HTTPException(400, "A display name is required")
    if not SLUG_RE.match(slug):
        raise HTTPException(400, "Slug must be lowercase letters/numbers/hyphens, 3-31 characters, starting with a letter")

    registry = load_registry()
    if slug in registry:
        raise HTTPException(400, f"A bot with slug '{slug}' already exists")

    port = data.get("port")
    if port:
        port = int(port)
        if not (PORT_MIN <= port <= PORT_MAX):
            raise HTTPException(400, f"Port must be between {PORT_MIN} and {PORT_MAX}")
        if port in {info["port"] for info in registry.values()} or _port_reachable(port):
            raise HTTPException(400, f"Port {port} is already in use")
    else:
        port = _next_free_port(registry)

    kind = data.get("kind") if data.get("kind") in ("voice", "chat") else "voice"
    language = data.get("language") if data.get("language") in ("en", "hi") else "en"
    greeting = (data.get("greeting") or "").strip() or f"Hi there! I'm {label}. What's your name, and how can I help you today?"
    whisper_model = data.get("whisper_model") if data.get("whisper_model") in ("small", "base", "medium") else "small"

    bot_dir = BOTS_DIR / slug
    if bot_dir.exists():
        raise HTTPException(400, f"Folder bots/{slug} already exists")
    if not BOT_TEMPLATE_DIR.exists():
        raise HTTPException(500, "bot-template/ is missing — cannot scaffold a new bot")

    shutil.copytree(BOT_TEMPLATE_DIR, bot_dir)
    for sub in ("kb_docs", "kb_store", "logs"):
        (bot_dir / sub).mkdir(exist_ok=True)

    (bot_dir / "branding.json").write_text(json.dumps({
        "bank_name": label, "tagline": "AI Chat Assistant" if kind == "chat" else "AI Voice Assistant",
        "badge_text": "Local AI Platform", "logo_emoji": data.get("logo_emoji") or ("💬" if kind == "chat" else "🤖"),
        "kind": kind,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    if kind == "chat":
        system_prompt = (
            f"You're a friendly, easygoing AI assistant named {label}, chatting with someone over text. "
            "Talk like a helpful person, not a script — casual and warm.\n"
            "Keep answers focused and not too long-winded, but you can use short lists or line breaks "
            "when that actually makes something clearer to read.\n"
            "If you don't know the person's name yet, ask for it early on in a casual way. Once you "
            "know it, use their first name naturally now and then, but don't overdo it."
        )
    else:
        system_prompt = (
            f"You're a friendly, easygoing AI voice assistant named {label}, helping a caller "
            "over the phone. Talk like a helpful person, not a script — casual and warm.\n"
            "Keep every answer to 2-3 SHORT sentences maximum — this is a voice call, not text.\n"
            "Never use bullet points, markdown, numbers, or lists — speak naturally.\n"
            "If you don't know the caller's name yet, ask for it early on in a casual way. Once you "
            "know it, use their first name naturally now and then, but don't overdo it."
        )

    (bot_dir / "prompt_config.json").write_text(json.dumps({
        "system_prompt": system_prompt,
        "greeting": greeting,
        "guardrails": [
            "Never ask for or store passwords, PINs, or other sensitive credentials",
            "Never guarantee an outcome will work — frame it as 'let's try this' rather than a promise",
        ],
        "kb_filter": "", "rag_top_k": 3,
        "fallback_message": "I'm sorry, I'm having trouble helping with that right now. Please try again shortly.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    (bot_dir / "runtime_config.json").write_text(json.dumps({
        "kokoro_voice": "af_heart" if language == "en" else "hf_alpha",
        "ollama_model": "llama3.1:8b",
        "convo_language": language,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    _write_bat_pair(slug, port, whisper_model)

    registry[slug] = {
        "label": label, "port": port, "language": language, "kind": kind,
        "whisper_model": whisper_model,
        "created_at": datetime.now().isoformat(),
        "folder": f"bots/{slug}",
    }
    save_registry(registry)

    _spawn_bot_process(slug, port, whisper_model)

    log.info("Bot '%s' (%s, kind=%s) created by %s on port %d",
             slug, label, kind, username, port)
    return {"status": "created", "slug": slug, "port": port}


@app.post("/admin/api/bots/{slug}/stop")
async def stop_bot(slug: str, username: str = Depends(require_superadmin)):
    if slug in BUILTIN_BOTS:
        _kill_port(BUILTIN_BOTS[slug]["port"])
        log.info("Bot '%s' stopped by %s", slug, username)
        return {"status": "stopped"}
    registry = load_registry()
    if slug not in registry:
        raise HTTPException(404, "Bot not found")
    _kill_port(registry[slug]["port"])
    log.info("Bot '%s' stopped by %s", slug, username)
    return {"status": "stopped"}


@app.post("/admin/api/bots/{slug}/start")
async def start_bot(slug: str, username: str = Depends(require_superadmin)):
    if slug in BUILTIN_BOTS:
        info = BUILTIN_BOTS[slug]
        if _port_reachable(info["port"]):
            raise HTTPException(400, "Already running")
        _spawn_builtin_process(slug)
        log.info("Bot '%s' started by %s on port %d", slug, username, info["port"])
        return {"status": "started"}
    registry = load_registry()
    if slug not in registry:
        raise HTTPException(404, "Bot not found")
    info = registry[slug]
    if _port_reachable(info["port"]):
        raise HTTPException(400, "Already running")
    _spawn_bot_process(slug, info["port"], info.get("whisper_model", "small"))
    log.info("Bot '%s' started by %s on port %d", slug, username, info["port"])
    return {"status": "started"}


def _spawn_system_app(key: str):
    info = APPS[key]
    args = [str(UVICORN_EXE), info["module"], "--host", "0.0.0.0", "--port", str(info["port"])]
    if info.get("reload"):
        args.append("--reload")
    _spawn_detached(args, info["cwd"], os.environ.copy())


@app.post("/admin/api/system-apps/{key}/stop")
async def stop_system_app(key: str, username: str = Depends(require_superadmin)):
    """Stop/start for the platform apps that aren't voice bots (Meeting
    Intelligence, Portal, and any future service registered in APPS with
    module/cwd/port set) — lets the admin free up GPU/CPU by turning off
    whichever functions aren't currently needed, same mechanism the AI Voice
    Bots tiles already use."""
    info = APPS.get(key)
    if not info or "port" not in info:
        raise HTTPException(404, "This app isn't remotely controllable")
    _kill_port(info["port"])
    log.info("App '%s' (%s) stopped by %s", key, info["label"], username)
    return {"status": "stopped"}


@app.post("/admin/api/system-apps/{key}/start")
async def start_system_app(key: str, username: str = Depends(require_superadmin)):
    info = APPS.get(key)
    if not info or "port" not in info:
        raise HTTPException(404, "This app isn't remotely controllable")
    if _port_reachable(info["port"]):
        raise HTTPException(400, "Already running")
    _spawn_system_app(key)
    log.info("App '%s' (%s) started by %s on port %d", key, info["label"], username, info["port"])
    return {"status": "started"}


@app.delete("/admin/api/bots/{slug}")
async def remove_bot(slug: str, username: str = Depends(require_superadmin)):
    """Stops the process and unregisters the bot. Does NOT delete its files —
    bots/<slug>/ (and its start_<slug>.bat/stop_<slug>.bat) are left on disk
    so the admin can remove them by hand if they're truly done with it,
    rather than this endpoint running an rm -rf on a user-influenced path."""
    registry = load_registry()
    if slug not in registry:
        raise HTTPException(404, "Bot not found")
    _kill_port(registry[slug]["port"])
    del registry[slug]
    save_registry(registry)
    log.warning("Bot '%s' removed from the registry by %s (files left on disk)", slug, username)
    return {"status": "removed", "note": f"Files remain on disk under bots/{slug}/ — delete manually if not needed"}


# ── Logging (one tab per service) ───────────────────────────────────────────
# Every app writes its own log file next to its own code, which until now meant
# a separate terminal window (or filesystem access) per service to read them.
# This collects them behind one API. Paths come from this table and the bot
# registry only — a key from the client is looked up here, never joined onto a
# path, so no request can walk out of these directories.
SERVICE_LOGS = {
    "launcher": {"label": "Launcher", "icon": "🚀", "path": BASE_DIR / "server_launcher.log"},
    "meet": {"label": "Meeting Intelligence", "icon": "🧠",
             "path": REPO_ROOT / "meeting-intelligence" / "server_meet.log"},
    "portal": {"label": "Unified Ops Portal", "icon": "🗂",
               "path": REPO_ROOT / "portal" / "server_portal.log"},
    "studio": {"label": "AI Studio", "icon": "🎛",
               "path": REPO_ROOT / "ai-studio" / "server_studio.log"},
    "bank": {"label": "Apex Bank Bot", "icon": "🏦", "path": REPO_ROOT / "logs" / "server.log"},
    "tech": {"label": "TechCare Support Bot", "icon": "💻",
             "path": REPO_ROOT / "logs_tech" / "server.log"},
}

# Only the last slice of a log is ever read — these files grow without bound
# and an admin asking for "the last 200 lines" shouldn't pull megabytes into
# memory to get them.
MAX_TAIL_BYTES = 2_000_000
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[.,]?\d*)\s*\[(?P<level>[A-Z]+)\]\s*(?P<msg>.*)$"
)


def _log_targets() -> dict:
    """The fixed services, plus one entry per provisioned bot."""
    targets = {k: dict(v) for k, v in SERVICE_LOGS.items()}
    for slug, info in load_registry().items():
        targets[f"bot:{slug}"] = {
            "label": info.get("label", slug),
            "icon": "💬" if info.get("kind") == "chat" else "🤖",
            "path": BOTS_DIR / slug / "logs" / "server.log",
        }
    return targets


def _read_tail(path: Path, limit: int) -> list[str]:
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > MAX_TAIL_BYTES:
            fh.seek(size - MAX_TAIL_BYTES)
            fh.readline()  # drop the partial line the seek landed in the middle of
        data = fh.read()
    return data.decode("utf-8", errors="replace").splitlines()[-limit:]


def _parse_log_lines(lines: list[str]) -> list[dict]:
    """Turns raw lines into {ts, level, text} entries. A line without its own
    `[LEVEL]` header is a continuation — a traceback body, usually — and
    inherits the level above it, so filtering to ERROR keeps whole tracebacks
    instead of just their first line."""
    entries: list[dict] = []
    last_level = "INFO"
    for line in lines:
        m = _LOG_LINE_RE.match(line)
        if m:
            last_level = m.group("level")
            entries.append({"ts": m.group("ts"), "level": last_level, "text": m.group("msg")})
        else:
            entries.append({"ts": None, "level": last_level, "text": line})
    return entries


@app.get("/admin/api/logs/services")
async def list_log_services(username: str = Depends(require_superadmin)):
    out = []
    for key, info in _log_targets().items():
        path: Path = info["path"]
        exists = path.exists()
        stat = path.stat() if exists else None
        out.append({
            "key": key, "label": info["label"], "icon": info["icon"],
            "exists": exists,
            "size_bytes": stat.st_size if stat else 0,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat() if stat else None,
            "filename": path.name,
        })
    return {"services": out}


@app.get("/admin/api/logs/{key}")
async def read_log(key: str, lines: int = 300, level: str = "ALL", q: str = "",
                   username: str = Depends(require_superadmin)):
    info = _log_targets().get(key)
    if not info:
        raise HTTPException(404, f"Unknown service '{key}'")
    path: Path = info["path"]
    if not path.exists():
        return {"key": key, "label": info["label"], "exists": False, "entries": [],
                "note": "No log file yet — this service hasn't been started since logging was added."}

    lines = max(10, min(lines, 5000))
    # Filtering happens after the tail is taken, so a narrow filter reads the
    # same recent window rather than scanning the whole file for old matches.
    entries = _parse_log_lines(_read_tail(path, lines))
    if level and level != "ALL":
        wanted = set(LOG_LEVELS[LOG_LEVELS.index(level):]) if level in LOG_LEVELS else {level}
        entries = [e for e in entries if e["level"] in wanted]
    if q:
        needle = q.lower()
        entries = [e for e in entries if needle in e["text"].lower()]

    stat = path.stat()
    return {
        "key": key, "label": info["label"], "exists": True, "entries": entries,
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "filename": path.name,
    }


@app.get("/admin/api/logs/{key}/download")
async def download_log(key: str, username: str = Depends(require_superadmin)):
    info = _log_targets().get(key)
    if not info:
        raise HTTPException(404, f"Unknown service '{key}'")
    path: Path = info["path"]
    if not path.exists():
        raise HTTPException(404, "No log file yet for this service")
    return FileResponse(str(path), media_type="text/plain",
                        filename=f"{key.replace(':', '_')}_{path.name}")


@app.post("/admin/api/logs/{key}/clear")
async def clear_log(key: str, username: str = Depends(require_superadmin)):
    """Truncates rather than deletes — the running service holds an open handle
    to this file, and deleting it out from under the handler would leave it
    logging into nowhere until the next restart."""
    info = _log_targets().get(key)
    if not info:
        raise HTTPException(404, f"Unknown service '{key}'")
    path: Path = info["path"]
    if not path.exists():
        raise HTTPException(404, "No log file yet for this service")
    with path.open("w", encoding="utf-8"):
        pass
    log.warning("Log for '%s' cleared by %s", key, username)
    return {"status": "cleared"}


# ── Reverse proxy (remote demo access) ──────────────────────────────────────
# Lets the whole platform be reached through one tunnel to just this app: the
# launcher forwards /proxy/<app>/... to the right backend and injects a small
# window.__BASE_PATH__ script into HTML responses so that app's own
# absolute-path fetch()/href/src calls keep resolving correctly when proxied.
# Every backend still does its own Depends(verify_admin) check against the
# same shared users.db, so the forwarded Authorization header satisfies it —
# no double login, no change to any existing permission check.
_PROXY_STRIP_REQ_HEADERS = {"host", "content-length"}
_PROXY_STRIP_RESP_HEADERS = {"content-encoding", "transfer-encoding", "connection", "content-length"}


def _proxy_target(app_key: str) -> str | None:
    if app_key in APPS:
        return APPS[app_key]["base"]
    registry = load_registry()
    if app_key in registry:
        return f"http://localhost:{registry[app_key]['port']}"
    return None


@app.api_route("/proxy/{app_key}/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy(app_key: str, path: str, request: Request, username: str = Depends(require_superadmin)):
    return await _forward_proxy(app_key, path, request, username, f"/proxy/{app_key}")


async def _forward_proxy(app_key: str, path: str, request: Request, username: str, base_path: str):
    target = _proxy_target(app_key)
    if not target:
        raise HTTPException(404, f"Unknown app '{app_key}'")

    url = f"{target}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"

    body = await request.body()
    req_headers = {k: v for k, v in request.headers.items() if k.lower() not in _PROXY_STRIP_REQ_HEADERS}

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=False) as client:
        upstream = await client.request(request.method, url, headers=req_headers, content=body)

    resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _PROXY_STRIP_RESP_HEADERS}
    content = upstream.content
    content_type = upstream.headers.get("content-type", "")
    if "text/html" in content_type:
        log_access(request, username, app_key, "/" + path)
        injected = f"<script>window.__BASE_PATH__='{base_path}';</script>"
        text = content.decode("utf-8", errors="replace")
        text = text.replace("<head>", "<head>" + injected, 1) if "<head>" in text else injected + text
        content = text.encode("utf-8")

    return Response(content=content, status_code=upstream.status_code, headers=resp_headers, media_type=content_type)


# Outbound campaigns share launcher hosting, with their own granular RBAC.
from outbound import create_router as create_outbound_router, require_voice_access
app.include_router(create_outbound_router(REPO_ROOT / "outbound_campaigns.db", STATIC_DIR, _bot_entries))


@app.api_route("/outbound/voice/{app_key}/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def outbound_voice_proxy(app_key: str, path: str, request: Request,
                               credentials=Depends(auth.security)):
    username = require_voice_access(app_key, credentials, {b['slug'] for b in _bot_entries()})
    try:
        return await _forward_proxy(app_key, path, request, username, f"/outbound/voice/{app_key}")
    except httpx.RequestError:
        raise HTTPException(503, "This voice bot is offline. Start it in AI Bots, then reload Voice AI settings.")
