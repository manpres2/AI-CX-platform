"""
main_launcher.py - Single front-door page for the local AI platform.
Tile grid linking out (new tab) to the other apps: Apex Bank Bot (8000),
TechCare Support Bot (8001), Meeting Intelligence (8002), Unified Ops Portal
(8003), plus a live status overview of all of them. The whole page requires
superadmin login (via the shared users.db all apps share) — this is the
platform's front door, not a public-facing page.

Run from app/ folder: uvicorn main_launcher:app --host 0.0.0.0 --port 8004
"""

import json
import os
import re
import shutil
import socket
import subprocess
from datetime import datetime
from html import escape
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse

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

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "apexbank2026")

DEFAULT_BRANDING = {"company_name": "Local AI Platform", "logo_emoji": "🤖"}

APPS = {
    "bank": {"label": "Apex Bank Bot", "base": "http://localhost:8000"},
    "tech": {"label": "TechCare Support Bot", "base": "http://localhost:8001"},
    "meet": {"label": "Meeting Intelligence", "base": "http://localhost:8002"},
    "portal": {"label": "Unified Ops Portal", "base": "http://localhost:8003"},
}

BUILTIN_BOTS = {
    "bank": {"label": "Apex Bank Bot", "port": 8000, "icon": "🏦"},
    "tech": {"label": "TechCare Support Bot", "port": 8001, "icon": "💻"},
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


@app.get("/")
async def root(username: str = Depends(require_superadmin)):
    index = STATIC_DIR / "index.html"
    if index.exists():
        html = index.read_text(encoding="utf-8").replace("%%USERNAME%%", escape(username))
        return HTMLResponse(html)
    return HTMLResponse("<h2>Place index.html in static_launcher/ folder.</h2>")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/admin/api/branding")
async def get_branding(username: str = Depends(require_superadmin)):
    return load_branding()


@app.post("/admin/api/branding")
async def save_branding_route(data: dict, username: str = Depends(require_superadmin)):
    company_name = (data.get("company_name") or DEFAULT_BRANDING["company_name"]).strip()
    logo_emoji = (data.get("logo_emoji") or DEFAULT_BRANDING["logo_emoji"]).strip()
    save_branding({"company_name": company_name, "logo_emoji": logo_emoji})
    return {"status": "saved", "company_name": company_name, "logo_emoji": logo_emoji}


@app.get("/admin/api/overview")
async def overview(username: str = Depends(require_superadmin)):
    results = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for key, info in APPS.items():
            base = info["base"]
            try:
                health_resp = (await client.get(f"{base}/health")).json()
                sysinfo_resp = (await client.get(f"{base}/api/sysinfo")).json()
                results[key] = {
                    "label": info["label"], "base": base, "status": "up",
                    "health": health_resp, "sysinfo": sysinfo_resp,
                }
            except Exception as e:
                results[key] = {"label": info["label"], "base": base, "status": "down", "error": str(e)}
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


def _spawn_bot_process(slug: str, port: int, whisper_model: str = "small"):
    bot_app_dir = BOTS_DIR / slug / "app"
    env = {**os.environ, "WHISPER_MODEL": whisper_model}
    subprocess.Popen(
        [str(UVICORN_EXE), "main_bot:app", "--host", "0.0.0.0", "--port", str(port)],
        cwd=str(bot_app_dir), env=env,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
    )


async def _bot_status(client: httpx.AsyncClient, base: str) -> dict:
    """Live status for one bot: up/down, uptime (from its own /api/sysinfo,
    same field every app already exposes), and the error if it's down."""
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


@app.get("/admin/api/bots")
async def list_bots(username: str = Depends(require_superadmin)):
    registry = load_registry()
    results = []
    async with httpx.AsyncClient(timeout=3.0) as client:
        for slug, info in BUILTIN_BOTS.items():
            base = f"http://localhost:{info['port']}"
            st = await _bot_status(client, base)
            results.append({
                "slug": slug, "label": info["label"], "icon": info["icon"],
                "port": info["port"], "base": base, "builtin": True, **st,
            })
        for slug, info in registry.items():
            base = f"http://localhost:{info['port']}"
            st = await _bot_status(client, base)
            results.append({
                "slug": slug, "label": info["label"], "icon": "🤖",
                "base": base, "builtin": False, **info, **st,
            })
    return {"bots": results}


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
        "bank_name": label, "tagline": "AI Voice Assistant",
        "badge_text": "Local AI Platform", "logo_emoji": data.get("logo_emoji") or "🤖",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    (bot_dir / "prompt_config.json").write_text(json.dumps({
        "system_prompt": (
            f"You're a friendly, easygoing AI voice assistant named {label}, helping a caller "
            "over the phone. Talk like a helpful person, not a script — casual and warm.\n"
            "Keep every answer to 2-3 SHORT sentences maximum — this is a voice call, not text.\n"
            "Never use bullet points, markdown, numbers, or lists — speak naturally.\n"
            "If you don't know the caller's name yet, ask for it early on in a casual way. Once you "
            "know it, use their first name naturally now and then, but don't overdo it."
        ),
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
        "label": label, "port": port, "language": language,
        "whisper_model": whisper_model,
        "created_at": datetime.now().isoformat(),
        "folder": f"bots/{slug}",
    }
    save_registry(registry)

    _spawn_bot_process(slug, port, whisper_model)

    return {"status": "created", "slug": slug, "port": port}


@app.post("/admin/api/bots/{slug}/stop")
async def stop_bot(slug: str, username: str = Depends(require_superadmin)):
    registry = load_registry()
    if slug not in registry:
        raise HTTPException(404, "Bot not found")
    _kill_port(registry[slug]["port"])
    return {"status": "stopped"}


@app.post("/admin/api/bots/{slug}/start")
async def start_bot(slug: str, username: str = Depends(require_superadmin)):
    registry = load_registry()
    if slug not in registry:
        raise HTTPException(404, "Bot not found")
    info = registry[slug]
    if _port_reachable(info["port"]):
        raise HTTPException(400, "Already running")
    _spawn_bot_process(slug, info["port"], info.get("whisper_model", "small"))
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
    return {"status": "removed", "note": f"Files remain on disk under bots/{slug}/ — delete manually if not needed"}
