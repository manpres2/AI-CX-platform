"""
main_studio.py - AI Studio: a generic LLM Chat tile plus a delegated-access
Agentic AI tile, sitting alongside the platform's other apps.

Chat and Agent each keep their own independent LLM provider config (local
Ollama model + CPU/GPU run mode, or a cloud OpenAI-compatible endpoint) —
see app/llm.py. The Agent additionally requires the admin to explicitly
"delegate" system access before it can touch files or run commands, and every
action it takes is classified safe/risky/blocked and audited — see app/agent.py.

Admin panel at: http://localhost:8005/admin  (login required)
Run from app/ folder: uvicorn main_studio:app --host 0.0.0.0 --port 8005
"""

import json
import logging
import os
import platform
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

import agent
import auth
import documents
import llm

_this_file = Path(__file__).resolve()
_env_path = _this_file.parent.parent.parent / ".env"
if _env_path.exists():
    try:
        load_dotenv(_env_path, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(_env_path, encoding="utf-16")

BASE_DIR = _this_file.parent.parent
REPO_ROOT = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static_studio"
STATIC_DIR.mkdir(exist_ok=True)

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "apexbank2026")

SERVER_START = time.time()

# AI Studio had no logging at all — it now writes the same file+console log
# every other app in the platform does, so the Launcher's Logging tab can
# show it and so failures survive the terminal window closing.
LOG_FILE = BASE_DIR / "server_studio.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger("ai-studio")
# Always leave a startup marker: it makes restarts visible in the Launcher's
# Logging tab, and stops a healthy-but-quiet service showing an empty log.
log.info("AI Studio starting (port 8005)")

app = FastAPI(title="AI Studio")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

auth.configure(REPO_ROOT / "users.db", app_key="studio")
auth.ensure_bootstrap_user(ADMIN_USER, ADMIN_PASS)
verify_admin = auth.verify_admin


@app.get("/")
async def root():
    return HTMLResponse("<meta http-equiv='refresh' content='0; url=/admin'>")


@app.get("/health")
async def health():
    cfg = agent.load_config()
    return {"status": "ok", "agent_delegated": cfg.get("delegated", False)}


@app.get("/api/sysinfo")
async def sysinfo():
    uptime_s = int(time.time() - SERVER_START)
    h, m, s = uptime_s // 3600, (uptime_s % 3600) // 60, uptime_s % 60
    runs = agent.list_runs(limit=1000)
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "agent_runs": len(runs),
        "uptime": f"{h}h {m}m {s}s",
    }


@app.get("/admin")
async def admin_panel(username: str = Depends(verify_admin)):
    admin_html = STATIC_DIR / "admin.html"
    if admin_html.exists():
        return HTMLResponse(admin_html.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Place admin.html in static_studio/ folder.</h2>")


# ── Provider settings (Chat + Agent, independent) ───────────────────────────
def _mask(cfg: dict) -> dict:
    out = json.loads(json.dumps(cfg))
    for section in ("chat", "agent"):
        out[section]["llm_cloud"]["has_key"] = bool(out[section]["llm_cloud"].pop("api_key", ""))
    return out


@app.get("/admin/api/providers")
async def get_providers(username: str = Depends(verify_admin)):
    cfg = _mask(llm.load_provider_config())
    cfg["available_local_models"] = await llm.list_ollama_models()
    return cfg


@app.post("/admin/api/providers")
async def save_providers(data: dict, username: str = Depends(verify_admin)):
    section = data.get("section")
    if section not in ("chat", "agent"):
        raise HTTPException(400, "section must be 'chat' or 'agent'")
    cfg = llm.load_provider_config()
    cfg[section]["llm_mode"] = data.get("llm_mode", cfg[section]["llm_mode"])
    incoming_local = data.get("llm_local", {})
    if incoming_local.get("ollama_model"):
        cfg[section]["llm_local"]["ollama_model"] = incoming_local["ollama_model"]
    if incoming_local.get("gpu_mode") in ("auto", "cpu", "gpu"):
        cfg[section]["llm_local"]["gpu_mode"] = incoming_local["gpu_mode"]
    incoming_cloud = data.get("llm_cloud", {})
    cfg[section]["llm_cloud"]["base_url"] = incoming_cloud.get("base_url", cfg[section]["llm_cloud"]["base_url"])
    cfg[section]["llm_cloud"]["model"] = incoming_cloud.get("model", cfg[section]["llm_cloud"]["model"])
    if incoming_cloud.get("api_key"):
        cfg[section]["llm_cloud"]["api_key"] = incoming_cloud["api_key"]
    llm.save_provider_config(cfg)
    return {"status": "saved"}


@app.get("/admin/api/providers/model-status")
async def model_status(username: str = Depends(verify_admin)):
    return {"loaded_models": await llm.get_loaded_models()}


@app.post("/admin/api/providers/pull-model")
async def pull_model(data: dict, username: str = Depends(verify_admin)):
    model = (data.get("model") or "").strip()
    if not model:
        raise HTTPException(400, "model required")
    if not llm.start_pull(model):
        raise HTTPException(409, f'"{model}" is already downloading')
    return {"status": "started", "model": model}


@app.get("/admin/api/providers/pull-status")
async def pull_status(model: str, username: str = Depends(verify_admin)):
    if not model:
        raise HTTPException(400, "model required")
    return llm.get_pull_status(model)


@app.post("/admin/api/providers/restart-model")
async def restart_model(data: dict, username: str = Depends(verify_admin)):
    model = data.get("model")
    if not model:
        raise HTTPException(400, "model required")
    await llm.unload_model(model)
    return {"status": "restarted", "model": model}


@app.post("/admin/api/providers/cloud-models")
async def list_cloud_models(data: dict, username: str = Depends(verify_admin)):
    """Ask an OpenAI-compatible provider which models it serves, so the admin can
    pick one instead of having to know its exact id. Falls back to the section's
    saved key when the form's key box is blank, which it is whenever a key is
    already stored."""
    section = data.get("section")
    if section not in ("chat", "agent"):
        raise HTTPException(400, "section must be 'chat' or 'agent'")
    saved = llm.load_provider_config()[section]["llm_cloud"]
    try:
        models = await llm.list_cloud_models(
            data.get("base_url") or saved.get("base_url", ""),
            data.get("api_key") or saved.get("api_key", ""),
        )
        return {"models": models}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


@app.post("/admin/api/providers/test")
async def test_provider(data: dict, username: str = Depends(verify_admin)):
    section = data.get("section")
    if section not in ("chat", "agent"):
        raise HTTPException(400, "section must be 'chat' or 'agent'")
    try:
        reply = await llm.generate_reply(section, [{"role": "user", "content": "Say OK if you can hear me."}])
        return {"reply": reply}
    except Exception as e:
        return {"error": str(e)}


# ── Chat ─────────────────────────────────────────────────────────────────────
@app.post("/admin/api/chat/upload")
async def chat_upload(file: UploadFile = File(...), username: str = Depends(verify_admin)):
    """Turns an uploaded PDF/Word/text file into text the model can read, or an
    image into a data URI a vision-capable model can look at. Nothing is stored
    server-side — the extracted result goes straight back to the browser, which
    attaches it to the next chat message."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in documents.ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext or '?'}'. "
                                 f"Allowed: {', '.join(sorted(documents.ALLOWED_EXTENSIONS))}")
    data = await file.read()
    if len(data) > documents.MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"File is too large (max {documents.MAX_UPLOAD_BYTES // (1024*1024)} MB)")

    if ext in documents.IMAGE_EXTENSIONS:
        cfg = llm.load_provider_config()["chat"]
        vision_ok = True
        if cfg["llm_mode"] == "local":
            vision_ok = await llm.model_supports_vision(cfg["llm_local"].get("ollama_model", ""))
        return {
            "name": file.filename, "kind": "image",
            "data_uri": documents.to_data_uri(file.filename, data),
            "vision_ok": vision_ok,
        }

    try:
        text = documents.extract_text(file.filename, data)
    except Exception as e:
        raise HTTPException(400, f"Could not read '{file.filename}': {e}")
    if not text.strip():
        raise HTTPException(400, f"No readable text found in '{file.filename}' "
                                 f"(if it's a scanned document, attach it as an image instead)")
    return {"name": file.filename, "kind": "document", "text": text, "chars": len(text)}


@app.post("/admin/api/chat")
async def chat(data: dict, username: str = Depends(verify_admin)):
    messages = data.get("messages", [])
    if not messages:
        raise HTTPException(400, "messages required")
    try:
        reply = await llm.generate_reply("chat", messages)
        return {"reply": reply}
    except Exception as e:
        return {"error": str(e)}


@app.post("/admin/api/chat/export")
async def chat_export(data: dict, username: str = Depends(verify_admin)):
    """Renders a model reply into a real .pdf/.docx/.txt for download."""
    content = (data.get("content") or "").strip()
    fmt = data.get("format")
    if not content:
        raise HTTPException(400, "content required")
    if fmt not in ("pdf", "docx", "txt"):
        raise HTTPException(400, "format must be pdf, docx or txt")
    title = (data.get("title") or "").strip() or "AI Studio Document"
    blob, mime = documents.make_document(content, title, fmt)
    safe_title = re.sub(r"[^A-Za-z0-9 _-]", "", title).strip() or "document"
    return Response(
        content=blob, media_type=mime,
        headers={"Content-Disposition": f'attachment; filename="{safe_title}.{fmt}"'},
    )


# ── Agent ────────────────────────────────────────────────────────────────────
@app.get("/admin/api/agent/config")
async def get_agent_config(username: str = Depends(verify_admin)):
    return agent.load_config()


@app.post("/admin/api/agent/config")
async def save_agent_config(data: dict, username: str = Depends(verify_admin)):
    cfg = agent.load_config()
    if "workspace" in data and data["workspace"]:
        ws = Path(data["workspace"])
        if not ws.exists() or not ws.is_dir():
            raise HTTPException(400, f"'{data['workspace']}' is not an existing directory")
        cfg["workspace"] = str(ws.resolve())
    if "delegated" in data:
        cfg["delegated"] = bool(data["delegated"])
    if "auto_approve_writes" in data:
        cfg["auto_approve_writes"] = bool(data["auto_approve_writes"])
    agent.save_config(cfg)
    return {"status": "saved", **cfg}


@app.post("/admin/api/agent/run")
async def agent_run(data: dict, username: str = Depends(verify_admin)):
    task = (data.get("task") or "").strip()
    if not task:
        raise HTTPException(400, "task required")
    try:
        run = await agent.start_run(task)
        return run
    except PermissionError as e:
        raise HTTPException(403, str(e))


@app.get("/admin/api/agent/runs")
async def agent_runs(username: str = Depends(verify_admin)):
    return {"runs": agent.list_runs()}


@app.get("/admin/api/agent/runs/{run_id}")
async def agent_run_detail(run_id: str, username: str = Depends(verify_admin)):
    run = agent.get_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return run


@app.post("/admin/api/agent/runs/{run_id}/approve")
async def agent_approve(run_id: str, username: str = Depends(verify_admin)):
    try:
        return await agent.resume_run(run_id, approve=True)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.post("/admin/api/agent/runs/{run_id}/deny")
async def agent_deny(run_id: str, username: str = Depends(verify_admin)):
    try:
        return await agent.resume_run(run_id, approve=False)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.get("/admin/api/agent/audit")
async def agent_audit(username: str = Depends(verify_admin)):
    return {"entries": agent.get_audit()}
