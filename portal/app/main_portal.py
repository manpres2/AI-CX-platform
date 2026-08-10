"""
main_portal.py - Unified dashboard for all three local apps on this machine
(BFSI bank bot :8000, tech-support bot :8001, meeting-intelligence :8002).

Does not merge the three codebases — each stays independent and low-risk to
change. Instead: an Overview tab aggregates health/stats from each app's
existing /health + /api/sysinfo (+ a couple of /admin/api/* endpoints) via
httpx, and one tab per app iframes that app's existing /admin page for full
drill-down management. One URL, one page, one login flow to reach everything.

Run from app/ folder: uvicorn main_portal:app --host 0.0.0.0 --port 8003
"""

import json
import logging
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
import os
from dotenv import load_dotenv

import auth

_this_file = Path(__file__).resolve()
BASE_DIR = _this_file.parent.parent
REPO_ROOT = BASE_DIR.parent
_env_path = REPO_ROOT / ".env"
if _env_path.exists():
    try:
        load_dotenv(_env_path, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(_env_path, encoding="utf-16")

STATIC_DIR = BASE_DIR / "static_portal"
REGISTRY_FILE = REPO_ROOT / "bots_registry.json"

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "apexbank2026")

# Logged to a file as well as the console so the Launcher's Logging tab has
# something to read — a console-only log dies with the terminal window.
LOG_FILE = BASE_DIR / "server_portal.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger("portal")
# Always leave a startup marker: it makes restarts visible in the Launcher's
# Logging tab, and stops a healthy-but-quiet service showing an empty log.
log.info("Unified Ops Portal starting (port 8003)")

BUILTIN_APPS = {
    "bank": {"label": "BFSI Bank Bot", "base": "http://localhost:8000"},
    "tech": {"label": "Tech Support Bot", "base": "http://localhost:8001"},
    "meet": {"label": "Meeting Intelligence", "base": "http://localhost:8002"},
    "studio": {"label": "AI Studio", "base": "http://localhost:8005"},
}
BUILTIN_APP_KEYS = ("bank", "tech", "meet", "portal", "studio")


def load_bot_registry() -> dict:
    """Custom bots created via the launcher's "Create New Bot" feature — read
    fresh on every call (not cached) so a bot created after this process
    started still shows up in Overview and becomes grantable in Users,
    without a Portal restart."""
    if REGISTRY_FILE.exists():
        try:
            return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def all_apps() -> dict:
    apps = dict(BUILTIN_APPS)
    for slug, info in load_bot_registry().items():
        apps[slug] = {"label": info["label"], "base": f"http://localhost:{info['port']}"}
    return apps


def all_app_keys() -> tuple:
    return BUILTIN_APP_KEYS + tuple(load_bot_registry().keys())


app = FastAPI(title="Unified Ops Portal")
security = HTTPBasic()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Multi-user admin auth — shared users.db at the repo root. See app/auth.py
# for the shared implementation (duplicated per app).
auth.configure(REPO_ROOT / "users.db", app_key="portal")
auth.ensure_bootstrap_user(ADMIN_USER, ADMIN_PASS)
verify_admin = auth.verify_admin
require_superadmin = auth.require_superadmin


@app.get("/")
async def root():
    return HTMLResponse("<meta http-equiv='refresh' content='0; url=/admin'>")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/admin")
async def admin_panel(username: str = Depends(verify_admin)):
    dashboard_html = STATIC_DIR / "dashboard.html"
    if dashboard_html.exists():
        return HTMLResponse(dashboard_html.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Place dashboard.html in static_portal/ folder.</h2>")


@app.get("/admin/api/overview")
async def overview(username: str = Depends(verify_admin)):
    results = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, info in all_apps().items():
            base = info["base"]
            try:
                health_resp = (await client.get(f"{base}/health")).json()
                sysinfo_resp = (await client.get(f"{base}/api/sysinfo")).json()
                results[name] = {
                    "label": info["label"], "base": base, "status": "up",
                    "health": health_resp, "sysinfo": sysinfo_resp,
                }
            except Exception as e:
                results[name] = {"label": info["label"], "base": base, "status": "down", "error": str(e)}
    return results


# ── Admin: user management (superadmin-only) ────────────────────────────────
@app.get("/admin/api/users")
async def list_users(username: str = Depends(require_superadmin)):
    return {"users": auth.list_users(), "app_keys": list(all_app_keys())}


@app.post("/admin/api/users")
async def create_user(data: dict, username: str = Depends(require_superadmin)):
    new_username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not new_username or not password:
        raise HTTPException(400, "username and password are required")
    app_keys = [k for k in data.get("app_keys", []) if k in all_app_keys()]
    try:
        user_id = auth.create_user(new_username, password, bool(data.get("is_superadmin")), app_keys)
    except Exception as e:
        raise HTTPException(400, f"Could not create user (username may already exist): {e}")
    log.info("User %r created by %r (superadmin=%s, apps=%s)", new_username, username, bool(data.get("is_superadmin")), app_keys)
    return {"status": "created", "id": user_id}


@app.post("/admin/api/users/{user_id}/permissions")
async def update_permissions(user_id: int, data: dict, username: str = Depends(require_superadmin)):
    user = auth.get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    app_keys = [k for k in data.get("app_keys", []) if k in all_app_keys()]
    auth.set_permissions(user_id, app_keys)
    log.info("Permissions for %r updated by %r: %s", user["username"], username, app_keys)
    return {"status": "updated"}


@app.post("/admin/api/users/{user_id}/password")
async def reset_user_password(user_id: int, data: dict, username: str = Depends(require_superadmin)):
    user = auth.get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    new_password = data.get("password") or ""
    if not new_password:
        raise HTTPException(400, "password is required")
    auth.reset_password(user_id, new_password)
    log.info("Password reset for %r by %r", user["username"], username)
    return {"status": "reset"}


@app.delete("/admin/api/users/{user_id}")
async def delete_user(user_id: int, username: str = Depends(require_superadmin)):
    user = auth.get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    if user["is_superadmin"] and auth.count_superadmins() <= 1:
        raise HTTPException(400, "Can't delete the last remaining superadmin")
    auth.delete_user(user_id)
    log.info("User %r deleted by %r", user["username"], username)
    return {"status": "deleted"}
