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

import logging
import secrets
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
import os
from dotenv import load_dotenv

_this_file = Path(__file__).resolve()
_env_path = _this_file.parent.parent.parent / ".env"
if _env_path.exists():
    try:
        load_dotenv(_env_path, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(_env_path, encoding="utf-16")

BASE_DIR = _this_file.parent.parent
STATIC_DIR = BASE_DIR / "static_portal"

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "apexbank2026")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("portal")

APPS = {
    "bank": {"label": "BFSI Bank Bot", "base": "http://localhost:8000"},
    "tech": {"label": "Tech Support Bot", "base": "http://localhost:8001"},
    "meet": {"label": "Meeting Intelligence", "base": "http://localhost:8002"},
}

app = FastAPI(title="Unified Ops Portal")
security = HTTPBasic()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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
        for name, info in APPS.items():
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
