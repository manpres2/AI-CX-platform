"""
main_launcher.py - Single front-door page for the local AI platform.
Public tile grid linking out (new tab) to the four apps: Apex Bank Bot (8000),
TechCare Support Bot (8001), Meeting Intelligence (8002), Unified Ops Portal
(8003). The page itself needs no login; only editing its logo/company name
does (superadmin-only, via the shared users.db all apps share).

Run from app/ folder: uvicorn main_launcher:app --host 0.0.0.0 --port 8004
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI
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
STATIC_DIR = BASE_DIR / "static_launcher"
BRAND_FILE = BASE_DIR.parent / "branding_launcher.json"

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "apexbank2026")

DEFAULT_BRANDING = {"company_name": "Local AI Platform", "logo_emoji": "🤖"}

app = FastAPI(title="Local AI Platform Launcher")

# Launcher isn't part of the per-app permission matrix (its settings are
# platform-wide, not one app's content) — only require_superadmin is used,
# to gate the branding-edit endpoint. See app/auth.py for the shared
# implementation (duplicated per app, same as every other cross-cutting
# concern in this repo).
auth.configure(BASE_DIR.parent / "users.db", app_key=None)
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
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Place index.html in static_launcher/ folder.</h2>")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/admin/api/branding")
async def get_branding():
    return load_branding()


@app.post("/admin/api/branding")
async def save_branding_route(data: dict, username: str = Depends(require_superadmin)):
    company_name = (data.get("company_name") or DEFAULT_BRANDING["company_name"]).strip()
    logo_emoji = (data.get("logo_emoji") or DEFAULT_BRANDING["logo_emoji"]).strip()
    save_branding({"company_name": company_name, "logo_emoji": logo_emoji})
    return {"status": "saved", "company_name": company_name, "logo_emoji": logo_emoji}
