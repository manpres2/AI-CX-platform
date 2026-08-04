"""Shared multi-user admin auth — duplicated per app (matching this repo's
no-shared-module convention), backed by one shared `users.db` SQLite file at
the repo root, so a single set of named admin accounts (and their per-app
permissions) works across every app on this machine.

Call `configure(db_path, app_key)` once at startup (before any request comes
in), then use `verify_admin` / `require_superadmin` as FastAPI dependencies
exactly like the old single-credential `verify_admin` was used — same
signature, same `Depends()` call sites, zero changes needed at each route.
"""

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import bcrypt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

_DB_PATH: Path | None = None
_APP_KEY: str | None = None
_DUMMY_HASH = bcrypt.hashpw(b"dummy", bcrypt.gensalt())  # constant-time cover for unknown usernames


def configure(db_path: Path, app_key: str | None = None):
    global _DB_PATH, _APP_KEY
    _DB_PATH = db_path
    _APP_KEY = app_key


@contextmanager
def _conn():
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _init_schema(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash BLOB NOT NULL,
            is_superadmin INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS permissions (
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            app_key TEXT NOT NULL,
            PRIMARY KEY (user_id, app_key)
        );
    """)


def ensure_bootstrap_user(admin_user: str, admin_pass: str):
    """Idempotent: if users.db has no users yet, seed one superadmin from the
    existing ADMIN_USER/ADMIN_PASS env vars so nobody gets locked out of any
    app when multi-user auth ships. Only the first app to start actually does
    anything here; later apps see users already present and no-op."""
    with _conn() as conn:
        _init_schema(conn)
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count == 0:
            pw_hash = bcrypt.hashpw(admin_pass.encode("utf-8"), bcrypt.gensalt())
            conn.execute(
                "INSERT INTO users (username, password_hash, is_superadmin, created_at) VALUES (?, ?, 1, ?)",
                (admin_user, pw_hash, time.strftime("%Y-%m-%dT%H:%M:%S")),
            )


def _get_user(username: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None


def _has_permission(user_id: int, app_key: str) -> bool:
    with _conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM permissions WHERE user_id = ? AND app_key = ?", (user_id, app_key)
        ).fetchone()
        return row is not None


def _authenticate(credentials: HTTPBasicCredentials) -> dict:
    user = _get_user(credentials.username)
    hash_to_check = user["password_hash"] if user else _DUMMY_HASH
    ok = bcrypt.checkpw(credentials.password.encode("utf-8"), hash_to_check)
    if not user or not ok:
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Basic"})
    return user


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    user = _authenticate(credentials)
    if not user["is_superadmin"] and not (_APP_KEY and _has_permission(user["id"], _APP_KEY)):
        raise HTTPException(status_code=403, detail="You don't have access to this app")
    return user["username"]


def require_superadmin(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    user = _authenticate(credentials)
    if not user["is_superadmin"]:
        raise HTTPException(status_code=403, detail="Superadmin access required")
    return user["username"]


# ── User management CRUD (only the Portal's admin routes call the write ones,
#    but every app carries the same copy of this module) ────────────────────

def list_users() -> list[dict]:
    with _conn() as conn:
        users = conn.execute(
            "SELECT id, username, is_superadmin, created_at FROM users ORDER BY username"
        ).fetchall()
        result = []
        for u in users:
            perms = conn.execute("SELECT app_key FROM permissions WHERE user_id = ?", (u["id"],)).fetchall()
            result.append({**dict(u), "app_keys": [p["app_key"] for p in perms]})
        return result


def get_user_by_id(user_id: int) -> dict | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def create_user(username: str, password: str, is_superadmin: bool, app_keys: list[str]) -> int:
    pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, is_superadmin, created_at) VALUES (?, ?, ?, ?)",
            (username, pw_hash, int(is_superadmin), time.strftime("%Y-%m-%dT%H:%M:%S")),
        )
        user_id = cur.lastrowid
        for key in app_keys:
            conn.execute("INSERT OR IGNORE INTO permissions (user_id, app_key) VALUES (?, ?)", (user_id, key))
        return user_id


def set_permissions(user_id: int, app_keys: list[str]):
    with _conn() as conn:
        conn.execute("DELETE FROM permissions WHERE user_id = ?", (user_id,))
        for key in app_keys:
            conn.execute("INSERT OR IGNORE INTO permissions (user_id, app_key) VALUES (?, ?)", (user_id, key))


def reset_password(user_id: int, new_password: str):
    pw_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt())
    with _conn() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (pw_hash, user_id))


def count_superadmins() -> int:
    with _conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM users WHERE is_superadmin = 1").fetchone()[0]


def delete_user(user_id: int):
    with _conn() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
