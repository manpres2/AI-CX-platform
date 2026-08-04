"""SQLite storage for meeting metadata, participants, tasks, and decisions.
Full transcript text lives in a JSON sidecar (see main_meet.py), not here —
this module only holds structured, queryable records. Plain stdlib sqlite3,
no ORM, one connection per call — matches the rest of the repo's
dependency-light style.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "meetings_meet.db"


@contextmanager
def _conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS meetings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            source_filename TEXT,
            uploaded_at TEXT NOT NULL,
            duration_secs REAL,
            status TEXT NOT NULL DEFAULT 'uploaded',
            whisper_model TEXT,
            participant_count INTEGER DEFAULT 0,
            transcript_json_path TEXT,
            processed_at TEXT,
            error_message TEXT
        );

        CREATE TABLE IF NOT EXISTS participants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
            speaker_label TEXT NOT NULL,
            display_name TEXT,
            talk_time_secs REAL DEFAULT 0,
            turn_count INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
            task_text TEXT NOT NULL,
            owner TEXT,
            deadline_date TEXT,
            priority TEXT DEFAULT 'medium',
            status TEXT NOT NULL DEFAULT 'open',
            source_segment_start REAL
        );

        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
            decision_text TEXT NOT NULL,
            timestamp_secs REAL,
            source_segment_start REAL
        );

        CREATE INDEX IF NOT EXISTS idx_tasks_deadline ON tasks(deadline_date, status);
        CREATE INDEX IF NOT EXISTS idx_participants_meeting ON participants(meeting_id);
        CREATE INDEX IF NOT EXISTS idx_tasks_meeting ON tasks(meeting_id);
        CREATE INDEX IF NOT EXISTS idx_decisions_meeting ON decisions(meeting_id);
        """)


# ── Meetings ─────────────────────────────────────────────────────────────────

def insert_meeting(title: str, source_filename: str, uploaded_at: str, whisper_model: str) -> int:
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO meetings (title, source_filename, uploaded_at, status, whisper_model) "
            "VALUES (?, ?, ?, 'uploaded', ?)",
            (title, source_filename, uploaded_at, whisper_model),
        )
        return cur.lastrowid


def update_meeting_status(meeting_id: int, status: str, error_message: str | None = None):
    with _conn() as conn:
        conn.execute(
            "UPDATE meetings SET status = ?, error_message = ? WHERE id = ?",
            (status, error_message, meeting_id),
        )


def finish_meeting(meeting_id: int, duration_secs: float, participant_count: int,
                    transcript_json_path: str, processed_at: str):
    with _conn() as conn:
        conn.execute(
            "UPDATE meetings SET status='done', duration_secs=?, participant_count=?, "
            "transcript_json_path=?, processed_at=? WHERE id=?",
            (duration_secs, participant_count, transcript_json_path, processed_at, meeting_id),
        )


def get_meeting(meeting_id: int) -> dict | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
        return dict(row) if row else None


def list_meetings() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM meetings ORDER BY uploaded_at DESC").fetchall()
        return [dict(r) for r in rows]


def delete_meeting(meeting_id: int):
    with _conn() as conn:
        conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))


# ── Participants ─────────────────────────────────────────────────────────────

def insert_participant(meeting_id: int, speaker_label: str, talk_time_secs: float, turn_count: int):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO participants (meeting_id, speaker_label, talk_time_secs, turn_count) "
            "VALUES (?, ?, ?, ?)",
            (meeting_id, speaker_label, talk_time_secs, turn_count),
        )


def rename_participant(participant_id: int, display_name: str):
    with _conn() as conn:
        conn.execute("UPDATE participants SET display_name = ? WHERE id = ?", (display_name, participant_id))


def list_participants(meeting_id: int) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM participants WHERE meeting_id = ? ORDER BY talk_time_secs DESC", (meeting_id,)
        ).fetchall()
        return [dict(r) for r in rows]


# ── Tasks ────────────────────────────────────────────────────────────────────

def insert_task(meeting_id: int, task_text: str, owner: str | None, deadline_date: str | None,
                 priority: str, source_segment_start: float | None):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO tasks (meeting_id, task_text, owner, deadline_date, priority, source_segment_start) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (meeting_id, task_text, owner, deadline_date, priority or "medium", source_segment_start),
        )


def list_tasks(meeting_id: int) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM tasks WHERE meeting_id = ? ORDER BY id", (meeting_id,)).fetchall()
        return [dict(r) for r in rows]


def list_tasks_due(include_done: bool = False) -> list[dict]:
    """Tasks due today or overdue, joined with their meeting title — the Phase 1
    'reminder view': a live query, not a persisted notification entity."""
    with _conn() as conn:
        status_filter = "" if include_done else "AND t.status = 'open'"
        rows = conn.execute(f"""
            SELECT t.*, m.title AS meeting_title
            FROM tasks t JOIN meetings m ON m.id = t.meeting_id
            WHERE t.deadline_date IS NOT NULL AND t.deadline_date <= date('now') {status_filter}
            ORDER BY t.deadline_date ASC
        """).fetchall()
        return [dict(r) for r in rows]


def set_task_status(task_id: int, status: str):
    with _conn() as conn:
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))


def count_open_tasks() -> int:
    with _conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'open'").fetchone()[0]


def count_overdue_tasks() -> int:
    with _conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status = 'open' AND deadline_date IS NOT NULL "
            "AND deadline_date <= date('now')"
        ).fetchone()[0]


# ── Decisions ────────────────────────────────────────────────────────────────

def insert_decision(meeting_id: int, decision_text: str, timestamp_secs: float | None,
                     source_segment_start: float | None):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO decisions (meeting_id, decision_text, timestamp_secs, source_segment_start) "
            "VALUES (?, ?, ?, ?)",
            (meeting_id, decision_text, timestamp_secs, source_segment_start),
        )


def list_decisions(meeting_id: int) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM decisions WHERE meeting_id = ? ORDER BY id", (meeting_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def count_meetings() -> int:
    with _conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM meetings").fetchone()[0]
