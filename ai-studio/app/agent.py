"""Agentic AI tool-execution engine — an LLM-driven loop that can read/write
files and run shell commands on this machine on the admin's behalf, gated by
a three-tier safety model (mirrors how a human operator would want to
supervise this: some things are fine to just do, some need a one-click
sign-off first, and some should never happen no matter what the model asks
for):

  BLOCKED — pattern-matched destructive/irreversible commands (formatting a
            drive, mass-deleting, shutting the machine down, disabling
            security tooling, piping a download straight into a shell,
            etc.). Never executed, regardless of approval state.
  RISKY   — anything that writes, deletes, installs, or otherwise changes
            state. Queued in agent_runs / surfaced in the admin UI as a
            pending action; only runs once the admin clicks Approve.
  SAFE    — read-only actions (listing/reading files, `git status`, etc.).
            Auto-run immediately, no approval needed.

File tools (read_file/list_dir/write_file) are additionally sandboxed to a
configurable workspace root — path traversal outside it is refused outright,
independent of the risk tier. Shell commands are not path-sandboxed (a
subprocess's shell can `cd` anywhere), so BLOCKED/RISKY classification is the
real safety boundary there.

Every action — blocked, queued, approved, denied, or auto-run — is written to
agent_audit.db so there's a full record of what the agent did or attempted.
"""

import json
import re
import sqlite3
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import llm

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "agent_config.json"
AUDIT_DB = BASE_DIR / "agent_audit.db"
RUNS_DB = BASE_DIR / "agent_runs.db"

MAX_STEPS = 15
COMMAND_TIMEOUT_S = 30
MAX_OUTPUT_CHARS = 4000

DEFAULT_CONFIG = {
    "delegated": False,          # off by default — admin must explicitly turn this on
    "workspace": str(BASE_DIR.parent),  # repo root
    "auto_approve_writes": False,
}


def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return {**DEFAULT_CONFIG, **json.loads(CONFIG_FILE.read_text(encoding="utf-8"))}
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Safety classification ───────────────────────────────────────────────────
# Matched case-insensitively against the whole command string. These cover the
# actions that are catastrophic or hard/impossible to reverse on a real
# Windows machine — formatting/partitioning, mass deletion of a drive root,
# shutting the system down, disabling security tooling, clearing audit
# trails, and the classic "download and immediately execute" remote-code
# pattern. This list is deliberately conservative (biased toward blocking) —
# it's fine for false positives here to fall through to RISKY (still runs,
# just needs a click) rather than silently under-blocking.
BLOCKED_PATTERNS = [
    r"\bformat\s+[a-z]:", r"\bdiskpart\b", r"\bshutdown\b", r"\brestart-computer\b",
    r"\bdel\s+/s\b", r"\brd\s+/s\b", r"\brmdir\s+/s\b", r"rm\s+-rf\s+/", r"rm\s+-rf\s+[a-z]:\\?\s*$",
    r"remove-item.*-recurse.*-force.*[a-z]:\\\s*$", r":\(\)\s*\{.*:\|:.*\}\s*;", r"\bmkfs\b",
    r"\bdd\s+if=", r"\bbcdedit\b", r"vssadmin\s+delete\s+shadows", r"\bcipher\s+/w",
    r"reg\s+(delete|add)\s+.*hklm", r"\bnetsh\s+advfirewall", r"\bwevtutil\s+cl\b",
    r"schtasks\s+.*\bdelete\b", r"icacls\s+.*everyone.*(/grant|:f)", r"net\s+user\s+.*\s+/add",
    r"disable.*defender", r"set-mppreference.*disable", r"taskkill.*\bexplorer\b",
    r"(curl|invoke-webrequest|iwr|wget)\b.*\|\s*(bash|sh|iex|powershell)", r"\|\s*iex\b",
    r"\bdrop\s+(database|table)\b", r"\btruncate\s+table\b",
]

# Prefix/keyword match against the leading command — anything matching is
# read-only and safe to auto-run. Everything else defaults to RISKY.
SAFE_PREFIXES = [
    r"^dir\b", r"^ls\b", r"^type\b", r"^cat\b", r"^git\s+(status|log|diff|show|branch)\b",
    r"^echo\b", r"^pwd\b", r"^whoami\b", r"^hostname\b", r"^python\s+--version",
    r"^node\s+--version", r"^where\b", r"^which\b", r"^findstr\b", r"^find\s+/i\b",
    r"^tasklist\b", r"^get-process\b", r"^get-childitem\b", r"^get-content\b",
    r"^date\b", r"^time\b", r"^ping\b", r"^ipconfig\b", r"^systeminfo\b", r"^ver\b",
]

_BLOCKED_RE = re.compile("|".join(BLOCKED_PATTERNS), re.IGNORECASE)
_SAFE_RE = re.compile("|".join(SAFE_PREFIXES), re.IGNORECASE)


def classify_command(command: str) -> str:
    if _BLOCKED_RE.search(command):
        return "blocked"
    if _SAFE_RE.match(command.strip()):
        return "safe"
    return "risky"


def classify_tool_call(tool: str, args: dict) -> str:
    """Every tool call (not just run_command) gets a risk tier. File reads and
    directory listings are always safe; writes are risky unless the admin has
    explicitly turned on auto-approve-writes."""
    if tool in ("read_file", "list_dir"):
        return "safe"
    if tool == "write_file":
        cfg = load_config()
        return "safe" if cfg.get("auto_approve_writes") else "risky"
    if tool == "run_command":
        return classify_command(args.get("command", ""))
    if tool == "finish":
        return "safe"
    return "blocked"  # unknown tool — refuse rather than guess


# ── Audit log ────────────────────────────────────────────────────────────────
@contextmanager
def _conn(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _init_audit():
    with _conn(AUDIT_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                run_id TEXT,
                tool TEXT NOT NULL,
                args TEXT NOT NULL,
                risk TEXT NOT NULL,
                status TEXT NOT NULL,
                output TEXT
            )
        """)


def log_audit(run_id: str, tool: str, args: dict, risk: str, status: str, output: str = ""):
    _init_audit()
    with _conn(AUDIT_DB) as conn:
        conn.execute(
            "INSERT INTO audit (ts, run_id, tool, args, risk, status, output) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (time.strftime("%Y-%m-%dT%H:%M:%S"), run_id, tool, json.dumps(args), risk, status,
             (output or "")[:MAX_OUTPUT_CHARS]),
        )


def get_audit(limit: int = 200) -> list[dict]:
    _init_audit()
    with _conn(AUDIT_DB) as conn:
        rows = conn.execute("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


# ── Run persistence (so approval can happen in a later HTTP request) ────────
def _init_runs():
    with _conn(RUNS_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                workspace TEXT NOT NULL,
                status TEXT NOT NULL,
                messages TEXT NOT NULL,
                pending TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)


def _save_run(run: dict):
    _init_runs()
    with _conn(RUNS_DB) as conn:
        conn.execute(
            """INSERT INTO runs (id, task, workspace, status, messages, pending, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET status=excluded.status, messages=excluded.messages,
                   pending=excluded.pending, updated_at=excluded.updated_at""",
            (run["id"], run["task"], run["workspace"], run["status"],
             json.dumps(run["messages"]), json.dumps(run["pending"]) if run["pending"] else None,
             run["created_at"], time.strftime("%Y-%m-%dT%H:%M:%S")),
        )


def get_run(run_id: str) -> dict | None:
    _init_runs()
    with _conn(RUNS_DB) as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["messages"] = json.loads(d["messages"])
        d["pending"] = json.loads(d["pending"]) if d["pending"] else None
        return d


def list_runs(limit: int = 30) -> list[dict]:
    _init_runs()
    with _conn(RUNS_DB) as conn:
        rows = conn.execute("SELECT id, task, status, created_at, updated_at FROM runs ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


# ── Tool execution ───────────────────────────────────────────────────────────
def _safe_path(workspace: Path, rel: str) -> Path | None:
    """Resolves `rel` against the workspace root and refuses anything that
    escapes it (../, absolute paths outside workspace, symlink tricks) —
    returns None if the path is outside the sandbox."""
    try:
        candidate = (workspace / rel).resolve() if not Path(rel).is_absolute() else Path(rel).resolve()
        candidate.relative_to(workspace.resolve())
        return candidate
    except Exception:
        return None


def _execute_tool(tool: str, args: dict, workspace: Path) -> str:
    if tool == "list_dir":
        p = _safe_path(workspace, args.get("path", "."))
        if p is None:
            return "ERROR: path is outside the allowed workspace."
        if not p.exists():
            return f"ERROR: {p} does not exist."
        entries = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())
        return "\n".join(entries) or "(empty directory)"

    if tool == "read_file":
        p = _safe_path(workspace, args.get("path", ""))
        if p is None:
            return "ERROR: path is outside the allowed workspace."
        if not p.is_file():
            return f"ERROR: {p} is not a file."
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"ERROR reading file: {e}"
        return text[:MAX_OUTPUT_CHARS] + ("\n...(truncated)" if len(text) > MAX_OUTPUT_CHARS else "")

    if tool == "write_file":
        p = _safe_path(workspace, args.get("path", ""))
        if p is None:
            return "ERROR: path is outside the allowed workspace."
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args.get("content", ""), encoding="utf-8")
        except Exception as e:
            return f"ERROR writing file: {e}"
        return f"Wrote {len(args.get('content', ''))} chars to {p}"

    if tool == "run_command":
        command = args.get("command", "")
        try:
            proc = subprocess.run(
                command, shell=True, cwd=str(workspace),
                capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S,
            )
            out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
            out = out[:MAX_OUTPUT_CHARS] + ("\n...(truncated)" if len(out) > MAX_OUTPUT_CHARS else "")
            return f"[exit code {proc.returncode}]\n{out}"
        except subprocess.TimeoutExpired:
            return f"ERROR: command timed out after {COMMAND_TIMEOUT_S}s."
        except Exception as e:
            return f"ERROR running command: {e}"

    return f"ERROR: unknown tool '{tool}'."


# ── Agent loop ────────────────────────────────────────────────────────────────
TOOLS_DESC = """You are an autonomous assistant with delegated access to files and a shell
on the admin's own Windows machine. You act step by step: each turn, decide
ONE next action and respond with ONLY a single-line JSON object, no prose, no
markdown fences:

{"thought": "brief reasoning", "tool": "<tool name>", "args": {...}}

Available tools:
- list_dir {"path": "relative/or/absolute/path"} — list a directory's contents
- read_file {"path": "..."} — read a text file
- write_file {"path": "...", "content": "..."} — create or overwrite a text file
- run_command {"command": "..."} — run a shell command (cmd.exe on Windows) and see its output
- finish {"summary": "what you did / the final answer for the admin"} — end the task

Some actions may require the admin's approval before they run — if you see a
tool result saying an action is pending approval or was denied, adapt your
plan accordingly rather than repeating the same call. You get one tool call
per turn; use as few steps as possible."""


def _new_run(task: str, workspace: str) -> dict:
    return {
        "id": uuid.uuid4().hex[:12],
        "task": task,
        "workspace": workspace,
        "status": "running",
        "messages": [
            {"role": "system", "content": TOOLS_DESC},
            {"role": "user", "content": f"Task: {task}"},
        ],
        "pending": None,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


async def _step(run: dict) -> dict:
    """Runs one iteration of the loop: asks the LLM for the next tool call,
    classifies and (auto-runs or queues) it, and updates run state in place.
    Returns the (possibly updated) run dict."""
    workspace = Path(run["workspace"])
    steps_taken = sum(1 for m in run["messages"] if m["role"] == "assistant")
    if steps_taken >= MAX_STEPS:
        run["status"] = "done"
        run["messages"].append({"role": "assistant", "content": json.dumps(
            {"thought": "step limit reached", "tool": "finish", "summary": "Stopped: reached the maximum step limit."})})
        _save_run(run)
        return run

    raw = await llm.generate_reply("agent", run["messages"], json_mode=True)
    run["messages"].append({"role": "assistant", "content": raw})

    try:
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        call = json.loads(cleaned)
        tool = call.get("tool", "")
        args = call.get("args")
        if not args:
            # Smaller local models don't always nest fields under "args" as the
            # schema asks (e.g. {"tool":"run_command","command":"..."} instead
            # of {"tool":"run_command","args":{"command":"..."}}) — fall back
            # to treating every other top-level key as an arg rather than
            # silently executing a tool call with empty args.
            args = {k: v for k, v in call.items() if k not in ("tool", "thought", "summary")}
    except Exception:
        run["messages"].append({"role": "user", "content": "Your last response wasn't valid JSON. "
                                 "Respond with ONLY the JSON object described in the system prompt."})
        _save_run(run)
        return run

    if tool == "finish":
        run["status"] = "done"
        log_audit(run["id"], "finish", {"summary": call.get("summary", "")}, "safe", "completed")
        _save_run(run)
        return run

    risk = classify_tool_call(tool, args)

    if risk == "blocked":
        log_audit(run["id"], tool, args, risk, "blocked")
        run["messages"].append({"role": "user", "content":
            "That action was blocked by a hard safety rule and will never be run. Choose a different approach."})
        _save_run(run)
        return run

    if risk == "risky":
        run["status"] = "awaiting_approval"
        run["pending"] = {"tool": tool, "args": args}
        log_audit(run["id"], tool, args, risk, "pending_approval")
        _save_run(run)
        return run

    # safe — auto-run
    output = _execute_tool(tool, args, workspace)
    log_audit(run["id"], tool, args, risk, "auto_run", output)
    run["messages"].append({"role": "user", "content": f"Result of {tool}:\n{output}"})
    _save_run(run)
    return run


async def start_run(task: str) -> dict:
    cfg = load_config()
    if not cfg.get("delegated"):
        raise PermissionError("System access hasn't been delegated to the agent yet.")
    run = _new_run(task, cfg["workspace"])
    _save_run(run)
    while run["status"] == "running":
        run = await _step(run)
    return run


async def resume_run(run_id: str, approve: bool) -> dict:
    run = get_run(run_id)
    if not run:
        raise KeyError("Run not found")
    if run["status"] != "awaiting_approval" or not run["pending"]:
        raise ValueError("This run has no pending action to approve or deny.")
    tool, args = run["pending"]["tool"], run["pending"]["args"]
    workspace = Path(run["workspace"])
    if approve:
        output = _execute_tool(tool, args, workspace)
        log_audit(run_id, tool, args, "risky", "approved", output)
        run["messages"].append({"role": "user", "content": f"Result of {tool} (admin-approved):\n{output}"})
    else:
        log_audit(run_id, tool, args, "risky", "denied")
        run["messages"].append({"role": "user", "content":
            f"The admin denied that {tool} action. Choose a different approach."})
    run["pending"] = None
    run["status"] = "running"
    _save_run(run)
    while run["status"] == "running":
        run = await _step(run)
    return run
