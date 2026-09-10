"""Shared client used by every bot; no speech model is loaded in a bot process."""
import os
import subprocess
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8021"
_start_lock = threading.Lock()


def _healthy():
    try:
        response = httpx.get(URL + "/health", timeout=2)
        return response.status_code == 200 and response.json().get("service") == "local-tts"
    except (httpx.HTTPError, ValueError):
        return False


def ensure_started():
    if _healthy():
        return
    with _start_lock:
        if _healthy():
            return
        python = ROOT / ".venv-local-tts" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.exists():
            raise RuntimeError("Local TTS is not installed. Run setup_local_tts.ps1 first.")
        with (ROOT / "local_tts_server.log").open("ab") as log:
            subprocess.Popen([str(python), "-m", "uvicorn", "local_tts_server:app",
                              "--host", "127.0.0.1", "--port", "8021"],
                             cwd=ROOT, stdout=log, stderr=log,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if _healthy():
                return
            time.sleep(0.5)
        raise RuntimeError("Local TTS did not start. See local_tts_server.log.")


def _post(path, payload):
    response = httpx.post(URL + path, json=payload, timeout=120)
    if response.is_error:
        try:
            message = response.json().get("detail", response.text)
        except ValueError:
            message = response.text
        raise RuntimeError(str(message))
    return response


def select(engine):
    # Saving an engine should not download/load it or start an idle service.
    if _healthy():
        _post("/select", {"engine": engine})


def synthesize(engine, text, **settings):
    ensure_started()
    return _post("/tts", {"engine": engine, "text": text, **settings}).content
