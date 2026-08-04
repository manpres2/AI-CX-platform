"""SMTP email sending for meeting task reminders — Gmail, Exchange/Office365,
or any other SMTP-AUTH server. One generic STARTTLS implementation (stdlib
smtplib, no new dependency) covers all three; the dashboard's "Gmail" /
"Office 365" buttons just quick-fill host/port, not separate code paths.
"""

import json
import smtplib
from email.mime.text import MIMEText
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "email_config_meet.json"

DEFAULT_CONFIG = {
    "smtp_host": "",
    "smtp_port": 587,
    "use_tls": True,
    "username": "",
    "password": "",
    "from_address": "",
    "from_name": "Meeting Intelligence",
}


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_FILE.exists():
        try:
            cfg.update(json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
    return cfg


def save_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def is_configured(cfg: dict | None = None) -> bool:
    cfg = cfg or load_config()
    return bool(cfg.get("smtp_host") and cfg.get("username") and cfg.get("password") and cfg.get("from_address"))


def send_email(to_addr: str, subject: str, body: str, cfg: dict | None = None):
    cfg = cfg or load_config()
    if not is_configured(cfg):
        raise RuntimeError("SMTP is not configured yet — set it up in Email Settings first.")

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = f"{cfg.get('from_name') or 'Meeting Intelligence'} <{cfg['from_address']}>"
    msg["To"] = to_addr

    host, port = cfg["smtp_host"], int(cfg.get("smtp_port") or 587)
    with smtplib.SMTP(host, port, timeout=20) as server:
        if cfg.get("use_tls", True):
            server.starttls()
        server.login(cfg["username"], cfg["password"])
        server.sendmail(cfg["from_address"], [to_addr], msg.as_string())
