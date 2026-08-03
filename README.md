# AIVoiceBotBFSI

Local Voice AI BFSI Demo — Whisper STT + LLaMA + Kokoro TTS + ChromaDB RAG

Built by Manpreet Singh | CPaaS Presales

## Changelog

Every change from here on gets a one-line entry below, plus its own commit/push, so any version can be recovered with `git log` + `git checkout <commit> -- <file>` or `git revert`.

- 2026-08-03 13:28 — Redesigned the bank admin panel to fill wide screens (fluid width + 2-column card layout) and refreshed it to an Apple-style dark theme — `static/admin.html`
- 2026-08-03 13:28 — Ported the same wide-screen/Apple-style redesign to the tech-support admin panel — `techsupport-voice-bot/static_tech/admin.html`
- 2026-08-03 13:28 — Added a dark/light theme toggle (shared via localStorage) to both tech-support pages — `techsupport-voice-bot/static_tech/admin.html`, `techsupport-voice-bot/static_tech/index.html`
- 2026-08-03 13:28 — Added Hindi voice support (Kokoro Hindi voices, Whisper Hindi STT, Hindi LLM replies, localized canned lines) plus a Conversation Language selector in the admin panel — `techsupport-voice-bot/app/main_tech.py`, `techsupport-voice-bot/static_tech/admin.html`
- 2026-08-03 13:28 — Switched the tech-support RAG embedding model to a multilingual one and rebuilt the KB index, so Hindi questions can retrieve the English knowledge base — `techsupport-voice-bot/app/main_tech.py`, `techsupport-voice-bot/prompt_config_tech.json`
- 2026-08-03 13:28 — Added a `techsupport-voice-app` launch config entry for local previewing — `.claude/launch.json`
- 2026-08-03 13:28 — Attempted a latency fix (streamed sentence-by-sentence replies); reverted in full after it cut speech mid-sentence — `techsupport-voice-bot/app/main_tech.py`, `techsupport-voice-bot/static_tech/index.html`, `techsupport-voice-bot/prompt_config_tech.json`
- 2026-08-03 13:28 — Catch-up commit bundling all of the above (this session had been working uncommitted) — see files list in the commit itself
