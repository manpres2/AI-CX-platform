@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv-qwen3tts\Scripts\python.exe" (
  echo Qwen3-TTS is not installed yet.
  echo Run setup_qwen3_tts.bat first.
  pause
  exit /b 1
)

title Qwen3-TTS Voice Clone Service
".venv-qwen3tts\Scripts\python.exe" -m uvicorn qwen3_tts_server:app --host 127.0.0.1 --port 8020
