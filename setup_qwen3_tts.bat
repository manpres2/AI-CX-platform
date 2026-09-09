@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv-qwen3tts\Scripts\python.exe" (
  py -3.12 -m venv .venv-qwen3tts
  if errorlevel 1 exit /b 1
)

".venv-qwen3tts\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
".venv-qwen3tts\Scripts\python.exe" -m pip install -r requirements-qwen3-tts.txt
if errorlevel 1 exit /b 1

echo.
echo Qwen3-TTS is installed. Run start_qwen3_tts.bat to start the service.
