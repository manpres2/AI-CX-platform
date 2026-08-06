@echo off
title Local AI Platform - Start All
echo.
echo  Starting all Local AI Platform services...
echo.

cd /d "%~dp0"

REM Every app on this platform talks to Ollama (localhost:11434) for local
REM LLM inference — start it first (if it isn't already running) so it's
REM ready by the time the apps below come up. stop_all.bat tears this back
REM down to match, rather than leaving it running in the background.
netstat -ano | findstr ":11434 " | findstr LISTENING >nul
if errorlevel 1 (
    echo  Starting Ollama server...
    start "Ollama Server (11434)" cmd /k ollama serve
    timeout /t 3 /nobreak >nul
) else (
    echo  Ollama server already running.
)

REM Bank Bot and Tech Support Bot both load Whisper-medium + Kokoro onto the
REM same GPU. Starting them at the exact same moment can make both cold
REM starts noticeably slower (CUDA init contention), so stagger them.
start "Apex Bank Bot (8000)" cmd /k call start.bat
timeout /t 15 /nobreak >nul
start "Tech Support Bot (8001)" cmd /k call start_tech.bat
start "Meeting Intelligence (8002)" cmd /k call meeting-intelligence\start_meet.bat
start "Unified Ops Portal (8003)" cmd /k call portal\start_portal.bat
start "Launcher (8004)" cmd /k call launcher\start_launcher.bat
start "AI Studio (8005)" cmd /k call ai-studio\start_studio.bat

echo.
echo  All six services are starting, each in its own window.
echo  Whisper/embedding models take a little while to load on first start —
echo  give it up to a couple of minutes, especially the first time.
echo.
echo  Opening the launcher in your browser in a few seconds...
timeout /t 10 /nobreak >nul
start http://localhost:8004

echo.
echo  Done. To stop everything, run stop_all.bat (or Ctrl+C in each window).
pause
