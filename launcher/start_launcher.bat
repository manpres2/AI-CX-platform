@echo off
title Local AI Platform Launcher
echo.
echo  Starting Launcher...
echo.

cd /d "%~dp0\.."
call venv\Scripts\activate

echo  Environment activated.
echo  Launcher page: http://localhost:8004
echo  Press Ctrl+C to stop.
echo.

cd launcher\app
uvicorn main_launcher:app --host 0.0.0.0 --port 8004
