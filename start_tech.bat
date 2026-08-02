@echo off
title Tech Support Voice Bot
echo.
echo  Starting Tech Support AI Voice Bot...
echo.

cd /d "%~dp0"
call venv\Scripts\activate

echo  Environment activated.
echo  Starting server on http://localhost:8001
echo  Admin panel: http://localhost:8001/admin
echo  Press Ctrl+C to stop.
echo.

cd techsupport-voice-bot\app
uvicorn main_tech:app --host 0.0.0.0 --port 8001
