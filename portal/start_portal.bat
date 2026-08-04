@echo off
title Unified Ops Portal
echo.
echo  Starting Unified Ops Portal...
echo.

cd /d "%~dp0\.."
call venv\Scripts\activate

echo  Environment activated.
echo  Starting server on http://localhost:8003
echo  Dashboard: http://localhost:8003/admin
echo  (Bank bot, tech-support bot, and meeting-intelligence should already be running for the Overview tab to show real data.)
echo  Press Ctrl+C to stop.
echo.

cd portal\app
uvicorn main_portal:app --host 0.0.0.0 --port 8003
