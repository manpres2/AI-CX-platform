@echo off
title Apex Bank Voice Bot
echo.
echo  Starting Apex Bank AI Voice Bot...
echo.

cd /d "%~dp0"
call venv\Scripts\activate

echo  Environment activated.
echo  Starting server on http://localhost:8000
echo  Admin panel: http://localhost:8000/admin
echo  Press Ctrl+C to stop.
echo.

cd app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
