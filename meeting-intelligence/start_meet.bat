@echo off
title Meeting Intelligence Platform
echo.
echo  Starting Meeting Intelligence Platform...
echo.

cd /d "%~dp0\.."
call venv\Scripts\activate

echo  Environment activated.
echo  Starting server on http://localhost:8002
echo  Admin panel: http://localhost:8002/admin
echo  Press Ctrl+C to stop.
echo.

cd meeting-intelligence\app
uvicorn main_meet:app --host 0.0.0.0 --port 8002
