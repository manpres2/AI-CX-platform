@echo off
title AI Studio
echo.
echo  Starting AI Studio...
echo.

cd /d "%~dp0\.."
call venv\Scripts\activate

echo  Environment activated.
echo  Starting server on http://localhost:8005
echo  Admin panel: http://localhost:8005/admin
echo  (Ollama should already be running at localhost:11434 for local models.)
echo  Press Ctrl+C to stop.
echo.

cd ai-studio\app
uvicorn main_studio:app --host 0.0.0.0 --port 8005
