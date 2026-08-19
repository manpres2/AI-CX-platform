@echo off
title Local AI Platform - Stop All
echo.
echo  Stopping all Local AI Platform services, including the Ollama server...
echo.

REM Ollama (11434) is included here alongside the six app ports so "stop
REM everything" actually frees all the GPU/VRAM it was using, not just the
REM apps that called it — it's started by start_all.bat and torn down here
REM to match, rather than being left running in the background afterward.
REM 8004 (the Launcher) is deliberately last: this script can be started
REM from the Launcher's own "Shut Down Platform" button, and killing the
REM Launcher earlier in the list risks taking this script down with it
REM before the remaining ports are stopped.
for %%P in (8000 8001 8002 8003 8005 11434 8004) do (
    for /f "tokens=5" %%I in ('netstat -ano ^| findstr ":%%P " ^| findstr LISTENING') do (
        echo  Killing process on port %%P ^(PID %%I^)...
        taskkill /F /T /PID %%I >nul 2>&1
    )
)

echo.
echo  Done. GPU memory should be freed within a few seconds.
echo.
pause
