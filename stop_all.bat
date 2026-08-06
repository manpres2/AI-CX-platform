@echo off
title Local AI Platform - Stop All
echo.
echo  Stopping all Local AI Platform services (ports 8000-8005)...
echo.

for %%P in (8000 8001 8002 8003 8004 8005) do (
    for /f "tokens=5" %%I in ('netstat -ano ^| findstr ":%%P " ^| findstr LISTENING') do (
        echo  Killing process on port %%P ^(PID %%I^)...
        taskkill /F /T /PID %%I >nul 2>&1
    )
)

echo.
echo  Done. GPU memory should be freed within a few seconds.
echo.
pause
