@echo off
title Stop Unified Ops Portal
echo.
echo  Stopping Unified Ops Portal (port 8003)...
echo.

set FOUND=0
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8003 " ^| findstr LISTENING') do (
    set FOUND=1
    echo  Killing process %%P ...
    taskkill /F /T /PID %%P >nul 2>&1
)

if "%FOUND%"=="0" (
    echo  No server found running on port 8003.
) else (
    echo.
    echo  Stopped.
)
echo.
pause
