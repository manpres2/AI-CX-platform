@echo off
title Stop Tech Support Voice Bot
echo.
echo  Stopping Tech Support Voice Bot (port 8001)...
echo.

set FOUND=0
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8001 " ^| findstr LISTENING') do (
    set FOUND=1
    echo  Killing process %%P ...
    taskkill /F /T /PID %%P >nul 2>&1
)

if "%FOUND%"=="0" (
    echo  No server found running on port 8001.
) else (
    echo.
    echo  Stopped. GPU memory should be freed within a few seconds.
)
echo.
pause
