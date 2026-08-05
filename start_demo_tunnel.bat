@echo off
title Remote Demo Tunnel
echo.
echo  Starting public tunnel to the launcher (localhost:8004)...
echo.
echo  First time setup (one-time only):
echo    1. Sign up free at https://dashboard.ngrok.com/signup
echo    2. Copy your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
echo    3. Run: tools\ngrok.exe config add-authtoken YOUR_TOKEN
echo    4. (Optional, for a URL that never changes) Claim a free static domain at
echo       https://dashboard.ngrok.com/domains, then edit this file and uncomment
echo       the --domain line below with your domain.
echo.
echo  Make sure start_all.bat is already running before sharing the link.
echo.

cd /d "%~dp0"

rem Without a claimed static domain (quick/throwaway link, changes every restart):
tools\ngrok.exe http 8004

rem With a claimed static domain (stable link, uncomment and edit):
rem tools\ngrok.exe http 8004 --domain=your-domain.ngrok-free.app
