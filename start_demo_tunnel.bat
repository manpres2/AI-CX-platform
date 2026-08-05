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
echo       https://dashboard.ngrok.com/domains, then edit this file: comment out
echo       the "start ... http 8004" line below and uncomment the --domain one.
echo.
echo  Make sure start_all.bat is already running before sharing the link.
echo.

cd /d "%~dp0"

rem Launch ngrok in its own window — its own dashboard lives there.
rem Without a claimed static domain (quick/throwaway link, changes every restart):
start "ngrok tunnel" tools\ngrok.exe http 8004

rem With a claimed static domain (stable link) — comment the line above,
rem uncomment this one, and fill in your domain:
rem start "ngrok tunnel" tools\ngrok.exe http 8004 --domain=your-domain.ngrok-free.app

echo  Waiting for the tunnel to come up...
echo.

powershell -NoProfile -Command ^
    "$url = $null;" ^
    "for ($i = 0; $i -lt 20 -and -not $url; $i++) {" ^
    "  Start-Sleep -Seconds 1;" ^
    "  try {" ^
    "    $r = Invoke-RestMethod -Uri 'http://127.0.0.1:4040/api/tunnels' -TimeoutSec 2;" ^
    "    if ($r.tunnels.Count -gt 0) { $url = $r.tunnels[0].public_url }" ^
    "  } catch {}" ^
    "}" ^
    "if ($url) {" ^
    "  Write-Host '';" ^
    "  Write-Host '============================================================';" ^
    "  Write-Host \"  Public demo URL: $url\";" ^
    "  Write-Host '============================================================';" ^
    "} else {" ^
    "  Write-Host '';" ^
    "  Write-Host '  Could not detect the tunnel URL after 20 seconds.';" ^
    "  Write-Host '  Check the separate \"ngrok tunnel\" window that just opened.';" ^
    "  Write-Host '  If that window closed immediately, ngrok likely failed to';" ^
    "  Write-Host '  start -- look for a Windows SmartScreen or Firewall prompt';" ^
    "  Write-Host '  (click \"More info\" -> \"Run anyway\", or \"Allow access\"),';" ^
    "  Write-Host '  then run this script again. You can also check';" ^
    "  Write-Host '  http://127.0.0.1:4040 directly in a browser once it is up.';" ^
    "}"

echo.
echo  This window can be closed — the tunnel keeps running in the
echo  separate "ngrok tunnel" window. Close THAT window to stop it.
echo.
pause
