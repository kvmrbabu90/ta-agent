@echo off
REM ===========================================================================
REM One-time setup: make Kubera start automatically every time you log in.
REM
REM Creates a shortcut to start_kubera.cmd in your user Startup folder. After
REM you run this once, Kubera will spin up on its own after every reboot —
REM no clicks needed.
REM
REM To undo: run uninstall_autostart.cmd (deletes the shortcut).
REM ===========================================================================

setlocal
set "ROOT=C:\dev\ta-agent"
set "TARGET=%ROOT%\scripts\windows\start_kubera.cmd"
set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SHORTCUT=%STARTUP%\Kubera.lnk"

if not exist "%TARGET%" (
    echo ERROR: %TARGET% not found.
    pause
    exit /b 1
)

echo Creating Startup shortcut at:
echo   %SHORTCUT%
echo pointing to:
echo   %TARGET%
echo.

REM Use PowerShell to make a real .lnk shortcut. (cmd has no native way.)
REM WindowStyle 7 = minimized; WorkingDirectory keeps relative paths sane.
powershell -NoProfile -Command ^
    "$ws = New-Object -ComObject WScript.Shell;" ^
    "$lnk = $ws.CreateShortcut('%SHORTCUT%');" ^
    "$lnk.TargetPath = '%TARGET%';" ^
    "$lnk.WorkingDirectory = '%ROOT%';" ^
    "$lnk.WindowStyle = 7;" ^
    "$lnk.Description = 'Start Kubera (API + frontend + backup)';" ^
    "$lnk.Save()"

if errorlevel 1 (
    echo Failed to create the shortcut. PowerShell may be blocked by policy.
    pause
    exit /b 1
)

echo.
echo Done. Kubera will now start automatically when you log in to Windows.
echo (You can still double-click start_kubera.cmd anytime to start it manually.)
echo.
pause
endlocal
exit /b 0
