@echo off
REM Remove the Kubera auto-start shortcut from your user Startup folder.
REM Kubera will no longer launch on login — you can still start it manually
REM by double-clicking start_kubera.cmd.

setlocal
set "SHORTCUT=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Kubera.lnk"

if exist "%SHORTCUT%" (
    del "%SHORTCUT%"
    echo Removed: %SHORTCUT%
    echo Kubera will no longer auto-start on login.
) else (
    echo Nothing to do — no Kubera shortcut found in Startup.
)

echo.
pause
endlocal
exit /b 0
