@echo off
REM ===========================================================================
REM Stop everything Kubera-related.
REM   Kills the API, frontend (Vite), backup daemon, and any Alpaca engine
REM   that was started from the dashboard's "Start Kubera" button.
REM ===========================================================================

setlocal
echo.
echo Stopping Kubera processes...

REM --- Kill uvicorn (the API) ---
REM Matches both the .venv python and the parent system-python launcher,
REM which is two processes per uvicorn run on Windows.
wmic process where "name='python.exe' and commandline like '%%uvicorn%%services.api.main%%'" call terminate >nul 2>&1

REM --- Kill backup loop ---
wmic process where "name='python.exe' and commandline like '%%backup_predictions_loop%%'" call terminate >nul 2>&1

REM --- Kill scheduler (APScheduler daemon) ---
wmic process where "name='python.exe' and commandline like '%%jobs.scheduler%%'" call terminate >nul 2>&1

REM --- Kill Alpaca sync + engine (if started from dashboard) ---
wmic process where "name='python.exe' and commandline like '%%services.alpaca.sync%%'" call terminate >nul 2>&1
wmic process where "name='python.exe' and commandline like '%%services.alpaca.engine%%'" call terminate >nul 2>&1

REM --- Kill Vite (the frontend) ---
REM Match by the project path so we don't accidentally kill other Vite servers
REM you might be running elsewhere on the same machine.
wmic process where "name='node.exe' and commandline like '%%services\\frontend%%vite%%'" call terminate >nul 2>&1

REM --- Close the launcher windows themselves ---
taskkill /fi "windowtitle eq Kubera-API" /t /f >nul 2>&1
taskkill /fi "windowtitle eq Kubera-Frontend" /t /f >nul 2>&1
taskkill /fi "windowtitle eq Kubera-Backup" /t /f >nul 2>&1
taskkill /fi "windowtitle eq Kubera-Scheduler" /t /f >nul 2>&1

echo Done.
echo.
timeout /t 2 /nobreak >nul
endlocal
exit /b 0
