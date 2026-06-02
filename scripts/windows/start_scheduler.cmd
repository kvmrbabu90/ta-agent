@echo off
REM ===========================================================================
REM Standalone launcher for the daily pipeline scheduler.
REM   Double-click this to start the scheduler in a minimized window.
REM   Does NOT touch the API / frontend / backup / Alpaca engine — safe
REM   to add to a running stack without disrupting anything else.
REM
REM The scheduler fires:
REM   - 08:35 CT (post-open) — refresh OHLCV + predictions + paper backtest
REM   - 17:00 CT (post-close) — same pipeline with end-of-day data
REM   - 22:30 / 23:00 / 23:30 UTC nightly jobs (ingest / predict / settlement)
REM   - 07:00 UTC first business day each month — monthly retrain
REM ===========================================================================

setlocal
set "ROOT=C:\dev\ta-agent"
set "VENV_PY=%ROOT%\.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo ERROR: venv Python not found at %VENV_PY%
    pause
    exit /b 1
)

start "Kubera-Scheduler" /min cmd /k "cd /d %ROOT% && title Kubera-Scheduler && %VENV_PY% -m jobs.scheduler"

echo Scheduler launched in a minimized window (title: Kubera-Scheduler).
echo Next tick: today at 17:00 CT (full daily pipeline post-close).
echo This window will close in 2 seconds.
timeout /t 2 /nobreak >nul
endlocal
exit /b 0
