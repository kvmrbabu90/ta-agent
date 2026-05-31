@echo off
REM ===========================================================================
REM Kubera one-click launcher.
REM   Double-click this file to start the API + frontend + backup daemon.
REM   Then visit http://localhost:5173 in your browser.
REM
REM   You can close any of the three popped-up windows to stop that piece.
REM   Run stop_kubera.cmd to stop everything at once.
REM ===========================================================================

setlocal
set "ROOT=C:\dev\ta-agent"
set "VENV_PY=%ROOT%\.venv\Scripts\python.exe"
set "FRONTEND=%ROOT%\services\frontend"

REM --- Sanity checks ---------------------------------------------------------
if not exist "%VENV_PY%" (
    echo.
    echo ERROR: Python venv not found at:
    echo   %VENV_PY%
    echo.
    echo Did you check out the project to a different folder? Edit the ROOT
    echo line at the top of this script to point at the right place.
    pause
    exit /b 1
)
if not exist "%FRONTEND%\package.json" (
    echo.
    echo ERROR: Frontend not found at:
    echo   %FRONTEND%\package.json
    pause
    exit /b 1
)

echo.
echo Starting Kubera...
echo.

REM --- API ------------------------------------------------------------------
REM Run uvicorn in an auto-restart loop so a transient crash doesn't kill it.
REM (See api_loop.cmd — keeps a small ":loop / goto loop" wrapper.)
start "Kubera-API" /min cmd /k "cd /d %ROOT% && call %ROOT%\scripts\windows\api_loop.cmd"

REM --- Frontend (Vite dev server) -------------------------------------------
start "Kubera-Frontend" /min cmd /k "cd /d %FRONTEND% && npm run dev"

REM --- Backup daemon --------------------------------------------------------
start "Kubera-Backup" /min cmd /k "cd /d %ROOT% && %VENV_PY% -m scripts.backup_predictions_loop --interval 1800 --keep 24"

echo Launched three minimized windows:
echo   Kubera-API       (FastAPI on http://localhost:8000)
echo   Kubera-Frontend  (dashboard on http://localhost:5173)
echo   Kubera-Backup    (snapshots predictions.sqlite every 30 min)
echo.

REM Give Vite a few seconds to spin up before opening the browser.
timeout /t 5 /nobreak >nul
start "" "http://localhost:5173"

echo Dashboard opened in your default browser.
echo This window will close in 3 seconds.
timeout /t 3 /nobreak >nul
endlocal
exit /b 0
