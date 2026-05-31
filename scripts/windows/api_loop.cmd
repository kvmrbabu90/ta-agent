@echo off
REM Auto-restart wrapper for the FastAPI server.
REM Called from start_kubera.cmd. If uvicorn crashes for any reason
REM (asyncio glitch, transient DuckDB lock, etc.) we wait 5 s and relaunch
REM so the dashboard stays online without you touching anything.

setlocal
set "ROOT=C:\dev\ta-agent"
set "VENV_PY=%ROOT%\.venv\Scripts\python.exe"

title Kubera-API

:loop
echo [%date% %time%] Kubera API starting...
"%VENV_PY%" -m uvicorn services.api.main:app --host 0.0.0.0 --port 8000
echo.
echo [%date% %time%] Kubera API exited (code %errorlevel%). Restarting in 5s...
timeout /t 5 /nobreak >nul
goto loop
