@echo off
REM Session-0-safe detached API launcher (no title/timeout console deps).
REM Auto-restart loop using ping as the delay. Logs to logs\api.log.
cd /d C:\dev\ta-agent
:loop
.venv\Scripts\python.exe -m uvicorn services.api.main:app --host 0.0.0.0 --port 8000 >> logs\api.log 2>&1
ping -n 6 127.0.0.1 >nul
goto loop
