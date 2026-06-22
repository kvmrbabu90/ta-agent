@echo off
REM Direct (no-window) detached launcher for the APScheduler daemon.
REM Fires the 08:35 + 17:00 CT daily pipeline. Logs to logs\scheduler.log.
cd /d C:\dev\ta-agent
.venv\Scripts\python.exe -m jobs.scheduler >> logs\scheduler.log 2>&1
