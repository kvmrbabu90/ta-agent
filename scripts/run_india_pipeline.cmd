@echo off
REM Run the ta-agent INDIA daily pipeline once. Used by the Windows
REM Scheduled Task `ta-agent-pipeline-india-am-ct` (fires at 6:00 AM CT
REM weekdays, ~1-2 hours after NSE close).
REM
REM Mirrors run_pipeline.cmd (US). See that file for the venv / path
REM rationale.

cd /d C:\dev\ta-agent

REM Date-stamped log file for this run (separate from US logs so an
REM India-side failure doesn't get buried under US output).
set "LOG=C:\dev\ta-agent\logs\scheduled_india_run_%date:~10,4%-%date:~4,2%-%date:~7,2%.log"

echo. >> "%LOG%"
echo ===== %date% %time% : india pipeline run starting ===== >> "%LOG%"

call "C:\dev\ta-agent\.venv\Scripts\activate.bat"

python -m jobs.run_india_pipeline >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"

echo ===== %date% %time% : india pipeline run finished, rc=%RC% ===== >> "%LOG%"
exit /b %RC%
