@echo off
REM Quarterly Optuna re-tune — runs the full hyperparameter search
REM (20 trials × ~10 min/trial) for each universe. Expensive (~3 hours
REM per universe) but rare: 4 times per year. Refreshes the cached
REM hyperparameters that the cheap monthly retrain then reuses for the
REM next 3 months.
REM
REM Fired on the 1st weekday of Jan/Apr/Jul/Oct by Task Scheduler.

cd /d C:\dev\ta-agent

set "LOG=C:\dev\ta-agent\logs\quarterly_retune_%date:~10,4%-%date:~4,2%-%date:~7,2%.log"

echo. >> "%LOG%"
echo ===== %date% %time% : quarterly Optuna re-tune starting ===== >> "%LOG%"

call "C:\dev\ta-agent\.venv\Scripts\activate.bat"

REM --do-tune: re-run Optuna search.
REM --only-if-first-business-day-of-quarter: Windows Task Scheduler fires
REM weekly, but this gate ensures we only re-tune on 1st Mon-Fri of
REM Jan/Apr/Jul/Oct (quarterly cadence).
python -m jobs.monthly_retrain --do-tune --n-trials 20 ^
    --only-if-first-business-day-of-quarter >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"

echo ===== %date% %time% : quarterly re-tune finished, rc=%RC% ===== >> "%LOG%"
exit /b %RC%
