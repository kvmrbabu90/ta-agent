@echo off
REM Monthly retrain — refreshes model weights using cached Optuna
REM hyperparameters from the most recent quarterly tune. ~5 min/universe.
REM Fired on the 1st weekday of each month by Task Scheduler (see
REM register_windows_tasks.ps1).

cd /d C:\dev\ta-agent

set "LOG=C:\dev\ta-agent\logs\monthly_retrain_%date:~10,4%-%date:~4,2%-%date:~7,2%.log"

echo. >> "%LOG%"
echo ===== %date% %time% : monthly retrain starting ===== >> "%LOG%"

call "C:\dev\ta-agent\.venv\Scripts\activate.bat"

REM do_tune defaults to False (cached hyperparams); quarterly job re-runs
REM the search. --only-if-first-business-day-of-month: Windows Task
REM Scheduler fires this weekly, but the gate ensures we only actually
REM retrain on the 1st Mon-Fri of the calendar month.
python -m jobs.monthly_retrain --only-if-first-business-day-of-month >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"

echo ===== %date% %time% : monthly retrain finished, rc=%RC% ===== >> "%LOG%"
exit /b %RC%
