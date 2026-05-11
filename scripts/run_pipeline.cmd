@echo off
REM Run the ta-agent daily pipeline once. Used by the Windows Scheduled
REM Tasks registered in scripts\register_windows_tasks.ps1.
REM
REM Important: this requires the venv's Python interpreter to live OUTSIDE
REM the user profile (e.g., at C:\Python311\). Task Scheduler-launched
REM processes can't reliably see paths under %APPDATA%\Roaming\... even
REM running as the same user — likely a USERPROFILE/LoadUserProfile quirk
REM exacerbated by OneDrive on this box. The .venv\Scripts\python.exe
REM trampoline reads pyvenv.cfg and tries to exec the cached interpreter,
REM which fails when that interpreter is at %APPDATA%\Roaming\uv\...
REM
REM Setup is documented in README; tldr: we copied the cached uv Python
REM to C:\Python311\ and recreated the venv with `uv venv --python C:\Python311\python.exe`.

cd /d C:\dev\ta-agent

REM Date-stamped log file for this run.
set "LOG=C:\dev\ta-agent\logs\scheduled_run_%date:~10,4%-%date:~4,2%-%date:~7,2%.log"

echo. >> "%LOG%"
echo ===== %date% %time% : pipeline run starting ===== >> "%LOG%"

REM Activate the venv. activate.bat puts .venv\Scripts on PATH so
REM `python` resolves to the trampoline, which exec's C:\Python311\python.exe
REM (visible to Task Scheduler context).
call "C:\dev\ta-agent\.venv\Scripts\activate.bat"

python -m jobs.run_pipeline >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"

echo ===== %date% %time% : pipeline run finished, rc=%RC% ===== >> "%LOG%"
exit /b %RC%
