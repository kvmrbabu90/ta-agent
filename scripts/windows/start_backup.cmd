@echo off
REM Detached launcher for the predictions backup daemon.
cd /d C:\dev\ta-agent
.venv\Scripts\python.exe -m scripts.backup_predictions_loop --interval 1800 --keep 24 >> logs\backup.log 2>&1
