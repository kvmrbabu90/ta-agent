@echo off
REM ===========================================================================
REM Live-design walk-forward backtest (gated, 2014-2026).
REM
REM Runs FULLY DETACHED from any terminal/Claude session. Double-click this
REM file to start OR RESUME the run — the WF skips retrain windows whose
REM predictions already exist, so re-running after any interruption picks up
REM where it left off. Output: data\processed\wf_gatetest_gated\.
REM
REM Matches the deployed system: 10y lookback, 60d train-end gap, quarterly
REM Optuna tune cadence (reuse monthly), promote/retain gate (both heads).
REM ===========================================================================
cd /d C:\dev\ta-agent

REM Cap per-process OpenMP threads so the parallel Optuna trials (n_jobs=4)
REM don't oversubscribe the 20 physical cores: 4 trials x 5 threads = 20.
set OMP_NUM_THREADS=5

echo. >> logs\wf_gated.log
echo ===== %date% %time% : wf_gated run starting/resuming ===== >> logs\wf_gated.log

.venv\Scripts\python.exe -m scripts.walkforward_backtest ^
    --universe SP500 --start 2014-01-01 --end 2026-05-31 ^
    --device cpu --gate --live-tune-cadence ^
    --train-lookback-years 10 --train-end-gap-days 60 ^
    --optuna-n-jobs 4 ^
    --out-dir data\processed\wf_gatetest_gated >> logs\wf_gated.log 2>&1

echo ===== %date% %time% : wf_gated run exited rc=%ERRORLEVEL% ===== >> logs\wf_gated.log
REM No pause: this script is launched detached (survives terminal/Claude
REM close). Monitor via logs\wf_gated.log or the Live WF dashboard tab.
exit /b %ERRORLEVEL%
