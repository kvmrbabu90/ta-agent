@echo off
REM ===========================================================================
REM Extend the BASELINE (V1) walk-forward backtest forward by resuming it.
REM
REM Matches the locked baseline recipe exactly (verified from wf_10yr.log):
REM   --per-retrain-optuna   strict, look-ahead-free per-retrain tuning
REM   --train-lookback-years 10
REM   --train-end-gap-days 6 (the look-ahead-free default; NOT the deployed 60)
REM   NO --gate, NO --live-tune-cadence (V1 = always deploy fresh model)
REM   monthly retrain cadence, 20 Optuna trials/head.
REM
REM Resume-safe: skips every retrain window whose predictions already exist,
REM so this only computes the new month(s) up to --end. Runs FULLY DETACHED
REM (survives terminal/Claude close). Monitor via logs\wf_extend.log or the
REM Live WF dashboard (baseline variant auto-refreshes once June lands).
REM ===========================================================================
cd /d C:\dev\ta-agent

REM 4 parallel Optuna trials x 5 OpenMP threads = 20 physical cores.
set OMP_NUM_THREADS=5

echo. >> logs\wf_extend.log
echo ===== %date% %time% : wf_extend (baseline -> 2026-06) starting/resuming ===== >> logs\wf_extend.log

.venv\Scripts\python.exe -m scripts.walkforward_backtest ^
    --universe SP500 --start 2014-01-01 --end 2026-06-30 ^
    --device cpu --per-retrain-optuna ^
    --train-lookback-years 10 --train-end-gap-days 6 ^
    --optuna-trials 20 --optuna-n-jobs 4 ^
    --out-dir data\processed\walkforward_10yr_strict >> logs\wf_extend.log 2>&1

echo ===== %date% %time% : wf_extend exited rc=%ERRORLEVEL% ===== >> logs\wf_extend.log
exit /b %ERRORLEVEL%
