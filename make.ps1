# PowerShell-friendly task runner. Usage:
#
#   .\make.ps1 setup
#   .\make.ps1 test
#   .\make.ps1 health
#
# Mirrors Makefile targets one-for-one — see Makefile for descriptions.

param(
    [Parameter(Position = 0)]
    [string]$Target = "help"
)

$ErrorActionPreference = "Stop"
$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$PY = Join-Path $PROJECT_ROOT ".venv\Scripts\python.exe"

function Invoke-Help {
    Write-Host "Targets:"
    Write-Host "  setup             Create .venv (uv) and install deps"
    Write-Host "  test              Run pytest (deselects integration)"
    Write-Host "  lint              Run ruff check"
    Write-Host "  ruff-fix          Run ruff check --fix"
    Write-Host "  ingest            python -m jobs.daily_ingest"
    Write-Host "  predict           python -m jobs.daily_predict"
    Write-Host "  train             Train all four (universe x target) models with --tune"
    Write-Host "  health            python -m scripts.healthcheck"
    Write-Host "  api               uvicorn services.api.main:app --reload"
    Write-Host "  frontend          npm run dev (services/frontend)"
    Write-Host "  frontend-build    npm run build (services/frontend)"
    Write-Host "  scheduler         python -m jobs.scheduler"
    Write-Host "  retrain           python -m jobs.monthly_retrain"
    Write-Host "  refresh-universes python -m scripts.refresh_universes --show"
    Write-Host "  refresh-macro     python -m scripts.refresh_macro"
}

switch ($Target.ToLower()) {
    "help"            { Invoke-Help }
    "setup" {
        uv venv --python 3.11
        uv pip install -e ".[dev]"
    }
    "test"            { & $PY -m pytest -v }
    "lint"            { & $PY -m ruff check packages/ tests/ scripts/ jobs/ services/ }
    "ruff-fix"        { & $PY -m ruff check --fix packages/ tests/ scripts/ jobs/ services/ }
    "ingest"          { & $PY -m jobs.daily_ingest }
    "predict"         { & $PY -m jobs.daily_predict }
    "train" {
        & $PY -m scripts.train_models --universe SP500 --target regression --tune --dataset data/processed/training_sp500.parquet
        & $PY -m scripts.train_models --universe SP500 --target classification --tune --dataset data/processed/training_sp500.parquet
        & $PY -m scripts.train_models --universe NIFTY100 --target regression --tune --dataset data/processed/training_nifty100.parquet
        & $PY -m scripts.train_models --universe NIFTY100 --target classification --tune --dataset data/processed/training_nifty100.parquet
    }
    "health"          { & $PY -m scripts.healthcheck }
    "api"             { & $PY -m uvicorn services.api.main:app --reload --host 0.0.0.0 --port 8000 }
    "frontend" {
        Push-Location (Join-Path $PROJECT_ROOT "services\frontend")
        try { npm run dev } finally { Pop-Location }
    }
    "frontend-build" {
        Push-Location (Join-Path $PROJECT_ROOT "services\frontend")
        try { npm run build } finally { Pop-Location }
    }
    "scheduler"       { & $PY -m jobs.scheduler }
    "retrain"         { & $PY -m jobs.monthly_retrain }
    "refresh-universes" { & $PY -m scripts.refresh_universes --show }
    "refresh-macro"   { & $PY -m scripts.refresh_macro }
    default {
        Write-Host "Unknown target: $Target" -ForegroundColor Red
        Invoke-Help
        exit 1
    }
}
