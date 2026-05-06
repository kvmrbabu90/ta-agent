# Common tasks. POSIX make (and GNU make on Windows / WSL).
# For Windows-native shells, see make.ps1.

PYTHON ?= .venv/Scripts/python.exe
UNIVERSE ?= SP500

.PHONY: help setup test lint ruff-fix \
        ingest predict train health \
        api frontend frontend-build \
        scheduler retrain refresh-universes refresh-macro \
        clean

help:
	@echo "Targets:"
	@echo "  setup             Create .venv (uv) and install deps"
	@echo "  test              Run pytest (deselects integration)"
	@echo "  lint              Run ruff check"
	@echo "  ruff-fix          Run ruff check --fix"
	@echo "  ingest            python -m jobs.daily_ingest"
	@echo "  predict           python -m jobs.daily_predict"
	@echo "  train             Train both targets for both universes (with --tune)"
	@echo "  health            python -m scripts.healthcheck"
	@echo "  api               uvicorn services.api.main:app --reload"
	@echo "  frontend          npm run dev (services/frontend)"
	@echo "  frontend-build    npm run build (services/frontend)"
	@echo "  scheduler         python -m jobs.scheduler"
	@echo "  retrain           python -m jobs.monthly_retrain"
	@echo "  refresh-universes python -m scripts.refresh_universes"
	@echo "  refresh-macro     python -m scripts.refresh_macro"

setup:
	uv venv --python 3.11
	uv pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -v

lint:
	$(PYTHON) -m ruff check packages/ tests/ scripts/ jobs/ services/

ruff-fix:
	$(PYTHON) -m ruff check --fix packages/ tests/ scripts/ jobs/ services/

ingest:
	$(PYTHON) -m jobs.daily_ingest

predict:
	$(PYTHON) -m jobs.daily_predict

train:
	$(PYTHON) -m scripts.train_models --universe SP500 --target regression --tune --dataset data/processed/training_sp500.parquet
	$(PYTHON) -m scripts.train_models --universe SP500 --target classification --tune --dataset data/processed/training_sp500.parquet
	$(PYTHON) -m scripts.train_models --universe NIFTY100 --target regression --tune --dataset data/processed/training_nifty100.parquet
	$(PYTHON) -m scripts.train_models --universe NIFTY100 --target classification --tune --dataset data/processed/training_nifty100.parquet

health:
	$(PYTHON) -m scripts.healthcheck

api:
	$(PYTHON) -m uvicorn services.api.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd services/frontend && npm run dev

frontend-build:
	cd services/frontend && npm run build

scheduler:
	$(PYTHON) -m jobs.scheduler

retrain:
	$(PYTHON) -m jobs.monthly_retrain

refresh-universes:
	$(PYTHON) -m scripts.refresh_universes --show

refresh-macro:
	$(PYTHON) -m scripts.refresh_macro

clean:
	@echo "(this leaves data/ and models/ untouched)"
	rm -rf .pytest_cache .ruff_cache .mypy_cache services/frontend/dist services/frontend/.vite
