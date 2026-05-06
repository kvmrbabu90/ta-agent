# ta-agent

**Local-only ML pipeline for ranking S&P 500 and NIFTY 100 stocks by predicted 5-day forward return.**

Daily bars in, ranked picks out. Two LightGBM models per universe — one
regression on 5-day forward log returns, one classification into
cross-sectional quintiles — served by a FastAPI backend and a small React
dashboard. Designed to be **scientifically honest**: survivorship-bias-free
universes, look-ahead-free features, purged walk-forward CV with embargo,
calibrated probabilities, and red-flag thresholds for "too good" metrics.

> ⚠️ This is a research project, not investment advice. See the disclaimer at the bottom.

## Status

- [x] Data ingestion: Interactive Brokers (US), Kite Connect (NSE), yfinance (fallback + macro)
- [x] Point-in-time S&P 500 membership; current-snapshot NIFTY 100 (PIT for India deferred)
- [x] DuckDB storage with idempotent upserts
- [x] Corporate-actions cross-source audit
- [x] ~100 technical features across 11 groups (price, trend, momentum, volatility, volume, microstructure, market structure, cross-sectional, regime, volume profile, swings/Fib) + optional macro
- [x] Forward returns and cross-sectional quintile labels (PIT-respecting)
- [x] Purged walk-forward CV, LightGBM training, Optuna tuning, isotonic calibration, file-based model registry
- [x] Daily inference + SHAP top-K + SQLite predictions log + automatic settlement
- [x] FastAPI backend (read-only) at `:8000` with OpenAPI docs at `/docs`
- [x] React + Vite + TypeScript frontend at `:5173` (Dashboard / Stock detail / Performance)
- [x] APScheduler-based orchestrator + monthly retrain with promote/retain
- [x] Health-check script + ops scripts (freshen, backtest summary)
- [x] 140 unit tests + 4 integration tests

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for diagrams and details.

```
data/processed/
    market.duckdb          (OHLCV + membership + macro)
    predictions.sqlite     (predictions log)
    features_*.parquet     (wide feature panel per universe)
    training_*.parquet     (features + labels + in_universe)
data/models/
    {universe}_{target}_{ts}/   (model.txt, calibrators.pkl, metadata.json, feature_importance.csv)
    retrain_reports/{date}.json (promote/retain decisions)
```

## Setup (Windows native)

### 1. Install `uv`

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart the shell so `uv` is on PATH.

### 2. Install Python and create the project venv

```powershell
cd C:\path\to\ta-agent
uv python install 3.11
uv venv --python 3.11
.venv\Scripts\activate
uv pip install -e ".[dev]"
```

### 3. (Optional) install pre-commit hooks

```powershell
pip install pre-commit
pre-commit install
```

### 4. Configure secrets

Copy `.env.example` to `.env` and fill in:

```
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1

KITE_API_KEY=your_key_here
KITE_API_SECRET=your_secret_here
KITE_ACCESS_TOKEN=                   # blank — minted via scripts.kite_login
```

`.env` is gitignored. Don't commit it.

### Kite authentication

Zerodha access tokens **expire daily ~6am IST**. Re-mint with:

```powershell
python -m scripts.kite_login
```

The script prints a login URL → you log in in a browser → it asks for the
`request_token` from the redirect URL → it prints the new
`KITE_ACCESS_TOKEN` line to add to `.env`.

## First-time bootstrap

This is what you run once to take the system from empty to producing
ranked picks. After this, normal operation is just `python -m
jobs.scheduler` (or running daily jobs by hand).

```powershell
# 1. Membership tables
python -m scripts.refresh_universes --show

# 2. Macro series (optional but enables the macro feature group)
python -m scripts.refresh_macro

# 3. Backfill OHLCV (one universe at a time; takes hours per universe)
#    For the IB run, start TWS or IB Gateway in paper mode first.
python -m scripts.ib_backfill   --universe SP500    --start 2014-01-01 --end 2024-12-31
python -m scripts.kite_backfill --universe NIFTY100 --start 2014-01-01 --end 2024-12-31

# 4. Build the master training datasets (features + labels join)
python -m scripts.build_dataset --universe SP500    --start 2014-01-01 --end 2024-12-31
python -m scripts.build_dataset --universe NIFTY100 --start 2014-01-01 --end 2024-12-31

# 5. Train both targets for both universes (with Optuna --tune)
python -m scripts.train_models --universe SP500    --target regression     --dataset data/processed/training_sp500.parquet    --tune
python -m scripts.train_models --universe SP500    --target classification --dataset data/processed/training_sp500.parquet    --tune
python -m scripts.train_models --universe NIFTY100 --target regression     --dataset data/processed/training_nifty100.parquet --tune
python -m scripts.train_models --universe NIFTY100 --target classification --dataset data/processed/training_nifty100.parquet --tune

# 6. Generate today's predictions and confirm the system is healthy
python -m jobs.daily_predict
python -m scripts.healthcheck

# 7. Start the API and frontend (separate terminals)
uvicorn services.api.main:app --reload --host 0.0.0.0 --port 8000
cd services/frontend; npm install; npm run dev
```

Browse <http://localhost:5173>.

## Daily operation

After bootstrap, you have two options.

### Option A — let the scheduler run

```powershell
python -m jobs.scheduler
```

This is a long-running blocking process. It runs daily ingest after each
market close, daily predict shortly after, settles mature predictions,
retrains monthly on the first weekday, and refreshes universe membership
quarterly. Keep it running in a terminal you don't close. It logs heavily
to both stderr and `logs/ta_agent_{date}.log`. See
[`docs/architecture.md`](docs/architecture.md) for the full schedule.

### Option B — run jobs manually

```powershell
python -m jobs.daily_ingest
python -m jobs.daily_predict
```

In either mode, refresh the Kite token once a day:

```powershell
python -m scripts.kite_login
```

## Common tasks

POSIX `make`:

```bash
make help        # list targets
make test        # pytest
make lint        # ruff check
make ingest      # run daily_ingest
make predict     # run daily_predict
make health      # run healthcheck
make api         # run uvicorn
make frontend    # run vite dev
make scheduler   # run jobs.scheduler
make retrain     # run jobs.monthly_retrain
```

PowerShell-native (no `make`):

```powershell
.\make.ps1 help
.\make.ps1 test
.\make.ps1 health
# ... same target names
```

## Troubleshooting

**"No predictions for today" in the dashboard.**
The dashboard shows whatever the API returns from `MAX(as_of)` — meaning
the most recent date with logged predictions. If predictions are stale or
missing, run `python -m jobs.daily_predict`. If that errors, check
`python -m scripts.healthcheck` — it'll tell you whether OHLCV, membership,
or models are missing.

**`ConnectionRefusedError` from IB.**
TWS or IB Gateway isn't running, or the API socket isn't enabled.
- Start IB Gateway in paper mode (port 7497).
- In Configure → API → Settings, check **Enable ActiveX and Socket Clients**.
- Add `127.0.0.1` to **Trusted IPs**.

**`kiteconnect.exceptions.TokenException`.**
Your Kite access token has expired (they reset around 6am IST). Re-run
`python -m scripts.kite_login` and update `.env`. The ingest job aborts
cleanly on token expiry — no auto-refresh, on purpose.

**Tests fail after pulling new code.**
Run `uv pip install -e ".[dev]"` again to pick up new deps. If you still
see import errors, your `.venv` may be stale — recreate it:
`rm -rf .venv && uv venv --python 3.11 && uv pip install -e ".[dev]"`.

**Frontend can't reach the API.**
Confirm `uvicorn services.api.main:app` is running on port 8000. The
frontend reads `VITE_API_BASE_URL` from `services/frontend/.env.local`
(default `http://localhost:8000`). Both `:5173` and `:3000` are in the
backend's CORS allow-list.

**Memory / disk space.**
A full 10-year backfill of both universes is ~3M OHLCV rows. DuckDB
compresses this aggressively (~100–200 MB on disk). Trained model
artifacts are tiny (~5–10 MB each). The frontend dev server's
`node_modules` is the biggest item by far (~250 MB).

## Known limitations

1. **NIFTY 100 PIT membership is a current-only snapshot.** Pre-rebalance
   dates are therefore survivorship-biased on the India side. Documented
   in `01_PRD.md` as Phase B work; not done in v1.
2. **Earnings-window and news-sentiment features are not implemented.**
   Both have well-defined extension points in
   `packages/features/extensions.py`; pick a data provider (Finnhub, FMP,
   NewsAPI, …) and write the adapter to wire them in.
3. **Volume-profile features are a daily-bar approximation.** Real volume
   profile uses intraday data, which we don't ingest.
4. **`pandas-ta` is not a dependency.** All ~100 technical indicators are
   hand-rolled — `pandas-ta` requires Python 3.12+ and is effectively
   unmaintained.
5. **Frontend bundle is ~180 KB gzipped** (Recharts dominates). Acceptable
   for a single-user internal tool; route-level code splitting is the
   natural future lever.
6. **Single-user, local only.** No login, no auth, no multi-tenancy. The
   API is read-only; the frontend cannot trigger training or predictions.
7. **All times in the scheduler are pinned to UTC** to dodge DST. Local
   firing time drifts by ±1 hour across DST boundaries — well after market
   close in either timezone, so this is OK in practice.

## Realistic expectations

Calibrated targets for the held-out IC and decile spread:

| metric | decent | good | suspicious (look for leakage) |
|---|---|---|---|
| Information coefficient (IC) | 0.02–0.05 | 0.05–0.08 | > 0.08 |
| Top-decile minus bottom-decile weekly return spread | ~0.3% | 0.5–0.8% | > 1.5% |
| Directional hit rate | 52–55% | 55–58% | > 60% |

If a freshly-trained model lands in the "suspicious" column, **stop and
look for a leakage bug**. The PRD lists the usual culprits.

Always paper-trade for 3–6 months before risking real capital, and
compare live predictions against realized outcomes via the Performance
page.

## Pre-commit hooks

```powershell
pip install pre-commit
pre-commit install
pre-commit run --all-files   # one-time check before committing
```

The config is at `.pre-commit-config.yaml`: ruff (with `--fix` and
`ruff-format`), trailing whitespace, end-of-file fixer, YAML/JSON syntax
checks, large-file blocker, and merge-conflict / private-key detectors.

## Disclaimer

This software produces ranked stock predictions for personal research.
Predictions are not a recommendation to buy, sell, or hold any security.
Past model performance — even on held-out data — does not guarantee
future performance. The author is not a registered investment advisor.
Use at your own risk.
