# ta-agent

Technical-analysis ML agent for **S&P 500** and **NIFTY 100** universes.

- **Bars:** daily
- **Horizon:** 1-week forward returns
- **Models:** LightGBM regression + classification per universe (4 models total)
- **Data sources:** Interactive Brokers (US), Kite Connect (India), yfinance (sanity check)
- **Survivorship-bias-free:** point-in-time index membership

## Status

- [x] Project scaffolding
- [x] DuckDB storage layer with idempotent upserts
- [x] Point-in-time S&P 500 membership (from Wikipedia)
- [x] NIFTY 100 current-constituents loader (Phase A)
- [ ] IB adapter (US daily bars)
- [ ] Kite adapter (India daily bars)
- [ ] yfinance adapter (backup)
- [ ] Corporate actions adjustment
- [ ] Feature engineering
- [ ] Label generation
- [ ] Purged walk-forward CV + LightGBM training
- [ ] FastAPI backend
- [ ] React frontend

## Setup (Windows native)

### 1. Install uv

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart the terminal so `uv` is on PATH.

### 2. Install Python and create the project venv

```powershell
cd C:\path\to\ta-agent
uv python install 3.11
uv venv --python 3.11
.venv\Scripts\activate
uv pip install -e ".[dev]"
```

### 3. Configure secrets

Create a `.env` file in the project root:

```
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1

KITE_API_KEY=your_key_here
KITE_API_SECRET=your_secret_here
KITE_ACCESS_TOKEN=
```

Never commit `.env` (it's in `.gitignore`).

### Kite authentication

Zerodha's Kite Connect access tokens **expire daily at ~6am IST**. Refreshing
requires interactive browser login — this cannot be automated.

To mint a fresh token:

```powershell
python -m scripts.kite_login
```

The script will:
1. Print a login URL — open it in any browser.
2. After you log in, Zerodha redirects to your registered redirect URL with
   a `request_token=...` query parameter. Copy that value.
3. Paste the `request_token` back at the prompt.
4. The script exchanges it and prints a `KITE_ACCESS_TOKEN=...` line.

Paste that line into `.env` (replacing any previous `KITE_ACCESS_TOKEN`),
then re-run whatever command needed it. Backfills and `daily_update` will
abort with a clear log message if the token is missing or rejected — they
never try to auto-refresh.

### 4. Run the tests

```powershell
pytest -v
```

### 5. Build point-in-time index membership

```powershell
python -m scripts.refresh_universes --show
```

This scrapes Wikipedia for S&P 500 history and the niftyindices.com CSV for
NIFTY 100, then writes everything into `data/processed/market.duckdb`.

You only need to run this once a quarter or so (when index reconstitutions
happen) — but it's also safe to run any time, since upserts are idempotent.

### 6. Macro series (optional)

The pipeline registers macro features (VIX, USD/INR) **only if** the
`macro_daily` table has data. To populate it:

```powershell
python -m scripts.refresh_macro
```

This pulls `^VIX` and `INR=X` from yfinance into a separate DuckDB table.
Once present, three macro features appear on every (symbol, bar_date) row:
`macro__vix_level_z_252`, `macro__vix_chg_5d`, `macro__fx_ret_5d`.

### 7. Daily ingest

The unified `daily_ingest` job pulls fresh bars from IB (S&P 500), Kite
(NIFTY 100), and falls back to yfinance for any symbols that fail in their
primary source:

```powershell
python -m jobs.daily_ingest
```

Exit codes:
- `0` — clean run (or skipped because today is not a trading day)
- `1` — unexpected exception
- `2` — coverage below 90% across the run
- `3` — total per-symbol exceptions exceeded 50

The job is idempotent and importable: `from jobs.daily_ingest import run`
makes it scheduler-friendly (APScheduler integration in Phase 10).

To audit inter-source price disagreements (potential split / dividend
adjustment errors):

```powershell
python -m scripts.audit_corporate_actions --universe SP500 --lookback 365
```

### 8. Build features and labels

Generate the technical-feature panel:

```powershell
python -m scripts.build_features --universe SP500 --start 2014-01-01 --end 2024-12-31
```

Then assemble the master training dataset (features + labels):

```powershell
python -m scripts.build_dataset --universe SP500 --start 2014-01-01 --end 2024-12-31 --horizon 5
```

The output parquet has `symbol, bar_date, <feature columns>, fwd_return_5d,
fwd_quintile_5d, in_universe`. Modeling code must filter to
`in_universe == True` AND non-null labels before training.

### 9. Train models

Once you have a training dataset on disk, run purged walk-forward CV +
LightGBM training. Two models per universe (regression on the next 5-day
log return, and classification on the cross-sectional quintile):

```powershell
python -m scripts.train_models --universe SP500 --target regression --dataset data/processed/training_sp500_2014-01-01_2024-12-31.parquet --tune --n-trials 50

python -m scripts.train_models --universe SP500 --target classification --dataset data/processed/training_sp500_2014-01-01_2024-12-31.parquet --tune --n-trials 50
```

What runs:
- 5-fold purged walk-forward CV with a 5-day embargo (no label leakage)
- Optional Optuna search (`--tune`) over the 7 main hyperparameters
- A final production model trained on data through `today - 60 days`,
  with the last 60 days as the early-stopping holdout
- For classification: per-class isotonic calibration on a slice strictly
  before the early-stopping window
- Model + metadata + feature_importance.csv saved under `data/models/`

If the model produces IC > 0.15, hit-rate > 65%, or top-bottom decile
spread > 2%/week — STOP. Those are red-flag levels for retail equities;
look for a leakage bug.

### 10. Daily predict

After Phase 6 has produced trained models for both universes, the
prediction job loads the latest models, builds inference features for
current members, predicts, and persists to a SQLite predictions log:

```powershell
python -m jobs.daily_predict
```

What runs:
- Settles any open predictions whose 5-day horizon has now closed
  (computes realized log return + cross-sectional realized quintile)
- Skips per-universe automatically on non-trading days (NYSE / NSE)
- Loads latest registered regression + classification models
- Builds inference features through `as_of` and validates that every
  feature the model expects is present, in the right order
- Logs predictions to `data/processed/predictions.sqlite` (idempotent —
  re-running the same day overwrites prediction columns but preserves
  any already-realized fields)
- Prints top-N long / short picks per universe

For SHAP attributions on individual picks, use
`packages.inference.explain.explain_predictions(...)` from a notebook or
the API server (Phase 8).

### 11. Query membership at any past date

```python
from packages.ingestion.universe.membership import members_on
print(members_on("SP500", "2018-06-15").head())
```

## Project layout

```
packages/
  common/          # config, logging, schemas — shared by everything
  ingestion/       # data adapters + storage + universe membership
  features/        # technical indicator feature engineering
  labels/          # forward returns + classification labels
  modeling/        # CV, training, calibration, evaluation
  inference/       # daily prediction + SHAP attribution
services/
  api/             # FastAPI backend
  frontend/        # React + Vite frontend
jobs/              # scheduled daily/monthly tasks
configs/           # YAML configs (universes, models)
scripts/           # one-shot CLI utilities
tests/             # unit + integration tests
data/              # local data (gitignored)
  raw/             # adapter dumps
  processed/       # DuckDB + parquet
  models/          # serialized LightGBM models + metadata
```

## Realistic expectations

This is a research project, not a money-printing machine. Calibrated targets:

| Metric | Decent | Good | Suspicious (check leakage) |
| --- | --- | --- | --- |
| Information Coefficient (IC) | 0.02–0.05 | 0.05–0.08 | > 0.08 |
| Top-decile minus bottom-decile weekly return spread | 0.3% | 0.5–0.8% | > 1.5% |
| Directional hit rate | 52–55% | 55–58% | > 60% |

Always paper-trade for 3–6 months before risking real capital, and compare
live predictions against realized outcomes. None of this output constitutes
investment advice.
