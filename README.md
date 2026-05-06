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

### 6. Query membership at any past date

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
