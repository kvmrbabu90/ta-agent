# Product Requirements Document
## ta-agent: Technical Analysis ML Agent for US & Indian Equities

**Version:** 1.0
**Status:** Draft, Phase 1 partially complete
**Owner:** [Your name]

---

## 1. Vision

A locally-hosted machine learning agent that ingests daily bar data for the
S&P 500 and NIFTY 100 universes, engineers technical features, trains
LightGBM models to predict 1-week forward stock performance, and serves
ranked daily picks via a FastAPI + React interface.

The system must be **scientifically honest**: free of survivorship bias,
free of look-ahead leakage, with calibrated probabilities and realistic
performance expectations.

---

## 2. Goals & non-goals

### 2.1 Goals
- Predict 1-week forward log returns for every stock in each universe, every trading day
- Provide both regression (continuous return) and classification (top vs bottom quintile) outputs
- Deliver calibrated confidence/probability estimates
- Surface daily ranked picks per universe through a web UI
- Track live prediction performance to validate model edge over time
- Run end-to-end on a single Windows PC with reasonable hardware (16GB RAM, 8 cores)

### 2.2 Non-goals (v1)
- Live order execution / automated trading
- Intraday predictions (2-hour horizons explicitly dropped)
- Options, futures, or non-equity instruments
- Sub-daily bar granularity
- Cloud deployment / multi-user access
- Backtesting framework with transaction costs and slippage modeling (deferred to v2)
- Portfolio construction / position sizing logic (deferred to v2)

---

## 3. Success criteria

The project is considered successful if it meets ALL of the following:

| Criterion | Target |
|---|---|
| End-to-end pipeline runs unattended | Daily ingest + predict completes in < 30 min |
| Survivorship bias eliminated | Point-in-time membership for both universes |
| Look-ahead bias eliminated | Purged walk-forward CV with embargo period |
| Out-of-sample IC (information coefficient) | > 0.02 on held-out years |
| Top-decile vs bottom-decile weekly spread | > 0.3% on held-out years |
| Probability calibration | Brier skill score > 0 vs naive baseline |
| API response time | < 200ms for top picks endpoint |
| Test coverage | > 70% on packages/ ; 100% on splits/labels |

If IC or spread targets are not met, the project still ships — the model
becomes a research baseline rather than a tradeable signal.

---

## 4. Users & use cases

**Primary user:** Single technical-analyst trader (the project owner).

**Use cases:**
1. Each morning before market open, view top 20 ranked picks in each universe
2. Drill into any stock to see SHAP feature attribution explaining the prediction
3. Review prediction performance over time (live tracking dashboard)
4. Trigger model retraining manually when desired

---

## 5. Functional requirements

### 5.1 Data ingestion
- **FR-1.1** System ingests daily OHLCV bars from Interactive Brokers API (US universe)
- **FR-1.2** System ingests daily OHLCV bars from Kite Connect API (India universe)
- **FR-1.3** yfinance serves as a backup / sanity-check source
- **FR-1.4** All bars are split- and dividend-adjusted at storage time
- **FR-1.5** Ingestion is idempotent — re-running for the same date range produces no duplicates and no corruption
- **FR-1.6** Adapters respect rate limits (IB: ~60 req/10min; Kite: 3 req/sec)
- **FR-1.7** A single `daily_ingest.py` job pulls latest bars for all current members of both universes
- **FR-1.8** Backfill mode supports up to 10+ years of history per symbol where available

### 5.2 Universe management
- **FR-2.1** Point-in-time S&P 500 membership reconstructed from Wikipedia change history
- **FR-2.2** NIFTY 100 starts with current snapshot (Phase A); historical reconstruction in Phase B
- **FR-2.3** `members_on(universe, date)` returns exactly the set of symbols in the index on that date
- **FR-2.4** Delisted/removed tickers retain price history up to their removal date

### 5.3 Feature engineering
- **FR-3.1** ~40-60 technical features computed per (symbol, date) row
- **FR-3.2** Feature categories: returns, trend, momentum, volatility, volume, microstructure, cross-sectional, regime
- **FR-3.3** All features are strictly causal — no future information leaks into a row dated T
- **FR-3.4** Cross-sectional features (e.g. RSI rank within universe) computed using only stocks that were members on date T
- **FR-3.5** Features handle missing data gracefully (NaN propagation, not silent imputation)
- **FR-3.6** Feature pipeline is deterministic and reproducible

### 5.4 Labels
- **FR-4.1** Regression target: 5-day forward log return = log(close[t+5] / close[t])
- **FR-4.2** Classification target: cross-sectional quintile of regression target within universe on day t
- **FR-4.3** Labels are NaN for the most recent 5 trading days (cannot be computed yet)
- **FR-4.4** Labels respect point-in-time membership for cross-sectional ranking

### 5.5 Modeling
- **FR-5.1** Two LightGBM models per universe: regression and 5-class classification
- **FR-5.2** Purged walk-forward cross-validation with 5-day embargo
- **FR-5.3** Optuna-based hyperparameter tuning over the same CV folds
- **FR-5.4** Probability calibration via isotonic regression on a held-out slice
- **FR-5.5** Evaluation metrics: IC, rank IC, hit rate, top-bottom decile spread, Brier score
- **FR-5.6** Model artifacts versioned and stored with full metadata (features used, hyperparams, CV scores, training date)

### 5.6 Inference
- **FR-6.1** Daily batch predict for all current members of both universes
- **FR-6.2** Output: predicted return, predicted quintile, calibrated probability, SHAP top-5 features
- **FR-6.3** Predictions persisted to enable live performance tracking

### 5.7 API & frontend
- **FR-7.1** FastAPI backend exposes: top picks per universe, stock detail with SHAP, performance over time
- **FR-7.2** React + Vite + TypeScript frontend with a Dashboard, StockDetail, and Performance page
- **FR-7.3** API uses SQLite (or Postgres) for predictions and performance data; DuckDB stays for analytics

### 5.8 Scheduling
- **FR-8.1** Daily ingest job runs after market close in each timezone
- **FR-8.2** Daily predict job runs after ingest completes
- **FR-8.3** Monthly retrain job runs on first business day of each month
- **FR-8.4** Quarterly universe-membership refresh

---

## 6. Non-functional requirements

- **NFR-1** Local-only deployment, no cloud dependencies
- **NFR-2** All secrets in `.env`, never committed
- **NFR-3** Logs structured (Loguru) and rotated daily
- **NFR-4** Reproducible builds via `uv` lockfile
- **NFR-5** Type hints on all public functions; mypy clean for `packages/common` and `packages/modeling`
- **NFR-6** Pre-commit hooks: ruff, mypy on changed files
- **NFR-7** Every phase delivers passing tests before next phase begins

---

## 7. Tech stack

| Layer | Choice | Rationale |
|---|---|---|
| Package manager | uv | Fast, modern, replaces pip+venv+poetry |
| Storage (analytics) | DuckDB + Parquet | Embedded, fast, columnar, zero ops |
| Storage (transactional) | SQLite | API state, predictions log |
| ML | LightGBM | Best tabular performance, fast training |
| Hyperparameter tuning | Optuna | Tree-of-Parzen estimators, study persistence |
| Calibration | scikit-learn IsotonicRegression | Standard, well-understood |
| Explainability | SHAP TreeExplainer | Native LightGBM support |
| Backend | FastAPI + uvicorn | Async, OpenAPI docs free |
| Frontend | React + Vite + TypeScript | Standard production stack |
| Charts | Recharts | Good React integration |
| Logging | Loguru | Better than stdlib logging |
| Config | pydantic-settings + YAML | Type-safe |
| Scheduling | APScheduler | Cross-platform, no separate process |

---

## 8. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Look-ahead leakage in features | High | Critical | Purged CV, leakage tests, code review |
| Survivorship bias in NIFTY 100 | Medium | High | Phase B historical reconstitution |
| Wikipedia layout changes break scraper | Medium | Medium | Defensive parsing, monthly automated test |
| IB API outage / rate limits | Medium | Medium | yfinance fallback, retry logic |
| Model overfits to recent regime | High | Medium | Walk-forward CV across many regimes |
| Probability miscalibration | High | Medium | Isotonic calibration on held-out slice |
| Disk space (10 yr × 600 stocks daily ≈ 1.5M rows) | Low | Low | Parquet compression, partitioning |

---

## 9. Out of scope deferred to v2

- Real-time intraday rebalancing
- Portfolio optimization (Markowitz, Black-Litterman, etc.)
- Transaction cost modeling
- Multi-user access / authentication
- Cloud deployment
- Mobile UI
- Alerting / notifications
- Alternative data sources (sentiment, fundamentals, news)
- Ensemble of multiple model families (XGBoost, neural nets)
