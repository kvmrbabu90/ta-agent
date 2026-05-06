# ta-agent — architecture

A single-user, locally-deployed ML pipeline for ranking S&P 500 and
NIFTY 100 stocks by predicted 5-day forward return. Designed to be
**scientifically honest** rather than maximally clever: survivorship-bias
free universes, look-ahead-free features, purged walk-forward CV, and
explicit calibration.

## High-level data flow

```
              ┌─────────────────────────────────────────────┐
              │            data adapters                    │
   IB ────►   │  ib_adapter, kite_adapter, yfinance_adapter │ ─┐
   Kite ──►   │              + macro adapter (yfinance)     │  │
              └─────────────────────────────────────────────┘  │
                                                               ▼
                                  DuckDB (analytics)
                              ┌────────────────────────┐
                              │   ohlcv_daily          │
                              │   index_membership     │
                              │   macro_daily          │
                              └────────────┬───────────┘
                                           │
                  ┌────────────────────────┴─────────────────┐
                  │                                          │
                  ▼                                          ▼
    packages.features.pipeline           packages.labels.dataset
    (~100 features, PIT-masked)          (forward returns + quintiles)
                  │                                          │
                  └─────────────────────┬────────────────────┘
                                        ▼
                            training dataset (parquet)
                                        │
                                        ▼
                      packages.modeling
                      (purged walk-forward CV → LightGBM
                       → calibration → registry: data/models/)
                                        │
                                        ▼
                         packages.inference + jobs.daily_predict
                                        │
                                        ▼
                         SQLite predictions_log
                              ┌────────────────────┐
                              │  predictions_log   │
                              └─────────┬──────────┘
                                        │
                          ┌─────────────┴─────────────┐
                          ▼                           ▼
                   FastAPI (services/api)      Scheduler (APScheduler)
                          │
                          ▼
                  React frontend (services/frontend, Vite)
```

## Storage layout

| store | engine | path | what lives there |
|---|---|---|---|
| Market analytics | DuckDB | `data/processed/market.duckdb` | `ohlcv_daily`, `index_membership`, `macro_daily` |
| Predictions log | SQLite | `data/processed/predictions.sqlite` | `predictions_log` |
| Features panel | Parquet | `data/processed/features_*.parquet` | wide feature matrix per universe |
| Training datasets | Parquet | `data/processed/training_*.parquet` | features + labels + in_universe |
| Trained models | filesystem | `data/models/{universe}_{target}_{ts}/` | `model.txt`, `calibrators.pkl`, `metadata.json`, `feature_importance.csv` |
| Retrain reports | JSON | `data/models/retrain_reports/{date}.json` | promote/retain decisions |
| Logs | text | `logs/ta_agent_{date}.log` | rotated daily by loguru |

DuckDB and SQLite live in the same directory but serve different access
patterns: DuckDB is columnar / analytical (millions of OHLCV rows, range
scans), SQLite is row-oriented / transactional (insert + lookup of
predictions per symbol-date).

## Core invariants

These are enforced by code, not by convention:

1. **Survivorship-bias-free universes.** Every cross-sectional or
   universe-wide computation goes through `members_on(universe, date)`.
   Stocks are filtered to those that were in the index *on the bar's date*,
   never on a later date.

2. **No look-ahead in features.** Every per-symbol feature is built from
   strictly trailing windows. The mandatory test
   `tests/unit/test_features_causality.py` corrupts post-cutoff OHLCV with
   extreme values and asserts pre-cutoff features are byte-identical.

3. **Purged walk-forward CV.** `PurgedWalkForwardSplit` removes training
   rows whose label horizon overlaps the validation window AND embargoes a
   buffer immediately before val. No row in the training set carries
   information about validation outcomes.

4. **Trading-bar horizons, not calendar days.** Forward returns are
   computed via `groupby('symbol').shift(-N)` so weekends and holidays
   are skipped naturally. Settlement uses the same convention.

5. **Strict feature alignment at inference.** When `predict_universe`
   loads a model, it validates that every feature the model expects is
   present in the live feature matrix, and reorders them into the model's
   expected order before predicting.

## Component boundaries

```
packages/
├── common/          shared schemas, config, logging
├── ingestion/       data adapters + DuckDB layer + universe membership
│   ├── adapters/    ib_adapter, kite_adapter, yfinance_adapter
│   ├── universe/    SP500 PIT, NIFTY100 current snapshot
│   ├── corporate_actions.py    inter-source price audit
│   └── macro.py     VIX / FX series ingestion
├── features/        ~100 features across 11 groups (extensions registry)
├── labels/          forward returns + quintile labels + master dataset
├── modeling/        purged CV, LightGBM, Optuna, calibration, registry
└── inference/       predict, rank, explain (SHAP), tracking (log/settle)

services/
├── api/             FastAPI backend (read-only)
└── frontend/        React + Vite + TS + Recharts + Tailwind

jobs/
├── daily_ingest.py  IB + Kite + yfinance fallback
├── daily_predict.py predict + log + settle
├── monthly_retrain.py compare + promote/retain
└── scheduler.py     APScheduler wrapping the above
```

## Extension points

`packages/features/extensions.py` defines a registry-based plugin system
for optional feature groups. The macro feature group is currently the
only one wired up — it auto-registers when imported and only attaches to
the panel if `macro_daily` actually has rows. The same pattern is the
intended landing zone for v2 work like earnings windows or news
sentiment: write a data adapter, write a `FeatureGroup`, register an
extension wrapper, done — no edits to the pipeline.

## Trade-offs we accepted (call-outs for future work)

- **Predictions log lives in SQLite, OHLCV in DuckDB.** Joins between them
  go through pandas. For a single-user tool this is fine; if traffic grows
  we'd consolidate in one engine.
- **Volume-profile is daily-bar approximation.** Real volume profile is
  built from intraday data, which we don't ingest. Documented inline.
- **NIFTY 100 uses a current-only membership snapshot** (Phase A). The
  India universe is therefore survivorship-biased on pre-rebalance dates;
  PIT reconstruction is deferred.
- **Frontend bundle is ~180 KB gzipped** (Recharts dominates). Acceptable
  for an internal tool; the natural lever is route-level code splitting.
- **No earnings or news-sentiment features.** Both are scaffolded via the
  extension registry but not implemented — they each need a real data
  source decision (Finnhub / FMP / NewsAPI / Polygon).

## What runs when

| trigger | UTC | local hint | what happens |
|---|---|---|---|
| `us_ingest` | weekdays 22:30 | 17:30 ET (post US close) | IB → SP500 daily_update |
| `india_ingest` | weekdays 10:30 | 16:00 IST (post NSE close) | Kite → NIFTY100 daily_update |
| `daily_predict` (×2) | weekdays 11:00, 23:00 | post each close | predict + log + settle |
| `settlement_catchup` | weekdays 23:30 | 18:30 ET | safety net for unsettled rows |
| `monthly_retrain` | day 1–3 weekdays 07:00 | first trading day ~02:00 ET | tune + train + compare + promote |
| `universe_refresh` | Jan/Apr/Jul/Oct 5th 07:00 | quarterly reconstitution catch-up | `refresh_all_universes` |

The scheduler logs every job start, end, duration, and any exception
with full traceback. It is the noisiest log producer in the system on
purpose.
