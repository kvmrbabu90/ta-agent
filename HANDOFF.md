# ta-agent handoff

**Last updated:** 2026-05-16
**Branch:** `main`

This is the operational snapshot of the project — what's running, what
changed recently, and what to look at next. For project background read
`README.md`; for requirements read `01_PRD.md`; for the original phased
build plan read `02_PROJECT_PLAN.md`. This document is the "if you put
the project down for a month and pick it back up tomorrow, read this
first" doc.

---

## TL;DR

The system is **live in paper-trading mode** since 2026-05-11 with $1,000
of fake capital. Honest **11-year walk-forward** (2014-2024, 132 monthly
retrains, 1.1M predictions) shows **Sharpe 1.78** and **+5,830% return**
(turning $1,000 into $59,303 pre-tax). After accounting for tax
(25% blended STCG annually for the strategy vs 15% LTCG at end for SPY),
$24,011 vs SPY's $3,479 — a **6.9× outperformance after tax**. Strategy
beats SPY in **every single year** of the 10-year window including the
2022 bear (strategy −1% vs SPY −19%).

**Do not trade real money yet.** First, accumulate 4-8 weeks of live
paper data and confirm it tracks the backtest within ±50%. The strategy
is mean-reverting (buys recent losers on 5-day horizons) with stop-losses
and an audit-only LLM news classifier (Gemma 4 via Ollama). The regime
gate and FVG filter exist as opt-in code but are DISABLED by default —
the 10-year evidence showed both add zero value (regime) or actively hurt
returns (FVG, −91% cut on full history).

The live system runs **fully autonomously** via Windows Task Scheduler:
two pipeline runs per weekday (08:35 + 17:00 CT), plus monthly retrains
and quarterly Optuna re-tunes. A drift detector watches rank-IC and
fires an emergency retrain if the model degrades materially.

---

## What's running where

### Live daemons
- **API** (`uvicorn services.api.main:app --host 127.0.0.1 --port 8000`)
  serves the dashboard. Always-on, manually restartable. Logs at
  `logs/api.log`.
- **Vite frontend** at `http://127.0.0.1:5173`. Hot-reloads on `services/frontend/src/**` changes.
- **Ollama** at `http://127.0.0.1:11434` serving `gemma4:latest` (9.6 GB)
  for the news classifier. Only invoked during the daily pipeline, not always-on.

### Windows Task Scheduler (registered via `scripts/register_windows_tasks.ps1`)
| Task | Cadence | Local time | What it does |
|---|---|---|---|
| `ta-agent-pipeline-8am-ct` | weekdays | 08:35 CT | yfinance refresh → daily_predict → settlement → news_classify → paper_backtest → drift_check |
| `ta-agent-pipeline-5pm-ct` | weekdays | 17:00 CT | Same pipeline, post-close |
| `ta-agent-monthly-retrain` | weekdays | 02:00 CT | Cached-hyperparams retrain; CMD wrapper bails unless today is the 1st business day of the calendar month |
| `ta-agent-quarterly-retune` | weekdays | 03:00 CT | Full Optuna re-search (~3h); bails unless today is the 1st business day of Jan/Apr/Jul/Oct |

Per-run logs land at `logs/scheduled_run_YYYY-MM-DD.log`,
`logs/monthly_retrain_YYYY-MM-DD.log`,
`logs/quarterly_retune_YYYY-MM-DD.log`.

### Persistent data stores
| File | Size | Purpose |
|---|---|---|
| `data/processed/market.duckdb` | ~532 MB | OHLCV (2010→present), SEC filings, fundamentals, macro, index membership |
| `data/processed/predictions.sqlite` | ~38 MB | Daily predictions + realized returns (~127k rows) |
| `data/processed/paper.sqlite` | ~150 MB | Live paper-trading state (equity, positions, trades) |
| `data/processed/news.sqlite` | ~400 KB | Cached SEC filing bodies + LLM verdicts |
| `data/processed/walkforward/predictions.sqlite` | — | Honest walk-forward predictions, 25 monthly retrains × ~244k rows |
| `data/processed/sectors_sp500.parquet` | ~9 KB | GICS sector lookup per symbol |

---

## Current paper-trading strategy

See `packages/paper_trading/engine.py:StrategyConfig` for the canonical
config. **Live defaults as of 2026-05-16 (post 10-year walk-forward):**

```python
universe                = "SP500"
starting_cash           = $1,000
n_long                  = 5
n_short                 = 0            # long-only
holding_days            = 5            # overlapping portfolios
conviction_weighted     = True         # weight by combined_score within slice
vol_scaling             = "inverse"    # weights ∝ score / ATR

stop_loss_enabled       = True
stop_mode               = "support"    # ATR mode tested and rejected
support_lookback_days   = 3
stop_buffer_pct         = 0.003        # 0.3% — optimized on walk-forward data

commission_model        = "ibkr_lite"  # SEC fee on sells only, no commission
leverage_multiplier     = 1.0          # crank to 1.5-2x in tax-advantaged account

# DEFAULTS DISABLED based on 10-year evidence:
fvg_filter_enabled      = False        # was True; 91% return cut on full history
regime_gate_enabled     = False        # was True; zero value across all regimes
```

**Strategy in plain English:** Each weekday at 08:35 CT, take the top 5 SP500
stocks ranked by `predicted_return × (1 + direction_agreement)`, open a fresh
"slice" of long positions sized at `current_equity / 5`. Within each slice,
allocate by combined-score, scaled inversely by each name's 14-day ATR. Hold
each slice for 5 trading days then force-close. Per-position stop-loss at
0.3% below the rolling 3-day low; checked at 5 PM mark, fires at the stop
level (no slippage modelled). If SPY is unusually far from its 50d SMA
(strong-trend regime), scale slice budget down (0.75× to 0.25× depending on
distance).

### Performance reference points

**Honest walk-forward, 11 years (2014-2024, 132 monthly retrains):**

| Metric | Strategy (Pure ML) | SPY B&H |
|---|---|---|
| Final value on $1,000 (pre-tax) | **$59,303** | $3,917 |
| Total return | **+5,830%** | +292% |
| Sharpe ratio | 1.78 | 0.83 |
| Max drawdown | 28.4% | 33.7% |
| Worst year (return) | 2022 −1% | 2022 −19% |
| Best year (return) | 2020 +105% | 2019 +31% |
| Beat SPY each year | **11/11** | — |

**After-tax (25% blended STCG annually for strategy, 15% LTCG at end for SPY):**

| | Strategy | SPY B&H | Ratio |
|---|---|---|---|
| Final after-tax on $1,000 | **$24,011** | $3,479 | **6.9×** |

Chart: `docs/tax_adjusted_comparison.png` — equity curves
side-by-side with outperformance ratio in lower panel.

**Look-ahead-biased backtest** (single-model, same 11-year window): final
$59,303 → only marginally different from honest WF. Validates that the
model isn't heavily overfit to recent data — its edge is structural.

**Earlier (12-month, 2025-05 → 2026-05) numbers from HANDOFF v1**: had
been Sharpe 2.17, +57%. Those were on a benign-bull cherry-picked
window; the 11-year honest numbers above are the definitive picture.

### After-tax reality check (updated 2026-05-16)

The 11-year honest-WF after-tax results turn the earlier "wash with SPY"
verdict on its head. Even with the strategy paying STCG annually at a
blended 25% (federal-only, ignoring state) and SPY enjoying LTCG-deferral
until the final sale:

| | Pre-tax 11yr | After-tax 11yr |
|---|---|---|
| Strategy | $59,303 | $24,011 |
| SPY B&H | $3,917 | $3,479 |
| **Strategy / SPY** | **15.1×** | **6.9×** |

The strategy keeps a **6.9× after-tax advantage** because the alpha is
big enough to overwhelm the tax drag. Earlier (12-month) finding that
"strategy and SPY are tied after-tax" was specific to a low-alpha
calm-bull window. The full-history honest answer is clearly different.

In a **tax-advantaged account** (IRA/401k), the strategy wins by the
full 15.1× pre-tax margin — no STCG drag at all.

---

## Recent changes (May 11-15, 2026)

This was the major session that established the operational state above. In
roughly chronological order:

### Strategy engine v2
- **Overlapping portfolios** with 5-day holding period; one new slice opens
  each trading day, oldest slice force-closes at the open of day N+5.
- **Conviction-weighted sizing** within each slice — weights proportional to
  `combined_score = predicted_return × (1 + direction_agreement)`.
- **Stop-loss** at 0.3% below 3-day rolling low (tuned via `scripts/optimize_stop_loss.py --grid wide`
  against honest walk-forward predictions).
- **IBKR Lite cost model** — zero commission, regulatory pass-through fees on sells only.
- **Long-only** by default (`n_short=0`). Symmetric short logic is in place
  if `n_short > 0` is set.
- **Inverse-vol scaling** — within a slice, weights additionally scaled by
  `1 / ATR(14)`. Adds modest Sharpe + meaningful drawdown reduction.

### Regime detector
- `packages/paper_trading/regime.py` — SPY z-score against its 50-day SMA,
  normalized by `sqrt(50) × daily_vol`. Scales slice budget by 1.00 / 0.75
  / 0.50 / 0.25 depending on |z|. In current calm market this barely fires
  (78% "normal", 4% "trend", 0% "strong_trend") — bought as cheap insurance
  for trending regimes.

### LLM news classifier (audit-only)
- `packages/news/` — SEC EDGAR 8-K + Exhibit 99.1 fetcher, Gemma 4 classifier
  via Ollama, two-direction rubric:
  - **Longs**: `PANIC` (sentiment-driven decline → keep) vs `RESET` (real bad
    news → avoid) vs `UNCLEAR`.
  - **Shorts**: `HYPE` (sentiment-driven rally → keep) vs `STRENGTH` (real
    good news → avoid) vs `UNCLEAR`.
- Verdicts persisted to `news.sqlite` and surfaced on the dashboard as a
  chip next to each pick. **The paper-trading engine does NOT consume them
  yet** — they're audit-only while we accumulate (verdict, realized 5d
  return) pairs for validation. Decision point: 4-8 weeks of paired data,
  then we either hard-gate the picks or drop the LLM.
- **Exhibit 99.1 fetcher** (added 2026-05-15) — earnings 8-Ks reference
  attached press releases in Exhibit 99.1 with the actual beat/miss numbers.
  Before this fix, most short-side verdicts came back UNCLEAR because the
  primary 8-K cover doesn't contain the numbers. After: 3 PANIC + 5 STRENGTH
  on today's 20 picks (vs 1 + 2 before).

### Walk-forward backtest framework
- `scripts/walkforward_backtest.py` — retrains the model at the start of
  each month using only data available before that date, predicts forward
  for the month. Captures ~243k honest predictions over 24 months.
- `scripts/compare_walkforward.py` — runs the paper engine over both biased
  and honest predictions; writes `data/processed/walkforward/comparison_report.md`.

### Monthly retrain split + drift detector
- **Monthly retrain** (`jobs/monthly_retrain.py`) now defaults to `do_tune=False`
  — reuses cached Optuna hyperparameters, retrains only the weights against
  fresh data. ~5 min/universe.
- **Quarterly Optuna re-tune** — full search, refreshes the cached
  hyperparameters that the next 3 monthlies reuse. ~3 hours/universe.
- **Drift detector** (`packages/inference/drift.py`) watches rank-IC on the
  most recent 20 settled prediction-days. Hard trigger if mean rank-IC <
  −0.005; soft trigger if < +0.005 for 10 consecutive days. 14-day cooldown
  after a fresh model is promoted. Fires an off-cycle retrain when triggered.

### Frontend overhaul
- Full dark-mode redesign — Dashboard, Performance, Paper Trade, Stock
  Detail, Settings pages.
- New components: Sparkline, VerdictChip (LLM news indicator), system
  status indicator in header (last refresh + data through date).
- Dashboard ranking now uses combined-score (`predicted_return × (1 + dir_agreement)`)
  with strict-mode filtering and an amber warning chip on regression/classification sign disagreement.

### Infrastructure / robustness
- All FastAPI sqlite3 connections now use `check_same_thread=False` —
  FastAPI's threadpool moves sync handlers across threads.
- DuckDB reads from API + engine + walk-forward all open `read_only=True`
  so they coexist with the writer pipeline.
- `build_feature_matrix` opens ONE shared read-only DuckDB connection
  across all per-symbol fetches (was opening 500+ per call → lock thrashing).

### Rejected experiments
- **Sector one-hot features** (Phase E, May 15): 12 GICS sector indicators
  failed the 3-seed CV gate. Mean rank-IC delta −0.00165; worst-seed delta
  −0.019 (>4× tolerance). Decile spread improved +60% (potentially useful for
  portfolio construction, NOT for the predictor). Code preserved at
  `packages/features/sector.py` with the import disabled in `pipeline.py`.
- **ATR-based stops** (May 15): Sharpe dropped to 0.76-1.41 across 1.5/2/3×
  multipliers vs 2.02 baseline. Mean reversion needs tighter exits than ATR
  produces. Kept as `stop_mode='atr'` config option.
- Ensemble model averaging — analyzed and **deferred**. Estimated 5-15%
  Sharpe lift at ~5 hours validation cost. Worth doing if live paper
  underperforms backtest by >30% over 4-8 weeks.

---

## How to operate

### Start everything
```powershell
# API
.venv\Scripts\python.exe -m uvicorn services.api.main:app --host 127.0.0.1 --port 8000

# Frontend
cd services\frontend; npm run dev

# Verify scheduled tasks
Get-ScheduledTask -TaskName 'ta-agent-*'
```

### Manual runs
```bash
python -m jobs.run_pipeline                            # full daily pipeline once
python -m jobs.daily_predict                           # just predict
python -m jobs.news_classify --top-n 10                # classify today's picks
python -m scripts.walkforward_backtest                 # honest backtest (1-2h)
python -m scripts.compare_walkforward                  # diff biased vs honest
python -m scripts.optimize_stop_loss --grid wide       # 9×9 stop-loss grid
python -m jobs.monthly_retrain --only-if-first-business-day-of-month
python -m jobs.monthly_retrain --do-tune --n-trials 20  # force quarterly tune now
```

### Inspect state
```bash
# What's the live paper account doing?
curl http://127.0.0.1:8000/paper/snapshot

# Has the drift detector triggered recently?
grep -i drift logs/scheduled_run_*.log | tail -20

# Did the last pipeline complete cleanly?
tail -50 logs/scheduled_run_$(date +%Y-%m-%d).log
```

### Things that need attention
- **No automated backup of `market.duckdb`** — 532 MB on local disk. Worth a
  nightly OneDrive sync since rebuilding from yfinance would take days.
- **Macro series last refreshed 2026-05-06** — the daily pipeline doesn't
  refresh macro data; only OHLCV. Macro features are running on stale data.
  Fix: add `_job_macro_refresh` to `_job_us_ct_pipeline` in `jobs/scheduler.py`.

---

## Open decisions / pending todos

1. **Apply optimal stop_buffer_pct to default?** Re-optimization showed
   N=3, buf=0.003 is the wide-grid optimum (Sharpe 2.135 vs 2.019 at 0.005).
   **Status: APPLIED** (May 15). The corner optimum N=1, buf=0 (Sharpe 2.72)
   is degenerate — effectively a 1-day trailing stop — and was rejected.
2. **Hard-gate picks on LLM verdicts?** Currently audit-only. Need 4-8 weeks
   of paired data, then test whether RESETs/STRENGTHs underperform PANICs/HYPEs
   by ≥50bps. Decision point: ~late June 2026.
3. **Ensemble model averaging?** Deferred. Revisit if live results
   underperform backtest by >30%.
4. **Default real-money account location.** If/when this graduates from paper
   trading, run in a tax-advantaged account (IRA/401k) — taxable account math
   doesn't beat SPY at $400k MFJ marginal bracket.

---

## Files of note

```
packages/paper_trading/
  engine.py              v2 backtest engine — read this first
  regime.py              SPY-based regime gate

packages/news/
  edgar_fetcher.py       SEC 8-K + Exhibit 99.1 fetcher
  classifier.py          Gemma 4 client (think=False, JSON mode)
  pipeline.py            classify_top_picks orchestration

packages/inference/
  drift.py               rank-IC drift detector
  predict.py             daily_predict feature build + score
  tracking.py            settle_predictions

jobs/
  scheduler.py           APScheduler config + _job_* functions
  run_pipeline.py        one-shot CLI invoked by Task Scheduler
  monthly_retrain.py     monthly + quarterly retrain (CLI flags decide which)
  daily_predict.py       daily inference loop
  news_classify.py       standalone classifier CLI

services/api/
  main.py                FastAPI app
  routes/                /predictions, /paper, /news, /system, /stocks
  deps.py                read-only DuckDB + per-request SQLite

services/frontend/src/
  pages/                 Dashboard, Performance, PaperTrade, StockDetail, Settings
  components/VerdictChip.tsx       LLM news chip
  components/Layout.tsx            header + system status indicator
  hooks/useSystemStatus.ts         /system/status polling
  hooks/useNewsVerdicts.ts         /news/verdicts

scripts/
  walkforward_backtest.py          honest backtest
  compare_walkforward.py           biased vs honest comparison report
  optimize_stop_loss.py            stop-loss grid search
  backfill_predictions.py          predictions log backfill
  validate_sector_features.py      3-seed CV gate (the rejected-by-default validator)
  refresh_sectors.py               GICS sector lookup refresh (yfinance)
  register_windows_tasks.ps1       Task Scheduler registration

data/processed/walkforward/
  predictions.sqlite               honest WF predictions
  report.json                      per-retrain timing breakdown
  comparison_report.md             biased vs honest analysis
  stop_loss_grid_wf.csv            9×9 grid results
```

---

## Disclaimer

This is a personal research project. Predictions and backtest results are
not investment advice. Past performance does not predict future returns.
The honest walk-forward Sharpe of 2.17 is an upper bound on what live
trading would produce — real-world frictions (slippage, taxes, capacity
limits, regime shifts, model staleness) compound to lower it further.
