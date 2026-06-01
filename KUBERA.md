# Kubera — V1 (LOCKED 2026-06-01)

**Internal codename for the strict walk-forward ML trading pipeline.**

> Kubera — कुबेर — the Hindu lord of wealth and prosperity, guardian of the
> treasury of the gods. The 12-year-5-month strict walk-forward backtest
> compounded $1,000 → $47,689 (42.6×) at +33.4% annualized pre-tax, with
> Sharpe 1.43 and Sortino 2.31.

## Status: V1 LOCKED

As of 2026-06-01, the strategy specification, the simulator implementation,
and the live Alpaca execution engine are locked at V1. All three surfaces
share identical entry / exit / stop logic. The 12-year-5-month walk-forward
results below are the canonical performance reference. Re-validation cadence:
extend the backtest 6-12 months forward each year (~1 day of compute) to
confirm the structural alpha hasn't decayed.

## Strategy specification (V1)

### Entry
- **When**: every trading day, submitted at ~8:00 CT pre-open as `TIF=OPG`
  (market-on-open). Fills at the 8:30 CT opening auction print — same price
  the backtest models.
- **What**: top 5 picks from yesterday's close, ranked by **conviction-weighted
  score** = `predicted_return × (1 + (top_q_proba − bot_q_proba))`.
- **How much**: slice budget = NAV ÷ 5. Within the slice, each name's allocation
  ∝ `combined_score / ATR(14)` — high conviction + low volatility get more
  capital. Steady state: ~25 lots simultaneously (5 days × 5 picks/day).

### Exit (normal)
- **When**: 5 trading days after entry, submitted as `TIF=OPG`. Fills at
  day-6 opening auction.
- **No take-profit at any price.** The mean-reversion alpha lives in
  *continuing* the reversion past prior support, not in stopping at it.
  An empirical audit across 1,028 broken-support entries confirmed every
  variant of a take-profit ladder reduced cumulative P&L.

### Stop-loss
- **Formula**: `min(low over prior 3 daily bars) × 0.997` — 3-day rolling
  low minus 30 bps.
- **Type**: GTC SELL STOP attached after entry fills (~8:33 CT).
- **Refresh**: every evening after market close. Old stop cancelled,
  new GTC placed at the updated rolling-low.
- **Broken-support guard**: if the rolling-low sits at or above the lot's
  current price (stock gapped down through prior 3-day support), **no
  stop is placed**. The lot relies on the 5-day expiry exit only.
  Covers ~7% of entries; empirically the optimal handling.
- **Intraday-fired stops**: the engine reconciles via Alpaca's order-history
  scan; the originating signal is marked `CLOSED` so subsequent runs ignore
  the lot. Clean audit trail.

### Architectural invariants
- Long-only. No shorts.
- No leverage. `leverage_multiplier = 1.0`.
- No regime gate. Default-off per the 12-year validation showing it adds
  zero alpha across the full history.
- No FVG (Fair Value Gap) filter. Default-off — strips 91% of returns
  on the full history despite looking like a win on 2024-26 only.
- Costs: zero commission on Alpaca; SEC + FINRA TAF on sells (~1-2 bps).

## V1 reference performance (12y 5m strict WF)

Strict walk-forward, SP500 universe, Jan 2014 → May 2026, **149 monthly retrains**.
Numbers are post the 2026-06-01 phantom-exit correction — see §11 of
`docs/kubera-learnings.md` for what the correction was and why.

| Metric | Strategy | SPY B&H |
|---|---|---|
| **Cumulative gain** | **+4,157% (42.6×)** | +413% (5.1×) |
| **Annualized return** | **+33.4%** | +13.4% |
| **Sharpe (rf=0, daily)** | **1.43** | 0.85 |
| **Sortino (rf=0, daily)** | **2.31** | 1.20 |
| **Daily vol (ann)** | 23.8% | 17.2% |
| **Worst MaxDD** | 30.9% (2020 COVID) | 33.7% (2020 COVID) |
| **Years beating SPY** | **11 of 13** | — |

### Per-year excess return

| Year | Strategy | SPY | Excess | Notes |
|---|---|---|---|---|
| 2014 | +33.97% | +14.56% | +19.41% | |
| 2015 | −3.69% | +1.29% | −4.98% | China crash / oil collapse; mean-reversion struggled |
| 2016 | +70.26% | +13.59% | **+56.67%** | Best a/t excess year (in-sample) |
| 2017 | +31.00% | +20.78% | +10.22% | Low-vol Goldilocks; modest alpha |
| 2018 | +25.62% | −5.25% | +30.87% | Q4 vol spike — down-tape protection |
| 2019 | +45.30% | +31.09% | +14.22% | |
| 2020 | +17.39% | +17.24% | +0.15% | Matched SPY; COVID whipsaw cost the alpha |
| 2021 | +54.70% | +30.51% | +24.19% | |
| 2022 | +7.95% | −18.65% | +26.60% | Bear-market protection while still positive |
| 2023 | +41.31% | +26.71% | +14.60% | |
| 2024 | +51.32% | +25.59% | +25.73% | |
| **2025 (OOS)** | **+61.28%** | +18.01% | **+43.27%** | Program-record OOS year |
| 2026 partial (Jan-May) | +19.86% | +11.02% | +8.84% | May 2026 first sub-1% month |

## The codebase

The codename refers to the **entire pipeline**, not any single component:

- **Training**: [`scripts/walkforward_backtest.py`](scripts/walkforward_backtest.py) — strict
  walk-forward with per-retrain Optuna hyperparameter tuning (10 trials,
  n_jobs=8), TPE sampler reseeded deterministically per retrain, 5-day embargo,
  rolling 5-year training window.
- **Features**: [`packages/features/pipeline.py`](packages/features/pipeline.py) — daily SP500
  universe with survivorship correction, look-ahead-free factor engineering.
- **Paper-trading simulator**: [`packages/paper_trading/engine.py`](packages/paper_trading/engine.py) —
  the V1-locked strategy implementation. Broken-support guard at lot open.
- **Live Alpaca engine**: [`services/alpaca/engine.py`](services/alpaca/engine.py) — daemon
  that fires the strategy on real Alpaca paper / live accounts. Three triggers
  per trading day: pre-open OPG submission (~8 AM CT), post-open stop arming
  (~8:33 CT), post-close stop refresh. Matches the simulator's accounting
  exactly.
- **API**: [`services/api/services/predictions_service.py`](services/api/services/predictions_service.py) —
  serves the dashboard with per-month cell drill-downs, equity curve, year table.
- **Dashboard**: [`services/frontend/src/pages/LiveWF.tsx`](services/frontend/src/pages/LiveWF.tsx)
  (Live WF tab) and [`LiveAlpaca.tsx`](services/frontend/src/pages/LiveAlpaca.tsx)
  (Live Alpaca tab with engine start/stop button).
- **Scheduler**: [`jobs/scheduler.py`](jobs/scheduler.py) — APScheduler with 8:35 CT
  and 17:00 CT pipeline ticks that refresh OHLCV + predictions + paper-trade
  simulator.
- **Windows launcher kit**: [`scripts/windows/`](scripts/windows/) — one-click
  `.cmd` files to start / stop everything; `install_autostart.cmd` for
  auto-launch on Windows login.
- **Analysis**: `.claude/commands/wf-analysis.md` — `/wf-analysis` slash
  command that publishes the per-retrain markdown analysis.

## Reference WF deployment

```
--universe SP500
--start 2014-01-01
--end 2026-05-31
--per-retrain-optuna
--optuna-trials 10
--optuna-n-jobs 8
--device gpu
--out-dir data/processed/walkforward_10yr_strict
```

149 monthly retrains. Lineage: original 10-year-window 132 retrains
(Jan 2014 → Dec 2024) + 16-retrain extension (Jan 2025 → Apr 2026)
+ 1-retrain extension (May 2026).

## Methodology learnings

See **[`docs/kubera-learnings.md`](docs/kubera-learnings.md)** — comprehensive
149-month per-month log (outlook + actuals + per-month learnings) plus an
11-section cross-cutting synthesis:

1. Outlook accuracy across 52 live-forecast months
2. The four regimes Kubera plays in
3. The *setup vs catalyst* distinction (worked example: TTD Mar 5 2026)
4. Concentration discipline — when 25%+ allocations win vs fail
5. Tax-drag asymmetry
6. Year-boundary STCG-reset artifact
7. The Mag-7 structural underweight
8. Vol-regime sensitivity (sweet spot: VIX 35-65)
9. Out-of-sample validation final result
10. Concrete methodology recommendations for going forward
11. **The phantom-exit discovery — apples-to-apples corrected numbers (added 2026-06-01)**

## Forward expectations for live deployment

Realistic forward targets when running the V1-locked engine live:
- **Annualized return**: +25-35% in a typical year, +50-60% in a standout year.
- **Sharpe**: 1.2-1.5 in steady state (live degrades a few tenths from backtest).
- **Sortino**: 1.8-2.3.
- **Max drawdown**: plan for 30%+ in stress regimes.
- **Losing years are part of the distribution.** 2015 was −3.7% strategy
  return; May 2026 was −4.4% alpha. These are *expected* outcomes, not bugs.
- **The strategy underperforms in low-dispersion bull tapes (VIX <20,
  smooth SPY grind).** Best regime: VIX 35-65.

## Operational runbook

- **Start the stack**: double-click `scripts/windows/start_kubera.cmd`.
  Pops 3 minimized windows + opens dashboard.
- **Start Kubera live trading**: click **Start Kubera** on the Live Alpaca
  tab. Engine + sync run as detached processes; survive API/terminal/Claude
  restart.
- **Stop live trading**: click **Stop Kubera** on the same tab, or
  double-click `scripts/windows/stop_kubera.cmd`.
- **Switch paper ↔ live**: edit `ALPACA_MODE` in `.env` and restart the
  engine (stop + start).
- **Re-validate the strategy**: re-run the WF with a fresh `--end` 6-12
  months later. ~1 day compute. Auto-publishes to the Live WF tab.

---

*V1 locked 2026-06-01. Reference results captured at retrain 149/149
(May 29 2026 prediction window) under the post-phantom-exit-fix simulator.*
