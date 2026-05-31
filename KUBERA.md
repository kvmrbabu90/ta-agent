# Kubera

**Internal codename for the strict walk-forward ML trading pipeline.**

> Kubera — कुबेर — the Hindu lord of wealth and prosperity, guardian of the
> treasury of the gods. Apt for a pipeline whose 12-year-4-month backtest
> compounded $1,000 into $193,430 pre-tax (193.43×) / $49,051 after-tax
> (49.05×) at 49.93% annualized pre-tax.

## What Kubera is

Kubera is the per-retrain Optuna-tuned LightGBM walk-forward backtest defined
in `scripts/walkforward_backtest.py`, with the live dashboard, paper trading
engine, and analysis tooling surrounding it. The codename refers to the
**entire pipeline**, not any single component:

- **Training**: `scripts/walkforward_backtest.py` — strict walk-forward with
  per-retrain Optuna hyperparameter tuning (10 trials, n_jobs=8), TPE
  sampler reseeded deterministically per retrain, 5-day embargo,
  rolling 5-year training window.
- **Features**: `packages/features/pipeline.py` — daily SP500 universe with
  survivorship correction, look-ahead-free factor engineering.
- **Paper trading**: `packages/paper_trading/engine.py` — top-5 long-only
  daily-rotated book, 5-day holding period, stop-losses, ibkr_lite cost
  model.
- **API**: `services/api/services/predictions_service.py` — serves the
  dashboard with per-month cell drill-downs, equity curve, year table.
- **Dashboard**: `services/frontend/src/pages/LiveWF.tsx` — Live WF tab with
  the equity curve, monthly excess heatmap, year table (with VIX peak +
  SPY MaxDD columns), histograms with Mean/Median/σ/Pareto stats, and
  the analysis panel.
- **Analysis**: `.claude/commands/wf-analysis.md` — `/wf-analysis` slash
  command that publishes the per-retrain markdown analysis.

## Configuration

The reference deployment (used to produce the 12-year-4-month results):

```
--universe SP500
--start 2014-01-01
--end 2026-04-30
--per-retrain-optuna
--optuna-trials 10
--optuna-n-jobs 8
--device gpu
--out-dir data/processed/walkforward_10yr_strict
```

148 monthly retrains. Original 10-year-window 132 retrains (Jan 2014 →
Dec 2024) plus a 16-retrain OOS extension (Jan 2025 → Apr 2026) added
during the May 2026 validation pass.

## Results summary (12-year-4-month backtest)

- **Pre-tax cumulative**: 193.43× ($1,000 → $193,430)
- **After-tax cumulative**: 49.05× ($1,000 → $49,051)
- **Annualized pre-tax**: 49.93%
- **Annualized after-tax**: 34.91%
- **Sharpe (program-wide)**: ~+2.26 (sustained above +2 for ~97 months)
- **Max drawdown ever**: 19.2% (Aug 2020 COVID recovery whipsaw)
- **Years with positive a/t excess**: 12/12 (100%)
- **OOS months with positive a/t excess**: 14/16 (Jan 2025 → Apr 2026)
- **Best year ever (a/t excess vs SPY)**: 2025 — +55.50 pts (TRUE
  out-of-sample data)
- **Best single month (a/t excess)**: Mar 2026 — +18.01% (also OOS)
- **Best single day (a/t excess)**: Mar 5 2026 — +11.86% (TTD squeeze)

## Methodology learnings

See **[`docs/kubera-learnings.md`](docs/kubera-learnings.md)** — comprehensive
148-month per-month log (outlook + actuals + per-month learnings) plus a
10-section cross-cutting synthesis that distills the methodology insights
for going-forward decisions:

1. Outlook accuracy across 52 live-forecast months
2. The four regimes Kubera plays in (down-tape protection / catalyst win /
   steady participation / up-tape miss + concentration drawdown)
3. The *setup vs catalyst* distinction (worked example: TTD Mar 5 2026)
4. Concentration discipline — when 25%+ allocations win vs fail
5. Tax-drag asymmetry — why bear years (2022, 2018) and giant-alpha years
   (2025) dominate the excess table
6. Year-boundary STCG-reset artifact
7. The Mag-7 structural underweight (the model's blind spot)
8. Vol-regime sensitivity (sweet spot: VIX 35-65)
9. Out-of-sample validation final result
10. Concrete methodology recommendations for going forward

## Status

**Locked.** The 12-year-4-month strict WF backtest is complete. The
methodology is validated in true out-of-sample data. Kubera is ready
for either: (a) periodic OOS re-validation (extend backtest 6-12
months ahead each year, ~1 day compute), or (b) live paper-trading
deployment with the existing 5-day-rotation engine.

---

*Codename adopted 2026-05-31. Reference results captured at retrain
148/148 (Apr 30 2026 prediction window).*
