# ta-agent handoff

**Last updated:** 2026-05-23
**Branch:** `main`

> **🚨 OPERATIONAL WARNING — git worktrees + `data/` junction.** If you
> create a git worktree that contains a directory junction back into the
> main tree (commonly `data/` so the worktree shares OHLCV with main),
> NEVER call `git worktree remove` directly. Git's recursive delete on
> Windows follows junctions and deletes the TARGET's contents — wiping
> out the main tree's `data/`. Use `make worktree-remove WT=<path>`
> which removes junctions first (without following them) before calling
> git worktree remove. See `scripts/safe_worktree_remove.ps1`.

This is the operational snapshot of the project — what's running, what
changed recently, and what to look at next. For project background read
`README.md`; for requirements read `01_PRD.md`; for the original phased
build plan read `02_PROJECT_PLAN.md`. This document is the "if you put
the project down for a month and pick it back up tomorrow, read this
first" doc.

---

## TL;DR

**Strict walk-forward in flight** (started 2026-05-21 14:44 CDT, currently
26.5% complete at 35/132 retrains, ETA Wed May 28 ~15:00 CDT). This is
the **clean, lookahead-free version** — per-retrain Optuna tuning,
expanding-window retraining, no shared-hyperparams shortcut. Built to
prove the strategy's edge survives a more honest backtest than the
v1 walk-forward.

**Through 35 months (2014-01 through 2016-12, full year 2016 locked
in)**:
- Strategy **+160.45%** pre-tax, **+104.07%** after-tax (locks in 2014
  + 2015 + 2016 taxes; 2017+ still pre-tax until those years complete)
- SPY **+31.80%** pre-tax / **+27.03%** after 15% LTCG
- **Pre-tax excess +128.65%; after-tax excess +77.04%**
- Strategy multiple **2.60×** ($1,000 → $2,604)
- Sharpe (running) **2.39**; MaxDD **12.5%** (Aug 2015 crash)

**Years locked in so far**:

| Year | Strategy | After-tax | SPY | Excess | Sharpe | MaxDD |
|---|---|---|---|---|---|---|
| 2014 | +38.54% | +26.98% | +14.56% | +23.98% | 2.40 | 6.9% |
| 2015 | +5.35% | +3.75% | +1.29% | +4.06% | 0.40 | 12.5% |
| 2016 | **+78.44%** | **+54.91%** | +13.59% | **+64.85%** | 2.95 | 4.0% |

**2016 was the freakish year.** Q1 alone delivered +27% excess driven
by V-bottom mean-reversion regime. Full year +64.85% excess is
above any reasonable steady-state estimate. This is partly real edge,
partly favorable regime fit (2016 had dispersion-heavy environment
with Brexit, energy reversal, election rotation), partly variance from
specific catalyst days (Jan 29 BOJ surprise was a 6% single-day move).
**2017 is the structural test** — low-vol melt-up year that historically
breaks short-horizon strategies.

**India universe (NIFTY100) was nuked end-to-end on 2026-05-22** — user
decided not to test on India. All data, models, code paths, API config,
frontend rendering removed. Bhavcopy adapter and kite adapter both
deleted. **Repo is now SP500-only.**

**Do not trade real money yet.** The strict-WF run needs to finish
(~5 more days) and the post-2016 years (2017-2024) need to validate
the edge in different regimes before any live capital decision.

---

## Active runs

### Strict walk-forward (the headline run)
- **Started**: 2026-05-21 14:44:41 CDT
- **Process**: PID 258, still alive
- **Command**:
  ```
  python -m scripts.walkforward_backtest \
    --universe SP500 --start 2014-01-01 --end 2024-12-31 \
    --per-retrain-optuna --optuna-trials 10 --optuna-n-jobs 1 \
    --device gpu \
    --out-dir data/processed/walkforward_10yr_strict
  ```
- **Progress**: 35 of 132 retrains done (last completed: Dec 2016)
- **Pace**: rolling avg 75.9 min/retrain (computed from last 5 retrains;
  used to be hardcoded at 75 min before commit `32707ee` 2026-05-23)
- **ETA**: Wed May 28 ~15:00 CDT
- **Output dir**: `data/processed/walkforward_10yr_strict/`
  - `predictions.sqlite` — predictions for each retrain's forward window
  - `analysis_live.sqlite` — rebuilt by API on each strict-WF endpoint
    request (mtime-keyed cache)
  - `report.json` — per-retrain timing breakdown

### Live API (always-on)
- **Process**: uvicorn at `127.0.0.1:8000` (PID 6128, no `--reload` flag
  because reload was missing predictions_service.py edits)
- **Restart command**: `cd C:/dev/ta-agent && .venv/Scripts/python.exe
  -m uvicorn services.api.main:app --host 0.0.0.0 --port 8000`
- **Key endpoints**:
  - `/performance/strict-wf/SP500` — Live WF dashboard data
  - `/performance/strict-wf/SP500/month/{year}/{month}` — drill-down for
    a single heatmap cell (added 2026-05-23 commit `d6d000b`)
  - `/performance/SP500`, `/predictions/...` — existing routes

### Vite frontend (always-on)
- **Process**: `npm run dev` in `services/frontend/`, serves at
  `http://127.0.0.1:5173`
- Hot-reloads on TS/TSX changes

### Daily pipeline (paused implicitly by the strict-WF GPU lock)
The Windows Task Scheduler entries (`ta-agent-pipeline-8am-ct`,
`ta-agent-pipeline-5pm-ct`, etc.) are still registered and will fire on
schedule, but the v1 daily pipeline isn't materially used while the
strict-WF run dominates the GPU. **Don't disable them** — they keep
predictions.sqlite + paper.sqlite warm. They DO sometimes briefly
collide with the strict-WF's DuckDB access (see "Known issues" below).

---

## Live WF dashboard (built 2026-05-21 through 23)

The big visible addition since the last handoff. New tab at
`/live-wf` shows the in-flight strict walk-forward in real time.
Auto-refreshes every 60s.

**Components** (top to bottom):
- **Progress strip**: `35 / 132 retrains · latest: 2016-12-30 (5 min ago) ·
  ETA in 5d · avg 76 min/retrain`. Live countdown to next retrain ticks
  every second under the bar.
- **Summary tiles** (4 across): Strategy cum, SPY cum, Strategy
  annualized, Strategy/Bench ratio. Each tile has a small after-tax /
  after-LTCG sub-line.
- **Equity curve chart** (recharts line): three lines — strategy
  pre-tax (solid emerald), strategy post-tax (dashed emerald, only
  visible after first year completes), SPY B&H (sky blue indexed to
  the strategy's starting capital). Single dot at the last date shows
  SPY's post-LTCG liquidation value.
- **Monthly excess heatmap**: 12 months × N years, color-scaled by
  strategy − SPY excess. Toggle pill switches between "Excess vs SPY"
  and "Strategy return" views. **Click any cell** to open the
  drill-down modal.
- **Year table**: Year / Strategy / After Tax / SPY / Excess / Sharpe /
  MaxDD / Days. After-tax column populates only when the calendar year
  fully elapses in the WF data.

**Click-to-expand cell modal** (added commit `d6d000b`):
- 4 headline tiles for the month
- Daily equity path chart (both lines rebased to 100 at month start)
- Best 3 / worst 3 days by excess
- Top 10 holdings ranked by days held, with avg-weight-when-held
  (fixed in commit `dff90a2` to sum lots per (symbol, date) first;
  previously under-counted by 2-5×)

**Tax treatment in the UI**:
- Strategy STCG **30%** (US short-term, applied to positive years only,
  losing years pass through unchanged, no carryforward)
- Benchmark LTCG **15%** (US federal LTCG, mid-bracket, **Texas
  resident** — no state add-on)
- Per-year tax bite kicks in on Jan 1 of the following year (modeled
  as a step-down in the post-tax equity line)

---

## Persistent data stores

| File | Size | Purpose |
|---|---|---|
| `data/processed/market.duckdb` | ~310 MB | OHLCV (post-NSE removal), SP500 membership, sector lookup, SEC filings, fundamentals, macro |
| `data/processed/walkforward_10yr_strict/predictions.sqlite` | ~57 MB | Strict-WF predictions, ~290k rows so far |
| `data/processed/walkforward_10yr_strict/analysis_live.sqlite` | ~6 MB | API-rebuilt paper backtest on the strict-WF predictions |
| `data/processed/predictions.sqlite` | ~38 MB | Legacy daily predictions (still updated by scheduled tasks) |
| `data/processed/paper.sqlite` | ~150 MB | Live paper-trading state (still updated by scheduled tasks) |
| `data/processed/news.sqlite` | ~400 KB | LLM news classifier output |

**Note on size**: `market.duckdb` dropped from ~532 MB → ~310 MB after
deleting all NSE OHLCV rows (386,127 bars) plus NIFTY100 membership.

---

## Recent changes (2026-05-21 through 23) — 19 commits

In chronological order, grouped by theme:

### Live WF dashboard buildout (the main thread)
1. `a01ec3a` — initial Live WF tab with progress + per-year table
2. `89271db` — graceful Optuna fallback when no trials complete
3. `3a4b595` — fix dtype coercion in `_align_features` (NIFTY100 era;
   stayed relevant for SP500 too)
4. `1e088e6` — NIFTYBEES backfill + correct retrain count (later moot)
5. `6b2f1f0` — after-tax strategy column (year table)
6. `10f96cd` — after-tax cumulative return + annualized (summary tile)
7. `8d71f84` — equity curve chart + benchmark LTCG sub-line
8. `593f651` — post-LTCG liquidation dot on the chart
9. `176b897` — INR formatting (later moot)
10. `afb0bb7` — bump Running-badge heuristic 2h → 5h
11. `e11a50d` — monthly excess heatmap
12. `eb789eb` — fix `monthly_wf_report.py` to match UI conventions
13. `d2855fc` — heatmap toggle (Excess vs SPY ↔ Strategy return)
14. `d6d000b` — click-to-expand monthly drill-down card + new endpoint
15. `3ab3d9d` — live countdown to next retrain
16. `32707ee` — compute real rolling-avg pace from retrain timestamps
17. `dff90a2` — fix avg holding weight aggregation (sum lots per day
    first, then average across days)

### India universe removal
- `82a75e3` — retry-on-lock in `predict_window` (intended for NIFTY100
  May 2016 gap; not actually applied to the running WF process)
- `e3a1773` — **nuke India / NIFTY100 universe end-to-end**:
  - Deleted ~12 code files: `nse_bhavcopy.py`, `kite_adapter.py`,
    `nifty100_history.py`, `nifty100_pit.py`, `run_india_pipeline.py`,
    `kite_backfill.py`, all India tests, etc.
  - Deleted ~48 NIFTY100 training parquets
  - Deleted `wf_nifty100_strict/` output dir
  - Deleted 386,127 NSE OHLCV rows from DuckDB
  - Deleted 100 NIFTY100 membership rows
  - Trimmed `predictions_service.py` of all NIFTY100 dicts
  - Removed NIFTY100 section from `LiveWF.tsx`
  - Trimmed `scheduler.py` of India job entries

### Bug fixes
- `89271db` — graceful Optuna fallback (no-trials case)
- `3a4b595` — coerce object-dtype features to numeric (NaN on failure)
- `82a75e3` — retry-on-lock in `_predict_window` (PATCHED CODE EXISTS,
  but the running WF process loaded the old code before this commit
  → still vulnerable to lock collisions)
- `eb789eb` — three reconciliation fixes between CLI report and UI

---

## Known issues / open items

### 1. DuckDB lock collisions during predict_window (HIGH IMPACT)
The strict-WF process has hit the lock collision bug 3 times:
- **Feb 2015** (retrain 14): predict_window fully failed → 0
  predictions for the month. Equity carried Jan positions through Feb.
- **May 2016 (NIFTY100)** — moot after India nuke
- **Nov 2016** (retrain 35): predict_window failed on Nov 9, 10, 11, 14,
  15, 16, 30. **Election-week predictions are MISSING.** Strategy
  carried Nov 8 positions through the post-election rotation. This is
  the second-worst data hole.

**Cause**: a SYSTEM Python 3.11 install (`C:\Python311\python.exe`,
NOT our `.venv\Scripts\python.exe`) periodically opens market.duckdb,
briefly holding the lock. Likely a Windows Task Scheduler job or a
background Jupyter kernel — never observed at the moment of collision
because it comes and goes.

**Fix exists but isn't applied**: commit `82a75e3` adds retry-on-lock
to `_predict_window`. The currently-running WF process loaded its code
before that commit landed, so it's not protected. **Restarting the WF
would apply the fix but cost ~36 hours of progress** (35 retrains
re-done). User explicitly opted to accept the gaps rather than restart.

**Going forward**: each remaining month has ~5% probability of another
collision. Worst case: 5 of remaining 97 retrains have gaps. Better
case (most likely): 1-2 more gaps before run completes.

### 2. Monitor for retrain completions is flaky
The `tail -F -n0` monitor process keeps dying (cause unclear — possibly
loguru's nightly log rotation, possibly grep pipe collapse). Self-respawn
loop respawns it within 3 seconds, but ANY retrain that completes during
the dead window gets missed because `-n0` starts tail at end-of-file.

**Current monitor**: task `be2nsq9tg` with `while true; do tail -F ...
sleep 3; done` outer loop.

**Mitigation**: I proactively check WF state at any user interaction
even between monitor events. Several monthly commentaries have been
delivered after-the-fact this way.

### 3. Hardcoded benchmark uses full calendar year
The Live WF year table's bench column uses **full calendar year SPY**
(Jan-Dec) regardless of how much of the year the strategy has actually
traded. So an in-progress year (e.g. partial 2016 mid-run) showed SPY's
full-year +13.59% even when the strategy only had 6 months done. This
is **intentional matching of the UI's existing convention** but
worth flagging if you read the dashboard mid-year. After the year
fully completes in the WF, the comparison is apples-to-apples.

### 4. The 75-min hardcoded avg pace (FIXED 2026-05-23)
Before commit `32707ee` the `avg_retrain_minutes` was literally
`avg_min = 75.0` — a placeholder from when there was no data. Now
computed from the last 5 retrains' MAX(created_at) timestamps.
Currently shows 75.9 min (coincidentally close to the old default).

### 5. Backups
Same as previous handoff — no automated backup of `market.duckdb`.
With NSE gone the file is smaller (~310 MB) but the SP500 history
behind it would still take hours to rebuild from yfinance.

---

## Performance comparison: strict-WF vs prior v1 honest WF

This is the headline story of the May 21+ work. The v1 honest WF
(documented in the previous handoff) showed:
- Strategy +5,830% over 11 years
- Sharpe 1.78
- 6.9× after-tax outperformance vs SPY

The v1 WF had **shared hyperparameters across all retrains** (one
Optuna tune at the start, reused for all 132 monthly retrains). This
is a subtle form of look-ahead — the hyperparams were tuned with
visibility into future data structure.

The strict WF re-tunes Optuna per retrain (5 CV folds × 10 trials per
retrain × 132 retrains). This is genuinely lookahead-free. **Through
35 months it's tracking well within range of the v1 numbers** —
+160% pre-tax cum at this checkpoint is comparable to the v1's
trajectory at the same point. The full comparison will be possible
once the strict-WF run finishes ~May 28.

---

## How to operate

### Start everything
```powershell
# API (no --reload; edits require kill+restart)
.venv\Scripts\python.exe -m uvicorn services.api.main:app --host 0.0.0.0 --port 8000

# Frontend
cd services\frontend; npm run dev

# Verify scheduled tasks
Get-ScheduledTask -TaskName 'ta-agent-*'
```

### Check strict-WF state
```bash
# Most recent retrain completed:
grep -E "\[[0-9]+/132\] [0-9]{4}-[0-9]{2}-[0-9]{2}: train_end" \
  "$WF_LOG" | tail -3

# Current API view:
curl -s http://localhost:8000/performance/strict-wf/SP500 | python -m json.tool

# Monthly breakdown that matches the UI exactly:
python -m scripts.monthly_wf_report
```

### Manual runs (mostly unchanged from v1)
```bash
python -m jobs.run_pipeline                            # full daily pipeline once
python -m scripts.walkforward_backtest --help          # strict WF flags
python -m scripts.monthly_wf_report                    # match-the-UI per-month table
```

---

## Open decisions / pending todos

1. **Identify and kill the recurring Python 3.11 culprit**. A system
   Python install is briefly holding the DuckDB lock 1-2× per day and
   blowing up predict_window. Restarting Windows or finding the
   scheduled job that owns `C:\Python311\python.exe` would prevent
   future gaps. Not yet done.
2. **Restart strict-WF with the retry-on-lock fix?** Currently opted out
   to preserve progress. Decision point: if another big-deal month
   (e.g. 2020 COVID crash) gets gapped, reconsider.
3. **Hard-gate picks on LLM verdicts?** Same as previous handoff — need
   4-8 weeks of paired data. Pending; LLM classifier is dormant during
   the strict-WF run.
4. **Ensemble model averaging?** Same as previous handoff — deferred.
5. **Strategy-vs-SPY commentary cadence**: agreed (2026-05-22) to
   deliver per-month commentary on every retrain completion. Format:
   headline numbers + macro context + sector moves + strategy
   positioning narrative + forward outlook. Sustaining this requires
   the monitor to fire reliably; see "Monitor for retrain completions
   is flaky" above.

---

## Files of note (delta from previous handoff)

### Added since 2026-05-17
```
scripts/
  monthly_wf_report.py        per-month CLI breakdown that matches the UI
  walkforward_backtest.py     extended with --cv-min-train-days,
                              graceful Optuna fallback

services/api/
  schemas.py                  added StrictWfMonthDetail, StrictWfDailyPoint,
                              StrictWfHolding, StrictWfMonthlyExcessCell,
                              StrictWfEquityCurve
  routes/performance.py       added /performance/strict-wf/{universe}/month/{year}/{month}
  services/predictions_service.py
                              get_strict_wf_status (live dashboard data)
                              get_strict_wf_month_detail (cell drill-down)
                              + retry-on-lock helper (unused yet — see issues #1)

services/frontend/src/
  api/types.ts                added StrictWf* types
  api/performance.ts          added fetchStrictWfMonth
  hooks/usePerformance.ts     added useStrictWf, useStrictWfMonth
  pages/LiveWF.tsx            the entire Live WF page
                              (UniverseSection, ProgressBar,
                              NextRetrainCountdown, EquityCurveChart,
                              MonthlyExcessHeatmap, YearTable,
                              MonthDetailModal, MonthDetailBody,
                              DetailTile, DayList)
```

### Removed since 2026-05-17 (India nuke)
```
scripts/
  backfill_niftybees.py
  analyze_nifty100_walkforward.py
  audit_nifty_pit.py
  kite_backfill.py
  run_india_pipeline.cmd

jobs/
  run_india_pipeline.py

packages/ingestion/adapters/
  nse_bhavcopy.py
  kite_adapter.py

packages/ingestion/universe/
  nifty100_history.py
  nifty100_pit.py

tests/unit/
  test_kite_adapter.py
  test_nifty100_pit.py

configs/universes/
  nifty100_changes.yaml
```

---

## Backlog (post-strict-WF ship)

Mostly unchanged from previous handoff, plus a few additions:

- **HMM regime detector** — still backlog
- **Paid survivorship-bias closure** — still backlog
- **India regime gate** — moot (India nuked)
- **Per-month commentary export** — save the rich monthly writeups as
  a versioned YAML so they can be re-rendered later (e.g. on the cell
  card). Currently they live only in chat history.
- **Identify recurring DuckDB locker** — investigate which Windows
  scheduled task or Jupyter kernel uses `C:\Python311\python.exe` and
  touches market.duckdb. Likely a one-line fix once identified.

## Disclaimer

This is a personal research project. Predictions and backtest results
are not investment advice. Past performance does not predict future
returns. The strict-WF Sharpe of 2.39 (running) is the cleanest
estimate the system has produced so far, but real-world frictions
(slippage, taxes, capacity limits, regime shifts, model staleness)
compound to lower it further. **2017 (low-vol melt-up) is the
structural test still ahead in the run.**
