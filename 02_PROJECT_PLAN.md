# Project Plan: ta-agent
## Phased build plan for Claude Code

This plan breaks the build into **9 phases**. Each phase is sized to be a
single Claude Code session (1-3 hours of focused work, depending on debug
cycles). Each phase has clear inputs, outputs, and acceptance criteria.

---

## Current state (Phase 0 already done)

You already have:
- Project scaffolding with `pyproject.toml` (uv-managed)
- `packages/common/` (config, logging, schemas)
- `packages/ingestion/storage.py` (DuckDB layer with idempotent upserts)
- `packages/ingestion/universe/sp500_history.py` (Wikipedia scraper)
- `packages/ingestion/universe/nifty100_history.py` (Phase A loader)
- `packages/ingestion/universe/membership.py` (unified query)
- `scripts/refresh_universes.py` (CLI to populate membership)
- 6 passing unit tests
- README with Windows + uv setup instructions

This is the foundation. Everything below builds on it.

---

## Phase dependency graph

```
Phase 0 (done) ──> Phase 1: IB adapter
                ├─> Phase 2: Kite adapter      ──┐
                └─> Phase 3: yfinance adapter  ──┤
                                                 ├─> Phase 4: Feature engineering
                                                 │
                                                 └─> Phase 5: Labels
                                                      │
                                                      ▼
                                           Phase 6: CV + training + calibration
                                                      │
                                                      ▼
                                           Phase 7: Inference + SHAP
                                                      │
                                                      ▼
                                           Phase 8: FastAPI backend
                                                      │
                                                      ▼
                                           Phase 9: React frontend
                                                      │
                                                      ▼
                                           Phase 10: Scheduling + ops polish
```

Phases 1, 2, 3 can be done in any order or in parallel. Everything from
Phase 4 onward is sequential.

---

## Phase summary

| # | Phase | Output | Est. session length |
|---|---|---|---|
| 1 | IB adapter | `packages/ingestion/adapters/ib_adapter.py` + tests | 2-3 hr |
| 2 | Kite adapter | `packages/ingestion/adapters/kite_adapter.py` + tests | 2 hr |
| 3 | yfinance adapter + corporate actions | `yfinance_adapter.py`, `corporate_actions.py` + ingest job | 2 hr |
| 4 | Feature engineering | `packages/features/` ~50 indicators + tests | 3 hr |
| 5 | Labels & dataset assembly | `packages/labels/` + master training-set builder | 1.5 hr |
| 6 | Modeling: CV, training, calibration, evaluation | `packages/modeling/` end-to-end | 3-4 hr |
| 7 | Inference & SHAP | `packages/inference/` + daily predict job | 2 hr |
| 8 | FastAPI backend | `services/api/` with all endpoints | 2-3 hr |
| 9 | React frontend | `services/frontend/` Dashboard + StockDetail + Performance | 3-4 hr |
| 10 | Scheduling & ops | `jobs/` schedulers, `Makefile`, monitoring | 2 hr |

**Total estimated effort: 22-30 hours of Claude Code sessions.**

---

## Critical conventions across all phases

These rules apply universally — every prompt enforces them, but listing
here as a reference:

### Code conventions
1. **Type hints on all public functions.** Use `from __future__ import annotations`.
2. **Loguru for logging.** Import via `from packages.common.logging import log`.
3. **Pydantic schemas for cross-module data.** Import from `packages.common.schemas`.
4. **Tests live in `tests/unit/` and `tests/integration/`** mirroring the package structure.
5. **Never use random k-fold on time series.** Always purged walk-forward.
6. **Never compute features using future data.** Every transform must be causal.
7. **Never bypass point-in-time membership.** Use `members_on(universe, date)`.

### Testing
- Every new module ships with unit tests.
- Tests must run in < 30 seconds total (use small fixtures).
- Integration tests that hit live APIs are marked with `@pytest.mark.integration`
  and skipped by default.

### Storage
- DuckDB is the single source of truth for OHLCV and membership.
- Features and labels are computed on demand from the OHLCV table.
- Trained models are stored as `data/models/{universe}_{target}_{timestamp}/`
  containing `model.txt`, `metadata.json`, and `calibrator.pkl`.

### Data quality guarantees
- Idempotent upserts everywhere. Re-running a job is always safe.
- Adjusted prices are stored in `close`; unadjusted in `close_unadj` for display only.
- All timestamps in UTC at the storage layer; localized only at presentation layer.

---

## How to use the phase prompts

For each phase:

1. Open Claude Code in your project directory.
2. Start a fresh session.
3. Paste the contents of `phase_NN_<name>.txt` as the first message.
4. Claude Code will plan, implement, run tests, and report back.
5. Review the diff. Run `pytest -v` yourself to confirm.
6. Commit the changes before moving to the next phase.

If a phase fails partway through (tests don't pass, or scope creeps), you can
either continue debugging in the same session or start a new session with the
same prompt — the prompts are written to be re-runnable on a partially-completed
state.

---

## Risk callouts per phase

- **Phase 1 (IB):** ib_insync requires TWS/Gateway running. Test mode requires
  paper trading account. Live data subscription needed for some symbols.
- **Phase 2 (Kite):** Access tokens expire daily. The prompt handles this.
- **Phase 4 (Features):** This is where look-ahead bugs hide. The prompt
  includes explicit causality tests.
- **Phase 5 (Labels):** The cross-sectional ranking step must use ONLY
  stocks that were members on date T. Easy to get wrong.
- **Phase 6 (Modeling):** Purged CV is non-trivial. The prompt provides the
  exact algorithm.
- **Phase 9 (Frontend):** This is the largest phase. Consider splitting into
  9a (Dashboard) and 9b (StockDetail + Performance) if it's too much.

---

## Definition of "done" for the whole project

The project is complete when:
- [ ] All 10 phases pass their acceptance criteria
- [ ] `pytest -v` runs all tests and they all pass
- [ ] You can run `python -m jobs.daily_ingest` followed by
      `python -m jobs.daily_predict` and see fresh predictions
- [ ] You can start the API (`uvicorn services.api.main:app`) and frontend
      (`npm run dev`) and see today's top picks rendered
- [ ] You have run a backtest on at least 2 years of held-out data and
      reviewed the IC, hit rate, and decile spread
- [ ] The README reflects the final state of the system

---

## What to do if something goes wrong

If a phase consistently produces broken code:
1. Check that the prior phases passed their tests
2. Check the file layout matches what the prompt expects
3. Try splitting the phase into smaller sub-prompts
4. As a last resort, ask a fresh Claude conversation to debug the specific
   error rather than continuing the Claude Code session

If model performance is wildly out of expected ranges (IC > 0.15 or hit
rate > 65%), STOP and look for leakage. Don't deploy a model with
"too good" metrics — they're almost always a bug.
