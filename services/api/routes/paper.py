"""Paper-trading API: equity curve, positions, recent trades."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
import duckdb

from services.api.deps import get_duckdb_conn

from packages.common.config import settings
from packages.paper_trading import StrategyConfig, backtest, init_paper_db
from services.api.schemas import (
    NextDayPick,
    NextDayPicksResponse,
    PaperBenchmarkPoint,
    PaperEquityPoint,
    PaperPosition,
    PaperPostTaxPoint,
    PaperRunSummary,
    PaperSnapshotResponse,
    PaperTrade,
    PaperTradesResponse,
)

# 30% blanket short-term capital-gains rate applied to the strategy's
# realized gain per calendar year. SPY B&H benchmark — no LTCG applied
# on the curve (the benchmark line is pre-tax; an LTCG haircut would
# only apply at sale, not on a continuous chart).
_PAPER_STRATEGY_STCG = 0.30
_PAPER_BENCHMARK_SYMBOL = "SPY"

router = APIRouter(prefix="/paper", tags=["paper_trading"])

_PAPER_DB = str(Path(settings.predictions_sqlite_path).parent / "paper.sqlite")


def _conn() -> sqlite3.Connection:
    init_paper_db(_PAPER_DB)
    # check_same_thread=False: FastAPI's threadpool may execute different
    # parts of a single request on different threads. Each request creates
    # a fresh connection that lives within one request — no cross-request
    # sharing — so disabling the same-thread guard is safe here.
    return sqlite3.connect(_PAPER_DB, check_same_thread=False)


@router.get("/snapshot", response_model=PaperSnapshotResponse)
def snapshot(
    run_id: str = Query("default"),
    lookback_days: int = Query(60, ge=1, le=2000),
    duck: duckdb.DuckDBPyConnection = Depends(get_duckdb_conn),
) -> PaperSnapshotResponse:
    conn = _conn()
    try:
        run_row = conn.execute(
            "SELECT run_id, universe, starting_cash, position_size, n_long, n_short, "
            "short_threshold, started_at, first_trade_date, last_trade_date, "
            "final_equity, final_realized_pnl, notes, "
            "holding_days, commission_model, stop_loss_enabled, "
            "support_lookback_days, stop_buffer_pct "
            "FROM paper_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if run_row is None:
            raise HTTPException(404, f"run_id={run_id} not found; backtest first")

        run = PaperRunSummary(
            run_id=run_row[0],
            universe=run_row[1],
            starting_cash=run_row[2],
            position_size=run_row[3],
            n_long=run_row[4],
            n_short=run_row[5],
            short_threshold=run_row[6],
            started_at=run_row[7],
            first_trade_date=_to_date(run_row[8]),
            last_trade_date=_to_date(run_row[9]),
            final_equity=run_row[10],
            final_realized_pnl=run_row[11],
            notes=run_row[12],
            holding_days=run_row[13],
            commission_model=run_row[14],
            stop_loss_enabled=bool(run_row[15]) if run_row[15] is not None else None,
            support_lookback_days=run_row[16],
            stop_buffer_pct=run_row[17],
        )

        # Equity curve over lookback window
        cutoff = (date.today().toordinal() - lookback_days)
        equity_rows = conn.execute(
            "SELECT trade_date, snapshot_kind, equity, cash, long_mv, short_mv, "
            "realized_pnl, unrealized_pnl FROM paper_equity "
            "WHERE run_id = ? ORDER BY trade_date, snapshot_kind",
            (run_id,),
        ).fetchall()
        equity_curve = [
            PaperEquityPoint(
                trade_date=date.fromisoformat(r[0]),
                snapshot_kind=r[1],
                equity=r[2],
                cash=r[3],
                long_mv=r[4],
                short_mv=r[5],
                realized_pnl=r[6],
                unrealized_pnl=r[7],
            )
            for r in equity_rows
            if date.fromisoformat(r[0]).toordinal() >= cutoff
        ]

        # --- Benchmark + post-tax overlays ----------------------------
        # Both align to the close_5pm_ct points of equity_curve (one
        # row per trading day at close). Open-snapshot rows are skipped
        # — the chart only needs the daily resolution.
        close_points = [p for p in equity_curve if p.snapshot_kind == "close_5pm_ct"]

        # SPY benchmark, rebased to starting capital of the paper run.
        # Fetch SPY closes over the equity_curve span from DuckDB.
        benchmark_curve: list[PaperBenchmarkPoint] = []
        if close_points:
            span_start = close_points[0].trade_date
            span_end = close_points[-1].trade_date
            bench_rows = duck.execute(
                "SELECT bar_date, close FROM ohlcv_daily "
                "WHERE symbol = ? AND bar_date >= ? AND bar_date <= ? "
                "ORDER BY bar_date",
                [_PAPER_BENCHMARK_SYMBOL, span_start, span_end],
            ).fetchall()
            if bench_rows:
                bclose: dict[str, float] = {str(d): float(c) for d, c in bench_rows}
                # First available bench close on/after span_start is the
                # normalization anchor.
                first_close = float(bench_rows[0][1])
                # Rebase to starting_cash of the paper run.
                scale = (run.starting_cash / first_close) if first_close > 0 else 0.0
                last_close = first_close
                for p in close_points:
                    d_str = p.trade_date.isoformat()
                    if d_str in bclose:
                        last_close = bclose[d_str]
                    benchmark_curve.append(
                        PaperBenchmarkPoint(
                            trade_date=p.trade_date,
                            equity=round(last_close * scale, 4),
                        )
                    )

        # Post-tax strategy curve (30% STCG, reduced-base compounding
        # year by year). IBKR Lite fees are already in the pre-tax
        # equity (paper engine deducts them).
        post_tax_curve: list[PaperPostTaxPoint] = []
        if close_points:
            # Per-year start/end equity from the pre-tax close series.
            soy_eq: dict[int, float] = {}
            eoy_eq: dict[int, float] = {}
            for p in close_points:
                y = p.trade_date.year
                if y not in soy_eq:
                    soy_eq[y] = p.equity
                eoy_eq[y] = p.equity
            max_d = close_points[-1].trade_date
            # Per-year multiplicative factor used to scale equity AT
            # START of each year. Anchor to the FIRST actual close so
            # the post-tax line starts at the same point as the strategy
            # line (matches the chart visually — divergence appears only
            # when a calendar year completes and the tax bite kicks in).
            sorted_years = sorted(soy_eq.keys())
            post_eq_start_of_year: dict[int, float] = {}
            starting_post = soy_eq[sorted_years[0]]
            cumul_multiple = 1.0
            for y in sorted_years:
                post_eq_start_of_year[y] = starting_post * cumul_multiple
                if max_d >= date(y, 12, 28):
                    # Year fully elapsed → apply STCG to the gain.
                    r = (eoy_eq[y] / soy_eq[y] - 1) if soy_eq[y] > 0 else 0
                    factor = (
                        1 + r * (1 - _PAPER_STRATEGY_STCG) if r > 0 else 1 + r
                    )
                    cumul_multiple *= factor
            # Walk close_points, scale intra-year by pre-tax growth.
            for p in close_points:
                y = p.trade_date.year
                intra = (
                    (p.equity / soy_eq[y]) if soy_eq[y] > 0 else 1.0
                )
                post_tax_curve.append(
                    PaperPostTaxPoint(
                        trade_date=p.trade_date,
                        equity=round(post_eq_start_of_year[y] * intra, 4),
                    )
                )

        # Latest positions (most recent trade_date). With overlapping-portfolio
        # construction, the same symbol may exist across multiple lots — aggregate
        # to a single per-symbol row for the UI: sum qty, qty-weighted avg entry,
        # earliest entry_date (= the slice's longest-held leg).
        last_date_row = conn.execute(
            "SELECT MAX(trade_date) FROM paper_positions WHERE run_id = ?", (run_id,)
        ).fetchone()
        positions: list[PaperPosition] = []
        last_close_price_by_sym: dict[str, float | None] = {}
        if last_date_row and last_date_row[0]:
            last_date = date.fromisoformat(last_date_row[0])
            position_rows = conn.execute(
                "SELECT symbol, side, qty, entry_price, entry_date, stop_level "
                "FROM paper_positions WHERE run_id = ? AND trade_date = ?",
                (run_id, last_date.isoformat()),
            ).fetchall()
            # Aggregate lots per (symbol, side). entry_price = qty-weighted avg.
            # earliest_entry = oldest lot's entry date.
            # latest_entry = newest lot's entry date (used to compute the
            # planned exit, which is the FURTHEST-OUT forced-close — the
            # symbol stays held until the last lot ages out).
            # stop_level_max = max stop across lots (the tightest active
            # stop for a long, since any lot hitting it closes that lot).
            agg: dict[tuple[str, str], dict[str, Any]] = {}
            for sym, side, qty, entry, entry_date, stop_level in position_rows:
                key = (sym, side)
                if key not in agg:
                    agg[key] = {
                        "symbol": sym, "side": side, "qty": 0.0,
                        "cost_basis": 0.0,
                        "earliest_entry": entry_date,
                        "latest_entry": entry_date,
                        "stop_level_max": stop_level,
                        "lots": 0,
                    }
                bucket = agg[key]
                bucket["lots"] += 1
                bucket["qty"] += float(qty)
                bucket["cost_basis"] += float(qty) * float(entry)
                if entry_date < bucket["earliest_entry"]:
                    bucket["earliest_entry"] = entry_date
                if entry_date > bucket["latest_entry"]:
                    bucket["latest_entry"] = entry_date
                # Tightest stop = highest stop_level for longs (closer to
                # last price). None values are ignored; the max() of all
                # non-None values is taken.
                if stop_level is not None:
                    if bucket["stop_level_max"] is None or stop_level > bucket["stop_level_max"]:
                        bucket["stop_level_max"] = stop_level
            symbols = [sym for sym, _side in agg.keys()]
            last_close_price_by_sym = _last_close_prices(symbols, last_date)
            # Planned exit = entry_date + holding_days TRADING days (not
            # calendar days). Use NYSE calendar for SP500. Computed once
            # then offset per symbol.
            holding_days = int(getattr(run, "holding_days", None) or 5)
            # Build a trading-day calendar covering the next ~3 weeks of
            # business days starting from the latest entry seen.
            try:
                import pandas_market_calendars as mcal
                cal_name = "NYSE"
                cal = mcal.get_calendar(cal_name)
                # Look ahead 60 calendar days from the latest entry date,
                # which always covers 5-10 trading days.
                latest_entry_str = max(b["latest_entry"] for b in agg.values())
                latest_entry_dt = date.fromisoformat(latest_entry_str)
                from datetime import timedelta as _td
                sched = cal.schedule(
                    start_date=latest_entry_dt.isoformat(),
                    end_date=(latest_entry_dt + _td(days=60)).isoformat(),
                )
                trading_days = [d.date() for d in sched.index]
            except Exception:  # noqa: BLE001 — fallback if calendar import fails
                trading_days = []

            def _planned_exit(entry_str: str) -> date | None:
                """Entry + holding_days trading days, using NYSE calendar."""
                entry_dt = date.fromisoformat(entry_str)
                if not trading_days:
                    # Fallback: +holding_days*1.4 calendar days (rough proxy).
                    from datetime import timedelta as _td
                    return entry_dt + _td(days=int(holding_days * 1.4))
                # Find entry's index in trading_days, then +holding_days.
                # If entry is on a trading day, exit = trading_days[i + holding_days].
                # If entry isn't (rare for paper-trade), find the next trading day.
                for i, td in enumerate(trading_days):
                    if td >= entry_dt:
                        target = i + holding_days
                        if target < len(trading_days):
                            return trading_days[target]
                        return None
                return None

            for (sym, side), bucket in agg.items():
                qty = bucket["qty"]
                avg_entry = bucket["cost_basis"] / qty if qty > 0 else 0.0
                last_px = last_close_price_by_sym.get(sym)
                if last_px is None:
                    unreal = 0.0
                    last_px_for_response = avg_entry
                elif side == "long":
                    unreal = qty * (last_px - avg_entry)
                    last_px_for_response = last_px
                else:
                    unreal = qty * (avg_entry - last_px)
                    last_px_for_response = last_px
                positions.append(
                    PaperPosition(
                        symbol=sym,
                        side=side,
                        qty=qty,
                        entry_price=avg_entry,
                        entry_date=date.fromisoformat(bucket["earliest_entry"]),
                        last_price=last_px_for_response,
                        unrealized_pnl=unreal,
                        # Planned exit = latest-lot's entry + holding_days
                        # (the position fully unwinds when the youngest
                        # lot expires).
                        planned_exit_date=_planned_exit(bucket["latest_entry"]),
                        stop_level=bucket["stop_level_max"],
                        lot_count=bucket["lots"],
                    )
                )

        return PaperSnapshotResponse(
            run=run,
            equity_curve=equity_curve,
            positions=positions,
            benchmark_curve=benchmark_curve,
            benchmark_symbol=_PAPER_BENCHMARK_SYMBOL if benchmark_curve else None,
            post_tax_curve=post_tax_curve,
            strategy_tax_rate=_PAPER_STRATEGY_STCG,
        )
    finally:
        conn.close()


@router.get("/trades", response_model=PaperTradesResponse)
def trades(
    run_id: str = Query("default"),
    limit: int = Query(50, ge=1, le=1000),
    closes_only: bool = Query(
        False,
        description="If true, only return closing trades (long_close, "
        "short_close, stop_close). Skips OPEN trades. Useful for a 'recent "
        "exits' view that emphasizes realized P&L.",
    ),
) -> PaperTradesResponse:
    conn = _conn()
    try:
        # Filter ON DB side for efficiency — applying after LIMIT would
        # under-count the result set.
        # Self-join close rows to their opening lot (same lot_id + symbol) to
        # surface the entry date and entry price alongside the exit. Open rows
        # match themselves on the join; we null their entry fields in Python.
        base_select = (
            "SELECT c.trade_date, c.symbol, c.side, c.qty, c.fill_price, "
            "c.cash_delta, c.realized_pnl, o.trade_date, o.fill_price "
            "FROM paper_trades c "
            "LEFT JOIN paper_trades o "
            "  ON o.run_id = c.run_id AND o.lot_id = c.lot_id AND o.symbol = c.symbol "
            "  AND o.side IN ('long_open', 'short_open') "
            "WHERE c.run_id = ? "
        )
        if closes_only:
            rows = conn.execute(
                base_select
                + "AND c.side IN ('long_close', 'short_close', 'stop_close') "
                "ORDER BY c.trade_date DESC, c.symbol LIMIT ?",
                (run_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                base_select + "ORDER BY c.trade_date DESC, c.symbol LIMIT ?",
                (run_id, limit),
            ).fetchall()
        close_sides = {"long_close", "short_close", "stop_close"}
        trades_out = [
            PaperTrade(
                trade_date=date.fromisoformat(r[0]),
                symbol=r[1],
                side=r[2],
                qty=r[3],
                fill_price=r[4],
                cash_delta=r[5],
                realized_pnl=r[6],
                entry_date=date.fromisoformat(r[7]) if (r[2] in close_sides and r[7]) else None,
                entry_price=r[8] if r[2] in close_sides else None,
            )
            for r in rows
        ]
        return PaperTradesResponse(run_id=run_id, trades=trades_out)
    finally:
        conn.close()


@router.post("/rebuild", response_model=PaperRunSummary)
def rebuild(
    run_id: str = Query("default"),
    universe: str = Query("SP500"),
    starting_cash: float = Query(1000.0, gt=0),
    n_long: int = Query(5, ge=1, le=50),
    n_short: int = Query(5, ge=0, le=50),
) -> PaperRunSummary:
    """Trigger a fresh backtest from logged predictions. Clears prior trades for run_id."""
    cfg = StrategyConfig(
        universe=universe, starting_cash=starting_cash,
        n_long=n_long, n_short=n_short,
        short_enabled=(n_short > 0),
        run_id=run_id,
        notes=f"manual rebuild via API",
    )
    backtest(cfg)
    return snapshot(run_id=run_id, lookback_days=2000).run


def _to_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _last_close_prices(symbols: list[str], on_or_before: date) -> dict[str, float | None]:
    """Fetch the most-recent close <= on_or_before from market.duckdb for each symbol."""
    if not symbols:
        return {}
    import duckdb

    duck = duckdb.connect(settings.duckdb_path, read_only=True)
    try:
        rows = duck.execute(
            """
            WITH ranked AS (
                SELECT symbol, bar_date, close,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY bar_date DESC) AS rn
                FROM ohlcv_daily
                WHERE symbol = ANY(?) AND bar_date <= ?
            )
            SELECT symbol, close FROM ranked WHERE rn = 1
            """,
            [symbols, on_or_before],
        ).fetchall()
    finally:
        duck.close()
    return {r[0]: float(r[1]) if r[1] is not None else None for r in rows}


# ---------------------------------------------------------------------------
# /paper/next-day-picks — top 5 picks the engine will trade tomorrow at open
# ---------------------------------------------------------------------------

# Strategy constants — must match packages/paper_trading/engine.py and
# services/alpaca/engine.py exactly. V1-locked values; do not vary.
_TOP_N_PICKS = 5
_HOLDING_DAYS = 5
_ATR_LOOKBACK = 14
_SUPPORT_LOOKBACK = 3
_STOP_BUFFER_PCT = 0.003


def _atr_wilder(highs: list[float], lows: list[float], closes: list[float],
                  lookback: int = _ATR_LOOKBACK) -> float | None:
    """Wilder's ATR over `lookback` bars. Returns most recent value."""
    if len(highs) < lookback + 1:
        return None
    trs: list[float] = []
    prev_close = closes[0]
    for h, l, c in zip(highs[1:], lows[1:], closes[1:]):
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
        prev_close = c
    if len(trs) < lookback:
        return None
    atr = sum(trs[:lookback]) / lookback
    for tr in trs[lookback:]:
        atr = (atr * (lookback - 1) + tr) / lookback
    return atr


@router.get("/next-day-picks", response_model=NextDayPicksResponse)
def next_day_picks(
    run_id: str = Query("default"),
    duck: duckdb.DuckDBPyConnection = Depends(get_duckdb_conn),
) -> NextDayPicksResponse:
    """Top 5 picks the strategy will trade at the next session's open print.

    Mirrors the engine logic exactly: pulls the latest as_of from
    predictions_log, ranks by combined_score = predicted_return ×
    (1 + (top_q − bot_q)), computes ATR(14) + rolling-low stop from
    DuckDB OHLCV, sizes per-stock as (NAV / 5) × normalized inverse-vol
    weight (combined_score / ATR(14)).

    Updates whenever the 17:00 CT daily_predict step writes a fresh
    as_of row — the dashboard re-queries on focus and picks update
    automatically.
    """
    from datetime import timedelta as _td

    # --- NAV (last close-mark equity for this paper run) ---
    conn = _conn()
    try:
        nav_row = conn.execute(
            "SELECT equity FROM paper_equity WHERE run_id = ? "
            "AND snapshot_kind = 'close_5pm_ct' "
            "ORDER BY trade_date DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        if nav_row is None:
            nav_row = conn.execute(
                "SELECT starting_cash FROM paper_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        nav = float(nav_row[0]) if nav_row else 0.0
        run_row = conn.execute(
            "SELECT universe FROM paper_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        universe = run_row[0] if run_row else "SP500"
    finally:
        conn.close()

    slice_budget = (nav / _HOLDING_DAYS) if nav > 0 else 0.0

    # --- Latest predictions ---
    pred_conn = sqlite3.connect(settings.predictions_sqlite_path,
                                  check_same_thread=False)
    try:
        latest_row = pred_conn.execute(
            "SELECT MAX(as_of) FROM predictions_log WHERE universe = ?",
            (universe,),
        ).fetchone()
        latest_as_of = latest_row[0] if latest_row else None
        if latest_as_of is None:
            return NextDayPicksResponse(
                as_of=date.today(),
                target_trade_date=None,
                nav=nav, slice_budget=slice_budget, universe=universe,
                picks=[],
            )
        top_rows = pred_conn.execute(
            "SELECT symbol, predicted_return, "
            "COALESCE(top_quintile_proba, 0.0) AS top_q, "
            "COALESCE(bottom_quintile_proba, 0.0) AS bot_q, "
            "predicted_return * (1.0 + (COALESCE(top_quintile_proba, 0.0) "
            "  - COALESCE(bottom_quintile_proba, 0.0))) AS combined_score "
            "FROM predictions_log "
            "WHERE universe = ? AND as_of = ? "
            "ORDER BY combined_score DESC, symbol ASC LIMIT ?",
            (universe, latest_as_of, _TOP_N_PICKS),
        ).fetchall()
        # Latest write timestamp for this as_of (UTC ISO). Lets the dashboard
        # distinguish morning-preliminary from evening-final.
        last_write_row = pred_conn.execute(
            "SELECT MAX(created_at) FROM predictions_log WHERE universe = ? AND as_of = ?",
            (universe, latest_as_of),
        ).fetchone()
        last_write = last_write_row[0] if last_write_row else None
    finally:
        pred_conn.close()

    as_of_d = date.fromisoformat(latest_as_of)

    # --- Next NYSE trading session strictly after as_of ---
    target_trade_date: date | None = None
    try:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar("NYSE")
        sched = cal.schedule(
            start_date=as_of_d.isoformat(),
            end_date=(as_of_d + _td(days=10)).isoformat(),
        )
        sessions = [d.date() for d in sched.index]
        target_trade_date = next((s for s in sessions if s > as_of_d), None)
    except Exception:
        target_trade_date = None

    if not top_rows:
        return NextDayPicksResponse(
            as_of=as_of_d, target_trade_date=target_trade_date,
            nav=nav, slice_budget=slice_budget, universe=universe, picks=[],
            predictions_written_at=last_write if 'last_write' in dir() else None,
        )

    # --- OHLCV for ATR + rolling-low ---
    symbols = [r[0] for r in top_rows]
    bars_rows = duck.execute(
        "WITH ranked AS ( "
        "  SELECT *, ROW_NUMBER() OVER ( "
        "    PARTITION BY symbol, bar_date ORDER BY ingested_at DESC "
        "  ) AS rn "
        "  FROM ohlcv_daily "
        "  WHERE symbol = ANY(?) AND bar_date BETWEEN ? AND ? "
        ") "
        "SELECT symbol, bar_date, high, low, close "
        "FROM ranked WHERE rn = 1 ORDER BY symbol, bar_date",
        [symbols, as_of_d - _td(days=60), as_of_d],
    ).fetchall()
    bars: dict[str, list[tuple]] = {s: [] for s in symbols}
    for sym, d, h, l, c in bars_rows:
        bars[sym].append((d, float(h), float(l), float(c)))
    for s in bars:
        bars[s].sort(key=lambda r: r[0])

    # --- Compute per-pick metrics + inverse-vol weights ---
    pick_meta: list[dict] = []
    for sym, pred, top_q, bot_q, score in top_rows:
        b = bars.get(sym, [])
        highs = [r[1] for r in b]
        lows = [r[2] for r in b]
        closes = [r[3] for r in b]
        atr = _atr_wilder(highs, lows, closes, _ATR_LOOKBACK)
        last_close = closes[-1] if closes else 0.0
        rolling_low_stop = None
        if len(lows) >= _SUPPORT_LOOKBACK:
            rolling_low_stop = round(
                min(lows[-_SUPPORT_LOOKBACK:]) * (1.0 - _STOP_BUFFER_PCT), 2
            )
        broken_support = (
            rolling_low_stop is not None and last_close > 0
            and rolling_low_stop >= last_close
        )
        raw_weight = (max(0.0, float(score)) / atr) if (atr and atr > 0) else 0.0
        pick_meta.append({
            "symbol": sym, "predicted_return": float(pred),
            "top_q": float(top_q) if top_q is not None else None,
            "bot_q": float(bot_q) if bot_q is not None else None,
            "combined_score": float(score), "last_close": float(last_close),
            "atr_14": float(atr) if atr else None,
            "rolling_low_stop": rolling_low_stop,
            "broken_support": broken_support, "raw_weight": raw_weight,
        })

    total_weight = sum(p["raw_weight"] for p in pick_meta)

    picks: list[NextDayPick] = []
    for rank, p in enumerate(pick_meta, start=1):
        if total_weight > 0 and p["last_close"] > 0 and p["raw_weight"] > 0:
            weight_pct = (p["raw_weight"] / total_weight) * 100.0
            alloc = slice_budget * (p["raw_weight"] / total_weight)
            qty = max(1, int(alloc / p["last_close"]))
        else:
            weight_pct = 0.0
            qty = 0
        notional = qty * p["last_close"]
        picks.append(NextDayPick(
            rank=rank, symbol=p["symbol"],
            predicted_return=p["predicted_return"],
            top_quintile_proba=p["top_q"], bottom_quintile_proba=p["bot_q"],
            combined_score=p["combined_score"], last_close=p["last_close"],
            atr_14=p["atr_14"], planned_qty=qty, planned_notional=notional,
            planned_weight_pct=weight_pct,
            rolling_low_stop=p["rolling_low_stop"],
            broken_support=p["broken_support"],
        ))

    return NextDayPicksResponse(
        as_of=as_of_d, target_trade_date=target_trade_date,
        nav=nav, slice_budget=slice_budget, universe=universe, picks=picks,
        predictions_written_at=last_write,
    )
