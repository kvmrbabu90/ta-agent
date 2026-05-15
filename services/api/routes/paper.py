"""Paper-trading API: equity curve, positions, recent trades."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from packages.common.config import settings
from packages.paper_trading import StrategyConfig, backtest, init_paper_db
from services.api.schemas import (
    PaperEquityPoint,
    PaperPosition,
    PaperRunSummary,
    PaperSnapshotResponse,
    PaperTrade,
    PaperTradesResponse,
)

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
                "SELECT symbol, side, qty, entry_price, entry_date "
                "FROM paper_positions WHERE run_id = ? AND trade_date = ?",
                (run_id, last_date.isoformat()),
            ).fetchall()
            # Aggregate lots per (symbol, side). entry_price = qty-weighted avg.
            agg: dict[tuple[str, str], dict[str, Any]] = {}
            for sym, side, qty, entry, entry_date in position_rows:
                key = (sym, side)
                if key not in agg:
                    agg[key] = {
                        "symbol": sym, "side": side, "qty": 0.0,
                        "cost_basis": 0.0, "earliest_entry": entry_date,
                    }
                bucket = agg[key]
                bucket["qty"] += float(qty)
                bucket["cost_basis"] += float(qty) * float(entry)
                if entry_date < bucket["earliest_entry"]:
                    bucket["earliest_entry"] = entry_date
            symbols = [sym for sym, _side in agg.keys()]
            last_close_price_by_sym = _last_close_prices(symbols, last_date)
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
                    )
                )

        return PaperSnapshotResponse(
            run=run, equity_curve=equity_curve, positions=positions
        )
    finally:
        conn.close()


@router.get("/trades", response_model=PaperTradesResponse)
def trades(
    run_id: str = Query("default"),
    limit: int = Query(50, ge=1, le=1000),
) -> PaperTradesResponse:
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT trade_date, symbol, side, qty, fill_price, cash_delta, realized_pnl "
            "FROM paper_trades WHERE run_id = ? ORDER BY trade_date DESC, symbol LIMIT ?",
            (run_id, limit),
        ).fetchall()
        trades_out = [
            PaperTrade(
                trade_date=date.fromisoformat(r[0]),
                symbol=r[1],
                side=r[2],
                qty=r[3],
                fill_price=r[4],
                cash_delta=r[5],
                realized_pnl=r[6],
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
