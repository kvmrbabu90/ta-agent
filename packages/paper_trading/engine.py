"""Paper-trading backtest engine — overlapping-portfolios long-only.

Strategy (see `StrategyConfig` for tunables):
  - Universe: SP500
  - Each trading day at 8:35 CT (5 min after open, ensures yfinance has
    today's open bar):
      * Look at the model's predictions made off yesterday's close
      * Rank by combined score: predicted_return × (1 + direction_agreement)
        where direction_agreement = top_quintile_proba - bottom_quintile_proba
      * Take the top `n_long` names
      * **Overlapping portfolios** (Jegadeesh-Titman style): open a NEW
        slice today; close the slice that's been held `holding_days`
        trading days. In steady state you hold up to n_long × holding_days
        positions across overlapping slices. Each lot is sized at
        `current_equity / holding_days` total, allocated within the slice
        proportionally to combined score.
      * Open at today's OPEN price (~8:30 CT)
  - At 5 PM CT mark-to-market with today's CLOSE.
  - **Stop-loss**: each open lot has stop_level = support_level × (1 - stop_buffer_pct),
    where support_level = rolling min(low) over the prior `support_lookback_days`
    bars. At the 5 PM mark, if today's close ≤ stop_level, the lot exits at
    stop_level (slippage and gaps explicitly ignored — per spec).
  - **Costs**: IBKR Lite — $0 commission for US listed stocks. Pass-through
    fees on sells only: SEC fee ($0.0000278 × notional) + FINRA TAF
    ($0.000166/share, max $8.30). Opens are free.

This is BACKTEST-ONLY. The shape of `step_one_day` is preserved enough that
a future live loop can be bolted on, but right now everything flows from
predictions_log → ohlcv_daily → paper.sqlite.

Schema (v2):
    paper_runs       — one row per backtest run (config + start/end + final_equity)
    paper_trades     — every fill: keyed by (run_id, trade_date, lot_id, symbol, side)
                       sides: 'long_open', 'long_close', 'stop_close'
    paper_positions  — open positions snapshot per (run_id, trade_date, lot_id, symbol)
    paper_equity     — equity curve, snapshot_kind in ('open_8am_ct', 'close_5pm_ct')
    paper_meta       — key/value (schema_version)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from packages.common.config import settings
from packages.common.logging import log
from packages.ingestion.storage import _connect as duck_connect

# Bump this and `init_paper_db` drops + recreates the tables. Cheap because
# the paper DB is fully derivable from predictions_log + ohlcv_daily — there
# is no historical state worth preserving.
_PAPER_SCHEMA_VERSION = 2


_PAPER_DDL = """
CREATE TABLE IF NOT EXISTS paper_runs (
    run_id          TEXT PRIMARY KEY,
    universe        TEXT NOT NULL,
    starting_cash   REAL NOT NULL,
    position_size   REAL NOT NULL,
    n_long          INTEGER NOT NULL,
    n_short         INTEGER NOT NULL,
    short_threshold REAL NOT NULL,
    started_at      TEXT NOT NULL,
    first_trade_date TEXT,
    last_trade_date  TEXT,
    final_equity    REAL,
    final_realized_pnl REAL,
    notes           TEXT,
    -- v2 additions:
    holding_days        INTEGER,
    commission_model    TEXT,
    stop_loss_enabled   INTEGER,
    support_lookback_days INTEGER,
    stop_buffer_pct     REAL
);

CREATE TABLE IF NOT EXISTS paper_trades (
    run_id          TEXT NOT NULL,
    trade_date      TEXT NOT NULL,
    lot_id          INTEGER NOT NULL,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,   -- 'long_open' | 'long_close' | 'stop_close'
    qty             REAL NOT NULL,
    fill_price      REAL NOT NULL,
    cash_delta      REAL NOT NULL,
    realized_pnl    REAL,
    cost            REAL DEFAULT 0,
    PRIMARY KEY (run_id, trade_date, lot_id, symbol, side)
);

CREATE TABLE IF NOT EXISTS paper_positions (
    run_id          TEXT NOT NULL,
    trade_date      TEXT NOT NULL,
    lot_id          INTEGER NOT NULL,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,   -- 'long' (long-only by default in v2)
    qty             REAL NOT NULL,
    entry_price     REAL NOT NULL,
    entry_date      TEXT NOT NULL,
    stop_level      REAL,
    PRIMARY KEY (run_id, trade_date, lot_id, symbol)
);

CREATE TABLE IF NOT EXISTS paper_equity (
    run_id          TEXT NOT NULL,
    trade_date      TEXT NOT NULL,
    snapshot_kind   TEXT NOT NULL,   -- 'open_8am_ct' | 'close_5pm_ct'
    equity          REAL NOT NULL,
    cash            REAL NOT NULL,
    long_mv         REAL NOT NULL,
    short_mv        REAL NOT NULL,
    realized_pnl    REAL NOT NULL,
    unrealized_pnl  REAL NOT NULL,
    PRIMARY KEY (run_id, trade_date, snapshot_kind)
);

CREATE TABLE IF NOT EXISTS paper_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS paper_equity_run_date
    ON paper_equity (run_id, trade_date);
CREATE INDEX IF NOT EXISTS paper_trades_run_date
    ON paper_trades (run_id, trade_date);
"""


def _paper_db_path() -> str:
    return str(Path(settings.predictions_sqlite_path).parent / "paper.sqlite")


def _current_schema_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute(
            "SELECT value FROM paper_meta WHERE key='schema_version'"
        ).fetchone()
        return int(row[0]) if row else 1
    except sqlite3.OperationalError:
        return 0


def init_paper_db(path: str | None = None) -> str:
    """Create the paper-trading schema. Drops + recreates if version stale."""
    p = path or _paper_db_path()
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    try:
        existing = _current_schema_version(conn)
        # existing == 0 means there is no paper_meta table — either a fresh DB
        # (no tables at all) OR a pre-versioning install with stale v1 tables.
        # In the latter case the v1 schema lacks our new columns, so the safest
        # move is to drop everything and rebuild.
        needs_rebuild = existing != _PAPER_SCHEMA_VERSION
        if needs_rebuild:
            if existing > 0:
                log.info(
                    f"paper.sqlite schema {existing} -> {_PAPER_SCHEMA_VERSION}; "
                    "dropping and recreating (no historical state preserved)"
                )
            for tbl in (
                "paper_meta", "paper_runs", "paper_trades",
                "paper_positions", "paper_equity",
            ):
                conn.execute(f"DROP TABLE IF EXISTS {tbl}")
        conn.executescript(_PAPER_DDL)
        conn.execute(
            "INSERT OR REPLACE INTO paper_meta (key, value) VALUES (?, ?)",
            ("schema_version", str(_PAPER_SCHEMA_VERSION)),
        )
        conn.commit()
    finally:
        conn.close()
    return p


@dataclass
class StrategyConfig:
    universe: str = "SP500"
    starting_cash: float = 1000.0
    n_long: int = 5
    n_short: int = 0  # long-only by default (v2)
    position_size: float = 100.0  # legacy field, unused with overlapping portfolios
    short_threshold: float = 0.001  # legacy
    short_enabled: bool = False

    # Overlapping-portfolios sizing. Each new slice lives exactly
    # `holding_days` trading days. Slice budget = current_equity / holding_days.
    holding_days: int = 5

    # Within a slice, allocate by combined_score = pred * (1 + dir_agreement)
    # (set conviction_weighted=False to fall back to equal weight)
    conviction_weighted: bool = True

    # Costs. 'ibkr_lite' is commission-free with regulatory pass-throughs
    # on sells only. 'none' = no costs at all (sanity-check baseline).
    commission_model: str = "ibkr_lite"

    # Stop-loss. Rolling N-day low support × (1 - buffer). Checked at 5 PM
    # close mark; exit price = stop_level (slippage/gaps ignored per spec).
    #
    # Defaults chosen by `scripts/optimize_stop_loss.py --grid wide` against
    # 12 months of backfilled predictions (2025-05 → 2026-05): N=3 / buf=0.5%
    # delivered Sharpe 3.77 / Sortino 3.31 / final +145%, a clear lift over
    # the prior N=20/buf=0.6% (Sharpe 3.21) without sitting at the degenerate
    # tightest-stop corner. Re-tune from honest forward data after enough
    # post-go-live history accumulates.
    stop_loss_enabled: bool = True
    # `stop_mode`:
    #   'support' — stop = min(low[t-N..t]) × (1 - stop_buffer_pct).
    #               Adaptive to recent structural support. Fixed buffer.
    #               This is the default — mean-reversion needs tight,
    #               decisive exits when the bounce signal itself fails.
    #   'atr'     — stop = entry_price - atr_multiplier × ATR(atr_lookback_days).
    #               REJECTED in the May 2026 validation: ATR stops are
    #               intrinsically wider on volatile names, which are
    #               exactly the names mean-reversion targets. Across
    #               1.5×/2×/3× multipliers, Sharpe fell to 0.76-1.41 on
    #               honest WF data (vs 2.02 baseline). Kept as an option
    #               for future regimes where ATR might shine — e.g. a
    #               longer-horizon strategy.
    stop_mode: str = "support"
    support_lookback_days: int = 3
    # Honest WF wide-grid optimum (scripts/optimize_stop_loss.py --grid wide
    # on WF DB, 2024-05 → 2026-05): N=3, buf=0.003 yields Sharpe 2.135 vs
    # the prior 2.019 at buf=0.005. Stayed at N=3 rather than corner N=1
    # to avoid the degenerate 1-day-trailing-stop optimum.
    stop_buffer_pct: float = 0.003
    atr_lookback_days: int = 14  # Wilder's standard (unused unless stop_mode='atr')
    atr_multiplier: float = 2.0

    # Position sizing within a slice.
    # `vol_scaling`:
    #   'none'    — weight ∝ combined_score.
    #   'inverse' — weight ∝ combined_score / ATR(atr_lookback_days).
    #               High-conviction + low-vol names get more capital;
    #               high-vol names take proportionally less risk.
    #               Validated May 2026 on honest WF data: +0.008 Sharpe,
    #               +$48 final equity, -1.3pp max drawdown vs 'none'.
    #               Drawdown improvement is the bigger deal.
    vol_scaling: str = "inverse"

    # Regime gate. Scales the slice budget down when SPY is far from its
    # 50-day SMA (mean-reversion fails in strong-trend regimes). See
    # packages/paper_trading/regime.py for thresholds and rationale.
    regime_gate_enabled: bool = True

    # Backtest bounds
    start_date: date | None = None
    end_date: date | None = None
    run_id: str = "default"
    notes: str = ""

    # Storage overrides (used by walkforward_backtest to compare an honest
    # walk-forward predictions DB against the look-ahead-biased default).
    # When set, the engine reads predictions from this path instead of
    # `settings.predictions_sqlite_path` and writes paper state to
    # `paper_db_path` instead of the default `data/processed/paper.sqlite`.
    predictions_sqlite_path: str | None = None
    paper_db_path: str | None = None


DEFAULT_CONFIG = StrategyConfig()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_predictions(
    universe: str, start: date | None, end: date | None,
    sqlite_path: str | None = None,
) -> pd.DataFrame:
    """Pull logged predictions + direction-agreement probas."""
    p = sqlite_path or settings.predictions_sqlite_path
    conn = sqlite3.connect(p)
    try:
        sql = (
            "SELECT as_of, symbol, predicted_return, top_quintile_proba, "
            "bottom_quintile_proba FROM predictions_log WHERE universe = ?"
        )
        params: list = [universe]
        if start is not None:
            sql += " AND as_of >= ?"
            params.append(start.isoformat())
        if end is not None:
            sql += " AND as_of <= ?"
            params.append(end.isoformat())
        sql += " ORDER BY as_of, predicted_return"
        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()
    if df.empty:
        return df
    df["as_of"] = pd.to_datetime(df["as_of"]).dt.date
    # direction_agreement in [-1, +1]; null-safe (treat missing as 0).
    df["dir_agree"] = (
        df["top_quintile_proba"].fillna(0.0) - df["bottom_quintile_proba"].fillna(0.0)
    )
    df["combined_score"] = df["predicted_return"] * (1.0 + df["dir_agree"])
    return df


def _load_ohlcv_for_symbols(
    duck: duckdb.DuckDBPyConnection,
    symbols: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    """Pull open/high/low/close for the universe in one shot.

    `low` is used for the rolling-support stop-loss calc;
    `high`+`low`+`close` together feed the True-Range calculation that
    ATR-mode stops depend on; `open`/`close` for execution and marking.
    """
    if not symbols:
        return pd.DataFrame(
            columns=["symbol", "bar_date", "open", "high", "low", "close"]
        )
    rows = duck.execute(
        """
        WITH ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY symbol, bar_date
                ORDER BY ingested_at DESC
            ) AS rn
            FROM ohlcv_daily
            WHERE symbol = ANY(?) AND bar_date BETWEEN ? AND ?
        )
        SELECT symbol, bar_date, open, high, low, close
        FROM ranked WHERE rn = 1
        ORDER BY symbol, bar_date
        """,
        [symbols, start, end],
    ).df()
    if not rows.empty:
        rows["bar_date"] = pd.to_datetime(rows["bar_date"]).dt.date
    return rows


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------


def _ibkr_lite_close_cost(qty: float, price: float) -> float:
    """Regulatory pass-through on sells only. Opens are free under IBKR Lite.

    Components (sells/closes only):
      - SEC Section 31 fee: $0.0000278 × notional. Reg fee, applies to sells.
      - FINRA TAF: $0.000166/share, capped at $8.30 per execution.
    Sum then min $0.01 (smallest fee charged in practice).
    """
    notional = qty * price
    sec_fee = 0.0000278 * notional
    taf = min(0.000166 * qty, 8.30)
    total = sec_fee + taf
    return max(0.01, total) if total > 0 else 0.0


def _close_cost(model: str, qty: float, price: float) -> float:
    if model == "none":
        return 0.0
    if model == "ibkr_lite":
        return _ibkr_lite_close_cost(qty, price)
    raise ValueError(f"unknown commission_model={model!r}")


# ---------------------------------------------------------------------------
# Stop-loss support
# ---------------------------------------------------------------------------


def _rolling_low(
    symbol: str,
    bars_by_sym: dict,
    on_date: date,
    lookback_days: int,
) -> float | None:
    """Lowest `low` over the prior `lookback_days` bars (inclusive of on_date).

    `bars_by_sym[symbol]` is a list of (bar_date, low) tuples sorted by date.
    Returns None if we don't have any bars in the window.
    """
    seq = bars_by_sym.get(symbol)
    if not seq:
        return None
    # Find the index of the last bar with bar_date <= on_date.
    # Binary-search would be ideal but the lists are short enough.
    lows = [low for d, low in seq if d <= on_date and not pd.isna(low)]
    if not lows:
        return None
    window = lows[-lookback_days:]
    return float(min(window)) if window else None


def _rolling_atr(
    symbol: str,
    hlc_by_sym: dict,
    on_date: date,
    lookback_days: int,
) -> float | None:
    """Wilder's Average True Range over the prior `lookback_days` bars.

    True Range per bar = max(
        high - low,
        |high - prev_close|,
        |low - prev_close|
    )
    ATR is the rolling mean (or Wilder's smoothing) of TR. We use a
    simple mean because the engine cares about a stable per-day value,
    not the fine-grained smoothing dynamics traders use intraday.

    `hlc_by_sym[symbol]` is a list of (bar_date, high, low, close) tuples
    sorted by date. Returns None if we don't have enough bars.
    """
    seq = hlc_by_sym.get(symbol)
    if not seq:
        return None
    valid = [
        (d, h, low, c)
        for d, h, low, c in seq
        if d <= on_date
        and h is not None
        and low is not None
        and c is not None
        and not pd.isna(h)
        and not pd.isna(low)
        and not pd.isna(c)
    ]
    if len(valid) < 2:
        return None
    window = valid[-(lookback_days + 1):]  # need prev_close for first TR
    if len(window) < 2:
        return None
    trs: list[float] = []
    for i in range(1, len(window)):
        _, h, low, _ = window[i]
        _, _, _, pc = window[i - 1]
        tr = max(float(h - low), abs(float(h - pc)), abs(float(low - pc)))
        trs.append(tr)
    if not trs:
        return None
    return float(sum(trs) / len(trs))


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------


def _persist_run_metadata(
    conn: sqlite3.Connection, cfg: StrategyConfig, started_at: str
) -> None:
    conn.execute(
        """
        INSERT INTO paper_runs (
            run_id, universe, starting_cash, position_size,
            n_long, n_short, short_threshold, started_at, notes,
            holding_days, commission_model, stop_loss_enabled,
            support_lookback_days, stop_buffer_pct
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            universe = excluded.universe,
            starting_cash = excluded.starting_cash,
            position_size = excluded.position_size,
            n_long = excluded.n_long,
            n_short = excluded.n_short,
            short_threshold = excluded.short_threshold,
            started_at = excluded.started_at,
            notes = excluded.notes,
            holding_days = excluded.holding_days,
            commission_model = excluded.commission_model,
            stop_loss_enabled = excluded.stop_loss_enabled,
            support_lookback_days = excluded.support_lookback_days,
            stop_buffer_pct = excluded.stop_buffer_pct
        """,
        (
            cfg.run_id, cfg.universe, cfg.starting_cash, cfg.position_size,
            cfg.n_long, cfg.n_short, cfg.short_threshold, started_at, cfg.notes,
            cfg.holding_days, cfg.commission_model, int(cfg.stop_loss_enabled),
            cfg.support_lookback_days, cfg.stop_buffer_pct,
        ),
    )


def _clear_existing_run(conn: sqlite3.Connection, run_id: str) -> None:
    for tbl in ("paper_trades", "paper_positions", "paper_equity"):
        conn.execute(f"DELETE FROM {tbl} WHERE run_id = ?", (run_id,))


def _seed_starting_equity(cfg: StrategyConfig) -> None:
    """Persist a $starting_cash row at cfg.start_date so the UI shows a
    baseline even before any predictions land.

    Used when the backtest finds no predictions in the requested window
    (e.g. fresh live account whose first prediction hasn't been logged yet).
    Bookends the live "go-live" line at $1,000 across the equity curve.
    """
    paper_db = init_paper_db(cfg.paper_db_path)
    seed_date = (cfg.start_date or date.today()).isoformat()
    started_at = pd.Timestamp.utcnow().isoformat()
    conn = sqlite3.connect(paper_db)
    try:
        _persist_run_metadata(conn, cfg, started_at)
        _clear_existing_run(conn, cfg.run_id)
        for kind in ("open_8am_ct", "close_5pm_ct"):
            conn.execute(
                "INSERT OR REPLACE INTO paper_equity "
                "(run_id, trade_date, snapshot_kind, equity, cash, long_mv, "
                "short_mv, realized_pnl, unrealized_pnl) "
                "VALUES (?, ?, ?, ?, ?, 0, 0, 0, 0)",
                (cfg.run_id, seed_date, kind, cfg.starting_cash, cfg.starting_cash),
            )
        conn.execute(
            "UPDATE paper_runs SET first_trade_date=?, last_trade_date=?, "
            "final_equity=?, final_realized_pnl=0 WHERE run_id=?",
            (seed_date, seed_date, cfg.starting_cash, cfg.run_id),
        )
        conn.commit()
    finally:
        conn.close()


@dataclass
class _Lot:
    """One open position. `lot_id` groups lots opened on the same day."""
    lot_id: int
    symbol: str
    qty: float
    entry_price: float
    entry_date: date
    entry_idx: int  # index into trade_dates
    stop_level: float | None


def backtest(cfg: StrategyConfig = DEFAULT_CONFIG) -> dict:
    """Run the long-only overlapping-portfolios backtest end-to-end."""
    paper_db = init_paper_db(cfg.paper_db_path)
    log.info(
        f"paper backtest: run_id={cfg.run_id} cash=${cfg.starting_cash:.0f} "
        f"n_long={cfg.n_long} holding_days={cfg.holding_days} "
        f"stop={'on' if cfg.stop_loss_enabled else 'off'} "
        f"(N={cfg.support_lookback_days}, buf={cfg.stop_buffer_pct:.4f}) "
        f"costs={cfg.commission_model}"
    )

    predictions = _load_predictions(
        cfg.universe, cfg.start_date, cfg.end_date,
        sqlite_path=cfg.predictions_sqlite_path,
    )
    if predictions.empty:
        log.warning(
            "paper backtest: no predictions in window; seeding starting "
            "equity row so the UI has a baseline"
        )
        _seed_starting_equity(cfg)
        return {"run_id": cfg.run_id, "n_trade_days": 0, "final_equity": cfg.starting_cash}

    trade_dates: list[date] = sorted(predictions["as_of"].unique().tolist())
    trade_date_idx = {d: i for i, d in enumerate(trade_dates)}
    pred_start = trade_dates[0]
    pred_end = trade_dates[-1]
    log.info(f"paper backtest: {len(trade_dates)} trade days {pred_start} -> {pred_end}")

    # Pull OHLCV for all symbols referenced. We extend the start backwards
    # by `support_lookback_days * 2` so the very first day's stop-level has
    # enough history (calendar days != trading days so widen generously).
    all_syms = sorted(set(predictions["symbol"].unique().tolist()))
    lookback_pad = max(60, cfg.support_lookback_days * 3)
    # Read-only OHLCV pull — backtest never writes to market.duckdb. Read-only
    # mode lets us coexist with the writer pipeline (yfinance refresh, walk-
    # forward training) without lock contention.
    duck = duckdb.connect(settings.duckdb_path, read_only=True)
    try:
        ohlcv = _load_ohlcv_for_symbols(
            duck,
            all_syms,
            pred_start - pd.Timedelta(days=lookback_pad).to_pytimedelta(),
            pred_end + pd.Timedelta(days=15).to_pytimedelta(),
        )
    finally:
        duck.close()
    if ohlcv.empty:
        log.error("paper backtest: no OHLCV for any prediction symbol; aborting")
        return {"run_id": cfg.run_id, "n_trade_days": 0, "final_equity": cfg.starting_cash}

    # Build the regime gate once (loads ~1y of SPY closes). When disabled
    # we use a no-op stub so the inner loop stays branch-free.
    if cfg.regime_gate_enabled:
        from packages.paper_trading.regime import RegimeGate
        regime_gate = RegimeGate(as_of_max=trade_dates[-1])
    else:
        regime_gate = None

    # Three views over OHLCV:
    #   bars[(sym, date)] -> {'open':..., 'high':..., 'low':..., 'close':...}
    #       for fast per-day price lookups (marking + execution)
    #   lows_by_sym[sym] = [(date, low), ...] sorted by date
    #       for the rolling-support stop calculation (stop_mode='support')
    #   hlc_by_sym[sym] = [(date, high, low, close), ...] sorted by date
    #       for the rolling-ATR stop calculation (stop_mode='atr') and
    #       for inverse-vol position sizing
    bars = ohlcv.set_index(["symbol", "bar_date"]).to_dict("index")
    lows_by_sym: dict[str, list[tuple[date, float]]] = {}
    hlc_by_sym: dict[str, list[tuple[date, float, float, float]]] = {}
    for (sym, d), row in bars.items():
        lows_by_sym.setdefault(sym, []).append((d, row.get("low")))
        hlc_by_sym.setdefault(sym, []).append((
            d, row.get("high"), row.get("low"), row.get("close"),
        ))
    for sym in lows_by_sym:
        lows_by_sym[sym].sort(key=lambda x: x[0])
    for sym in hlc_by_sym:
        hlc_by_sym[sym].sort(key=lambda x: x[0])

    paper_conn = sqlite3.connect(paper_db)
    paper_conn.execute("BEGIN")
    try:
        started_at = pd.Timestamp.utcnow().isoformat()
        _persist_run_metadata(paper_conn, cfg, started_at)
        _clear_existing_run(paper_conn, cfg.run_id)

        cash = float(cfg.starting_cash)
        realized_pnl = 0.0
        next_lot_id = 0
        open_lots: list[_Lot] = []

        for d_idx, d in enumerate(trade_dates):
            day_preds = predictions[predictions["as_of"] == d]
            if day_preds.empty:
                continue

            # 1) 8:35 CT mark using today's OPEN price (proxy for the
            #    snapshot taken right after the bell). Pre-trade.
            long_mv_open, unreal_open = _mark_lots(open_lots, bars, d, "open")
            equity_8am = cash + long_mv_open
            paper_conn.execute(
                "INSERT OR REPLACE INTO paper_equity (run_id, trade_date, snapshot_kind, "
                "equity, cash, long_mv, short_mv, realized_pnl, unrealized_pnl) "
                "VALUES (?, ?, 'open_8am_ct', ?, ?, ?, 0, ?, ?)",
                (cfg.run_id, d.isoformat(), equity_8am, cash, long_mv_open,
                 realized_pnl, unreal_open),
            )

            # 2) Force-close lots whose holding window has elapsed.
            #    A lot opened on entry_idx exits at the open of entry_idx + holding_days.
            still_open: list[_Lot] = []
            for lot in open_lots:
                if d_idx - lot.entry_idx >= cfg.holding_days:
                    cash, realized_pnl = _close_lot(
                        paper_conn, cfg, d, lot, bars, cash, realized_pnl,
                        side="long_close", at_price=None,  # use today's open
                    )
                else:
                    still_open.append(lot)
            open_lots = still_open

            # 3) Open a new slice for today, sized at equity_8am / holding_days.
            #    Within slice, weight by combined_score.
            top_longs = day_preds.nlargest(cfg.n_long, "combined_score")
            # Regime gate: scale slice budget down in strong-trend regimes.
            # See packages/paper_trading/regime.py for the band table.
            regime_mult = 1.0
            if regime_gate is not None:
                regime_mult = regime_gate.multiplier_for(d)
            slice_budget = max(0.0, equity_8am / cfg.holding_days) * regime_mult
            if not top_longs.empty and slice_budget > 0:
                # If vol_scaling is on, weights = combined_score / ATR per
                # name. Names with bigger ATR get less capital — equal-risk
                # rather than equal-conviction. Falls back to plain combined-
                # score weights if ATR is unavailable for a name.
                atr_by_sym: dict[str, float] = {}
                if cfg.vol_scaling == "inverse":
                    for sym in top_longs["symbol"]:
                        atr = _rolling_atr(sym, hlc_by_sym, d, cfg.atr_lookback_days)
                        if atr is not None and atr > 0:
                            atr_by_sym[sym] = atr
                weights = _slice_weights(
                    top_longs,
                    conviction=cfg.conviction_weighted,
                    vol_scaling=cfg.vol_scaling,
                    atr_by_sym=atr_by_sym,
                )
                lot_id_today = next_lot_id
                next_lot_id += 1
                for sym, weight in weights.items():
                    if weight <= 0:
                        continue
                    bar = bars.get((sym, d))
                    if not bar or pd.isna(bar.get("open")):
                        continue
                    px = float(bar["open"])
                    if px <= 0:
                        continue
                    pos_dollars = weight * slice_budget
                    qty = pos_dollars / px
                    if qty <= 0:
                        continue
                    # Stop-level. Two modes:
                    #   'support' — N-day rolling low × (1 - buffer)
                    #   'atr'     — entry_price - K × ATR(N)
                    # Either way, falls back to None if there isn't enough
                    # history; engine treats None as "no stop on this lot".
                    stop_level = None
                    if cfg.stop_loss_enabled:
                        if cfg.stop_mode == "atr":
                            atr = atr_by_sym.get(sym) or _rolling_atr(
                                sym, hlc_by_sym, d, cfg.atr_lookback_days
                            )
                            if atr is not None and atr > 0:
                                stop_level = px - cfg.atr_multiplier * atr
                        else:  # 'support' or any unknown → support fallback
                            support = _rolling_low(
                                sym, lows_by_sym, d, cfg.support_lookback_days
                            )
                            if support is not None:
                                stop_level = support * (1.0 - cfg.stop_buffer_pct)
                    cash -= qty * px  # no commission on opens under IBKR Lite
                    paper_conn.execute(
                        "INSERT OR REPLACE INTO paper_trades (run_id, trade_date, lot_id, "
                        "symbol, side, qty, fill_price, cash_delta, realized_pnl, cost) "
                        "VALUES (?, ?, ?, ?, 'long_open', ?, ?, ?, 0, 0)",
                        (cfg.run_id, d.isoformat(), lot_id_today, sym,
                         qty, px, -(qty * px)),
                    )
                    open_lots.append(_Lot(
                        lot_id=lot_id_today, symbol=sym, qty=qty,
                        entry_price=px, entry_date=d, entry_idx=d_idx,
                        stop_level=stop_level,
                    ))

            # 4) Persist intra-day position snapshot (after open trades, before stops).
            for lot in open_lots:
                paper_conn.execute(
                    "INSERT OR REPLACE INTO paper_positions "
                    "(run_id, trade_date, lot_id, symbol, side, qty, entry_price, "
                    "entry_date, stop_level) VALUES (?, ?, ?, ?, 'long', ?, ?, ?, ?)",
                    (cfg.run_id, d.isoformat(), lot.lot_id, lot.symbol,
                     lot.qty, lot.entry_price, lot.entry_date.isoformat(),
                     lot.stop_level),
                )

            # 5) Stop-loss evaluation at 5 PM mark.
            #    If today's close <= stop_level, force-exit at stop_level
            #    (per spec — gaps/slippage explicitly ignored).
            if cfg.stop_loss_enabled:
                survivors: list[_Lot] = []
                for lot in open_lots:
                    if lot.stop_level is None:
                        survivors.append(lot)
                        continue
                    bar = bars.get((lot.symbol, d))
                    close_px = bar.get("close") if bar else None
                    if close_px is None or pd.isna(close_px):
                        survivors.append(lot)
                        continue
                    if float(close_px) <= lot.stop_level:
                        cash, realized_pnl = _close_lot(
                            paper_conn, cfg, d, lot, bars, cash, realized_pnl,
                            side="stop_close", at_price=lot.stop_level,
                        )
                    else:
                        survivors.append(lot)
                open_lots = survivors

            # 6) 5 PM CT mark — using today's CLOSE for survivors.
            long_mv_close, unreal_close = _mark_lots(open_lots, bars, d, "close")
            equity_5pm = cash + long_mv_close
            paper_conn.execute(
                "INSERT OR REPLACE INTO paper_equity (run_id, trade_date, snapshot_kind, "
                "equity, cash, long_mv, short_mv, realized_pnl, unrealized_pnl) "
                "VALUES (?, ?, 'close_5pm_ct', ?, ?, ?, 0, ?, ?)",
                (cfg.run_id, d.isoformat(), equity_5pm, cash, long_mv_close,
                 realized_pnl, unreal_close),
            )

        final_long_mv, final_unreal = _mark_lots(open_lots, bars, trade_dates[-1], "close")
        final_equity = cash + final_long_mv
        paper_conn.execute(
            "UPDATE paper_runs SET first_trade_date=?, last_trade_date=?, final_equity=?, "
            "final_realized_pnl=? WHERE run_id=?",
            (trade_dates[0].isoformat(), trade_dates[-1].isoformat(),
             final_equity, realized_pnl, cfg.run_id),
        )
        paper_conn.commit()

        return {
            "run_id": cfg.run_id,
            "n_trade_days": len(trade_dates),
            "first_trade_date": trade_dates[0].isoformat(),
            "last_trade_date": trade_dates[-1].isoformat(),
            "starting_cash": cfg.starting_cash,
            "final_equity": final_equity,
            "final_realized_pnl": realized_pnl,
        }
    except Exception:
        paper_conn.rollback()
        raise
    finally:
        paper_conn.close()


# ---------------------------------------------------------------------------
# Mark / open / close helpers
# ---------------------------------------------------------------------------


def _mark_lots(
    lots: list[_Lot], bars: dict, on_date: date, price_field: str
) -> tuple[float, float]:
    """Return (long_mv, unrealized_pnl) at the given price_field for `on_date`."""
    long_mv = unreal = 0.0
    for lot in lots:
        bar = bars.get((lot.symbol, on_date))
        if not bar or pd.isna(bar.get(price_field)):
            # Use entry as fallback when today's bar is missing (holiday, halt).
            price = lot.entry_price
        else:
            price = float(bar[price_field])
        long_mv += lot.qty * price
        unreal += lot.qty * (price - lot.entry_price)
    return long_mv, unreal


def _close_lot(
    conn: sqlite3.Connection,
    cfg: StrategyConfig,
    on_date: date,
    lot: _Lot,
    bars: dict,
    cash: float,
    realized_pnl: float,
    *,
    side: str,
    at_price: float | None,
) -> tuple[float, float]:
    """Close `lot` on `on_date`. `at_price` overrides bar price (used for stops).

    Returns updated (cash, realized_pnl). Persists a row in paper_trades.
    """
    if at_price is not None:
        price = float(at_price)
    else:
        bar = bars.get((lot.symbol, on_date))
        price = (
            float(bar["open"])
            if bar and not pd.isna(bar.get("open"))
            else lot.entry_price
        )
    gross = lot.qty * price
    cost = _close_cost(cfg.commission_model, lot.qty, price)
    cash_delta = gross - cost
    trade_pnl = lot.qty * (price - lot.entry_price) - cost
    conn.execute(
        "INSERT OR REPLACE INTO paper_trades (run_id, trade_date, lot_id, symbol, side, "
        "qty, fill_price, cash_delta, realized_pnl, cost) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (cfg.run_id, on_date.isoformat(), lot.lot_id, lot.symbol, side,
         lot.qty, price, cash_delta, trade_pnl, cost),
    )
    return cash + cash_delta, realized_pnl + trade_pnl


def _slice_weights(
    top_longs: pd.DataFrame,
    *,
    conviction: bool,
    vol_scaling: str = "none",
    atr_by_sym: dict[str, float] | None = None,
) -> pd.Series:
    """Return a {symbol: weight} mapping summing to ≤ 1.

    Three layered modifiers:
      1. conviction=True ⇒ raw score = clip(combined_score, 0, ∞).
         conviction=False ⇒ raw score = 1 (equal weight).
      2. vol_scaling='inverse' AND atr_by_sym non-empty ⇒ multiply raw
         scores by 1/ATR per symbol (lower-vol names get more weight).
         Names without ATR keep their raw score (defensive fallback).
      3. Normalize so weights sum to 1. If sum ≤ 0, fall back to equal.
    """
    n = len(top_longs)
    symbols = top_longs["symbol"].values
    if conviction:
        raw = top_longs["combined_score"].clip(lower=0.0).to_numpy(dtype=float)
    else:
        raw = pd.Series([1.0] * n, index=range(n)).to_numpy(dtype=float)

    if vol_scaling == "inverse" and atr_by_sym:
        adjusted = raw.copy()
        for i, sym in enumerate(symbols):
            atr = atr_by_sym.get(sym)
            if atr is not None and atr > 0:
                adjusted[i] = raw[i] / atr
        raw = adjusted

    total = float(raw.sum())
    if total <= 0:
        eq = 1.0 / n
        return pd.Series(eq, index=symbols)
    return pd.Series(raw / total, index=symbols)
