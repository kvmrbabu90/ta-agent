"""Corporate-actions cross-checker.

This module does NOT apply price adjustments — IB and Kite already return
adjusted prices, and yfinance does too when ``auto_adjust=True``. Its job
is to flag *disagreements* between sources, which usually mean one of them
has a stale or missed split / dividend adjustment.

We only detect; we don't auto-correct. Resolution is a human decision —
typically: re-run the offending adapter, or manually wipe and re-ingest
the affected symbol.

Public API:
    compare_sources(symbol, start, end, tolerance_pct=1.0) -> DataFrame
    audit_universe(universe, lookback_days=365)            -> DataFrame
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from packages.common.logging import log
from packages.ingestion.storage import get_conn
from packages.ingestion.universe.membership import members_on


def _load_per_source_closes(
    symbol: str, start: date, end: date, *, duckdb_path: str | None = None
) -> pd.DataFrame:
    """Return one row per (bar_date, source) for the symbol over the window.
    Columns: bar_date, source, close, close_unadj."""
    with get_conn(duckdb_path) as conn:
        return conn.execute(
            """
            SELECT bar_date, source, close, close_unadj
            FROM ohlcv_daily
            WHERE symbol = ?
              AND bar_date BETWEEN ? AND ?
            ORDER BY bar_date, source
            """,
            [symbol, start, end],
        ).df()


def _classify_diff(pct_diff: float, ratio_a: float | None, ratio_b: float | None) -> str:
    """Heuristic label for an inter-source disagreement.

    pct_diff:        |close_a - close_b| / mean(close_a, close_b) * 100
    ratio_a / ratio_b: close_unadj / close per source (None if unavailable)
    """
    if pct_diff > 30.0:
        return "split"

    # Dividend heuristic: small but real disagreement, AND the unadj/adj ratio
    # differs noticeably between sources (one sees the ex-div drop, the other
    # doesn't).
    if (
        0.5 <= pct_diff <= 5.0
        and ratio_a is not None
        and ratio_b is not None
        and abs(ratio_a - ratio_b) > 0.005
    ):
        return "dividend"

    return "outlier"


def compare_sources(
    symbol: str,
    start: date,
    end: date,
    *,
    tolerance_pct: float = 1.0,
    duckdb_path: str | None = None,
) -> pd.DataFrame:
    """Find dates where two sources disagree on close price by > tolerance_pct.

    Returns columns:
        bar_date, source_a, close_a, source_b, close_b,
        pct_diff, suspected_cause
    """
    rows = _load_per_source_closes(symbol, start, end, duckdb_path=duckdb_path)
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "bar_date", "source_a", "close_a", "source_b", "close_b",
                "pct_diff", "suspected_cause",
            ]
        )

    # Pivot per-source close into a wide frame keyed on bar_date.
    closes = rows.pivot(index="bar_date", columns="source", values="close")
    unadj = rows.pivot(index="bar_date", columns="source", values="close_unadj")

    sources = list(closes.columns)
    if len(sources) < 2:
        return pd.DataFrame(
            columns=[
                "bar_date", "source_a", "close_a", "source_b", "close_b",
                "pct_diff", "suspected_cause",
            ]
        )

    out: list[dict] = []
    for i, src_a in enumerate(sources):
        for src_b in sources[i + 1 :]:
            joined = pd.DataFrame(
                {
                    "close_a": closes[src_a],
                    "close_b": closes[src_b],
                    "unadj_a": unadj[src_a] if src_a in unadj.columns else None,
                    "unadj_b": unadj[src_b] if src_b in unadj.columns else None,
                }
            ).dropna(subset=["close_a", "close_b"])

            if joined.empty:
                continue

            mean_close = (joined["close_a"] + joined["close_b"]) / 2.0
            joined["pct_diff"] = (
                (joined["close_a"] - joined["close_b"]).abs() / mean_close * 100.0
            )
            offenders = joined[joined["pct_diff"] > tolerance_pct]

            for bar_dt, row in offenders.iterrows():
                ratio_a = (
                    float(row["unadj_a"]) / float(row["close_a"])
                    if row["close_a"] and row["unadj_a"] is not None and not pd.isna(row["unadj_a"])
                    else None
                )
                ratio_b = (
                    float(row["unadj_b"]) / float(row["close_b"])
                    if row["close_b"] and row["unadj_b"] is not None and not pd.isna(row["unadj_b"])
                    else None
                )
                out.append(
                    {
                        "bar_date": bar_dt,
                        "source_a": src_a,
                        "close_a": float(row["close_a"]),
                        "source_b": src_b,
                        "close_b": float(row["close_b"]),
                        "pct_diff": float(row["pct_diff"]),
                        "suspected_cause": _classify_diff(
                            float(row["pct_diff"]), ratio_a, ratio_b
                        ),
                    }
                )

    return pd.DataFrame(
        out,
        columns=[
            "bar_date", "source_a", "close_a", "source_b", "close_b",
            "pct_diff", "suspected_cause",
        ],
    )


def audit_universe(
    universe: str,
    lookback_days: int = 365,
    *,
    tolerance_pct: float = 1.0,
    duckdb_path: str | None = None,
) -> pd.DataFrame:
    """Run compare_sources for every current member of the universe.

    Returns a per-symbol summary sorted by max disagreement, descending.
    Columns: symbol, n_disagreements, max_pct_diff, top_cause.
    """
    today = date.today()
    start = today - timedelta(days=lookback_days)
    members = members_on(universe, today)
    if members.empty:
        log.warning(f"audit_universe: empty membership for {universe}")
        return pd.DataFrame(
            columns=["symbol", "n_disagreements", "max_pct_diff", "top_cause"]
        )

    rows: list[dict] = []
    for sym in members["symbol"].tolist():
        try:
            diffs = compare_sources(
                sym,
                start,
                today,
                tolerance_pct=tolerance_pct,
                duckdb_path=duckdb_path,
            )
        except Exception as exc:  # noqa: BLE001
            log.error(f"audit_universe: compare_sources({sym}) failed: {exc!r}")
            continue
        if diffs.empty:
            continue
        top_cause = diffs["suspected_cause"].mode().iat[0]
        rows.append(
            {
                "symbol": sym,
                "n_disagreements": len(diffs),
                "max_pct_diff": float(diffs["pct_diff"].max()),
                "top_cause": top_cause,
            }
        )

    df = pd.DataFrame(
        rows, columns=["symbol", "n_disagreements", "max_pct_diff", "top_cause"]
    )
    return df.sort_values("max_pct_diff", ascending=False).reset_index(drop=True)
