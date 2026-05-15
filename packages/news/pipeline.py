"""Top-level entry: classify today's long picks and persist verdicts.

Daily flow:
    daily_predict
       └─▶ classify_top_picks(universe, as_of, top_n)
              ├─▶ for each pick:
              │     - load company_name + trailing returns from market.duckdb
              │     - fetch_filings_for_symbol(symbol, since=as_of − 21d)
              │     - classify_decline(...)
              │     - upsert into predictions_news_verdict
              └─▶ return summary dict (counts by verdict)

Audit-only: the paper-trading engine does NOT read verdicts. We collect
paired (verdict, realized_return_5d) data for a few weeks first.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

from packages.common.config import settings
from packages.common.logging import log
from packages.news.classifier import DEFAULT_MODEL, classify_decline
from packages.news.edgar_fetcher import fetch_filings_for_symbol
from packages.news.storage import news_conn

# Look back this many calendar days for SEC filings. ~3 weeks covers the
# 20-day return window the model conditions on.
_FILING_LOOKBACK_DAYS = 21


def _load_top_picks(
    universe: str, as_of: date, top_n: int, direction: str = "long",
) -> pd.DataFrame:
    """Pull today's top-N picks by combined_score from predictions_log.

    Mirrors the ranking the dashboard uses:
        combined_score = predicted_return × (1 + dir_agreement)
        where dir_agreement = top_quintile_proba − bottom_quintile_proba

    For longs:  filter predicted_return>0 and combined>0, take top by combined.
    For shorts: filter predicted_return<0 and combined<0, take MOST NEGATIVE.
    """
    conn = sqlite3.connect(settings.predictions_sqlite_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT symbol, predicted_return, top_quintile_proba, bottom_quintile_proba
            FROM predictions_log
            WHERE universe = ? AND as_of = ?
            """,
            conn, params=[universe, as_of.isoformat()],
        )
    finally:
        conn.close()
    if df.empty:
        return df
    df["dir_agree"] = (
        df["top_quintile_proba"].fillna(0.0) - df["bottom_quintile_proba"].fillna(0.0)
    )
    df["combined"] = df["predicted_return"] * (1.0 + df["dir_agree"])
    if direction == "short":
        # Short side: pick most-negative combined (most-confident shorts).
        df = df[df["predicted_return"] < 0]
        df = df[df["combined"] < 0]
        return df.nsmallest(top_n, "combined")[
            ["symbol", "predicted_return", "combined", "dir_agree"]
        ].reset_index(drop=True)
    # Long side: pick most-positive combined.
    df = df[df["predicted_return"] > 0]
    df = df[df["combined"] > 0]
    return df.nlargest(top_n, "combined")[
        ["symbol", "predicted_return", "combined", "dir_agree"]
    ].reset_index(drop=True)


def _trailing_returns(
    symbol: str, as_of: date, *, duckdb_path: str | None = None
) -> tuple[float | None, float | None]:
    """Compute (trail_5d, trail_20d) log returns ending on `as_of`."""
    p = duckdb_path or settings.duckdb_path
    conn = duckdb.connect(p, read_only=True)
    try:
        rows = conn.execute(
            """
            SELECT bar_date, close FROM ohlcv_daily
            WHERE symbol = ? AND bar_date <= ?
            ORDER BY bar_date DESC LIMIT 25
            """,
            [symbol, as_of],
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return None, None
    rows = list(reversed(rows))  # ascending date
    closes = np.array([r[1] for r in rows], dtype=float)
    closes = closes[closes > 0]
    if len(closes) < 6:
        return None, None
    trail_5 = float(np.log(closes[-1] / closes[-6])) if len(closes) >= 6 else None
    trail_20 = float(np.log(closes[-1] / closes[-21])) if len(closes) >= 21 else None
    return trail_5, trail_20


def _company_name(symbol: str, *, duckdb_path: str | None = None) -> str | None:
    """Best-effort company name lookup from universe_membership."""
    p = duckdb_path or settings.duckdb_path
    conn = duckdb.connect(p, read_only=True)
    try:
        row = conn.execute(
            "SELECT company_name FROM universe_membership "
            "WHERE symbol = ? AND company_name IS NOT NULL LIMIT 1",
            [symbol],
        ).fetchone()
    except duckdb.Error:
        # Schema may differ — soft-fail. The classifier handles None name.
        return None
    finally:
        conn.close()
    return row[0] if row else None


def _upsert_verdict(
    conn: sqlite3.Connection,
    *,
    universe: str,
    symbol: str,
    as_of: date,
    verdict_obj,
    model_name: str,
    n_sources: int,
    trail_5d: float | None,
    trail_20d: float | None,
    predicted_return: float,
) -> None:
    conn.execute(
        """
        INSERT INTO predictions_news_verdict
            (universe, symbol, as_of, verdict, confidence,
             key_factors_json, evidence_sources_json, model_name,
             n_sources, trail_5d, trail_20d, predicted_return)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (universe, symbol, as_of) DO UPDATE SET
            verdict = excluded.verdict,
            confidence = excluded.confidence,
            key_factors_json = excluded.key_factors_json,
            evidence_sources_json = excluded.evidence_sources_json,
            model_name = excluded.model_name,
            n_sources = excluded.n_sources,
            trail_5d = excluded.trail_5d,
            trail_20d = excluded.trail_20d,
            predicted_return = excluded.predicted_return,
            created_at = CURRENT_TIMESTAMP
        """,
        (
            universe, symbol, as_of.isoformat(),
            verdict_obj.verdict, verdict_obj.confidence,
            json.dumps(verdict_obj.key_factors),
            json.dumps(verdict_obj.evidence_sources),
            model_name, n_sources,
            trail_5d, trail_20d, predicted_return,
        ),
    )


def _classify_one_side(
    *,
    universe: str,
    as_of: date,
    top_n: int,
    direction: str,
    since: date,
    model: str,
    duckdb_path: str | None,
) -> tuple[dict[str, int], list[dict]]:
    """Classify top-N picks for one side (long or short). Returns (counts, per_pick_results)."""
    picks = _load_top_picks(universe, as_of, top_n, direction=direction)
    counts: dict[str, int] = {}
    results: list[dict] = []
    if picks.empty:
        log.info(f"news_classify[{direction}]: no picks for {universe} {as_of}")
        return counts, results
    log.info(f"news_classify[{direction}]: {len(picks)} picks to classify")

    for _i, row in picks.iterrows():
        sym = row["symbol"]
        t5, t20 = _trailing_returns(sym, as_of, duckdb_path=duckdb_path)
        company = _company_name(sym, duckdb_path=duckdb_path)
        filings = fetch_filings_for_symbol(
            sym, since=since, max_filings=8, duckdb_path=duckdb_path,
        )
        v = classify_decline(
            symbol=sym,
            filings=filings,
            company_name=company,
            trail_5d=t5,
            trail_20d=t20,
            predicted_return=float(row["predicted_return"]),
            direction=direction,
            model=model,
        )
        with news_conn() as conn:
            _upsert_verdict(
                conn,
                universe=universe, symbol=sym, as_of=as_of,
                verdict_obj=v, model_name=model,
                n_sources=len(filings), trail_5d=t5, trail_20d=t20,
                predicted_return=float(row["predicted_return"]),
            )
            conn.commit()
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
        results.append({
            "symbol": sym, "direction": direction, "verdict": v.verdict,
            "confidence": v.confidence, "n_filings": len(filings),
        })
        log.info(
            f"  [{direction}] {sym}: {v.verdict} ({v.confidence:.2f})  "
            f"trail_20d={t20*100 if t20 else 0:+.1f}%  "
            f"sources={len(filings)}  factors={v.key_factors[:2]}"
        )
    return counts, results


def classify_top_picks(
    universe: str = "SP500",
    as_of: date | None = None,
    *,
    top_n: int = 10,
    sides: tuple[str, ...] = ("long", "short"),
    model: str = DEFAULT_MODEL,
    duckdb_path: str | None = None,
) -> dict:
    """Classify the day's top-N picks per side. Returns a summary dict.

    Idempotent: re-running for the same (universe, as_of) overwrites
    prior verdicts. Filings are fetched lazily — the SEC HTTP path only
    fires the first time we encounter an accession.

    Defaults to BOTH sides — pass sides=('long',) or ('short',) to limit.
    The same (universe, symbol, as_of) row can't be both long and short
    on the same day (positive and negative predicted return are disjoint),
    so verdicts don't collide.
    """
    as_of = as_of or date.today()
    log.info(
        f"news_classify: universe={universe} as_of={as_of} "
        f"top_n={top_n} sides={sides}"
    )

    since = as_of - timedelta(days=_FILING_LOOKBACK_DAYS)
    t_start = time.monotonic()
    total_counts: dict[str, int] = {}
    all_results: list[dict] = []

    for direction in sides:
        side_counts, side_results = _classify_one_side(
            universe=universe, as_of=as_of, top_n=top_n,
            direction=direction, since=since, model=model,
            duckdb_path=duckdb_path,
        )
        for k, v in side_counts.items():
            total_counts[k] = total_counts.get(k, 0) + v
        all_results.extend(side_results)

    elapsed = time.monotonic() - t_start
    summary = {
        "universe": universe,
        "as_of": as_of.isoformat(),
        "n_classified": len(all_results),
        "by_verdict": total_counts,
        "elapsed_s": round(elapsed, 1),
        "picks": all_results,
    }
    log.info(f"news_classify complete: {total_counts}  elapsed={elapsed:.1f}s")
    return summary
