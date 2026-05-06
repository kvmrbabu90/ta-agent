"""Prediction targets: forward log returns and cross-sectional quintile labels.

Two non-negotiable rules baked into this module:

1. **Trading-day horizons.** ``horizon_days`` counts bars in each symbol's
   own time series, not calendar days. We use a per-symbol ``shift(-N)`` so
   weekends and holidays are skipped naturally.

2. **PIT-respecting cross-sectional ranks.** Quintile labels for date T are
   computed using ONLY symbols that were members of the universe on date T.
   Pulling the membership snapshot at T+horizon is leakage.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def compute_forward_log_return(
    ohlcv: pd.DataFrame,
    horizon_days: int = 5,
) -> pd.DataFrame:
    """For each (symbol, bar_date) row, compute log(close[t + horizon] / close[t]).

    Inputs:
        ohlcv: long-format DataFrame with columns symbol, bar_date, close.
               Within each symbol, rows must be sorted ascending by bar_date.
    Output:
        DataFrame with columns: symbol, bar_date, fwd_return_{horizon}d.
        The last ``horizon_days`` rows of each symbol are NaN.
    """
    required = {"symbol", "bar_date", "close"}
    missing = required - set(ohlcv.columns)
    if missing:
        raise ValueError(f"compute_forward_log_return: missing columns {missing}")

    out = ohlcv[["symbol", "bar_date"]].copy()
    df = ohlcv.sort_values(["symbol", "bar_date"]).copy()
    df["close"] = df["close"].astype(float)
    # Per-symbol shift so the horizon counts trading bars within each symbol.
    df["future_close"] = df.groupby("symbol", sort=False)["close"].shift(-horizon_days)
    df[f"fwd_return_{horizon_days}d"] = np.log(df["future_close"] / df["close"])

    # Re-align to the original ohlcv index so callers can join cleanly.
    df = df[["symbol", "bar_date", f"fwd_return_{horizon_days}d"]]
    out = out.merge(df, on=["symbol", "bar_date"], how="left")
    return out


def compute_quintile_labels(
    forward_returns: pd.DataFrame,
    membership_table: pd.DataFrame,
    universe: str,
    horizon_days: int = 5,
    n_classes: int = 5,
) -> pd.DataFrame:
    """Cross-sectional class labels: ``n_classes`` quantile buckets of
    forward returns *within each (universe, date)*.

    Inputs:
        forward_returns: columns symbol, bar_date, fwd_return_{horizon}d
        membership_table: rows for the universe with columns
                          universe, symbol, start_date, end_date
        universe: name string, used to filter the membership table
    Output:
        symbol, bar_date, fwd_return_{horizon}d, fwd_quintile_{horizon}d.
        ``fwd_quintile_*`` is in {0, …, n_classes-1} or NaN where:
          • the symbol was NOT a member on that date, OR
          • the forward return is undefined (insufficient future data), OR
          • on a given date the universe has fewer than n_classes
            distinct rankable returns.
    """
    if "symbol" not in forward_returns.columns or "bar_date" not in forward_returns.columns:
        raise ValueError("forward_returns must have symbol, bar_date columns")

    fwd_col = f"fwd_return_{horizon_days}d"
    out_col = f"fwd_quintile_{horizon_days}d"
    if fwd_col not in forward_returns.columns:
        raise ValueError(f"forward_returns missing column {fwd_col}")

    # Filter membership to this universe, then materialize start/end as dates.
    mem = membership_table[membership_table["universe"] == universe].copy()
    if mem.empty:
        out = forward_returns.copy()
        out[out_col] = pd.NA
        return out[["symbol", "bar_date", fwd_col, out_col]]

    mem["start_date"] = pd.to_datetime(mem["start_date"]).dt.date
    mem["end_date"] = pd.to_datetime(
        mem["end_date"].fillna(pd.Timestamp.max.date())
    ).dt.date

    panel = forward_returns.copy()
    panel["bar_date"] = pd.to_datetime(panel["bar_date"]).dt.date

    # Mark in-universe membership at each date BEFORE ranking.
    membership_pairs = panel.merge(
        mem[["symbol", "start_date", "end_date"]], on="symbol", how="left"
    )
    in_window = (
        (membership_pairs["start_date"] <= membership_pairs["bar_date"])
        & (membership_pairs["bar_date"] <= membership_pairs["end_date"])
    )
    is_member = (
        in_window.groupby([membership_pairs["symbol"], membership_pairs["bar_date"]]).max()
        .rename("is_member").reset_index()
    )
    panel = panel.merge(is_member, on=["symbol", "bar_date"], how="left")
    panel["is_member"] = panel["is_member"].fillna(False).astype(bool)

    # Restrict ranking to in-universe + non-NaN forward return.
    rankable_mask = panel["is_member"] & panel[fwd_col].notna()

    def _bucketize(group: pd.Series) -> pd.Series:
        valid = group.dropna()
        if len(valid) < n_classes:
            return pd.Series(np.nan, index=group.index)
        try:
            buckets = pd.qcut(valid, n_classes, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(np.nan, index=group.index)
        out = pd.Series(np.nan, index=group.index)
        out.loc[valid.index] = buckets.astype(float)
        return out

    rankable = panel.loc[rankable_mask, ["bar_date", fwd_col]].copy()
    rankable[out_col] = (
        rankable.groupby("bar_date", observed=True, sort=False)[fwd_col]
        .transform(_bucketize)
    )

    panel[out_col] = pd.NA
    panel.loc[rankable.index, out_col] = rankable[out_col].values

    return panel[["symbol", "bar_date", fwd_col, out_col]]


def members_table_to_intervals(
    raw: pd.DataFrame, universe: str | None = None
) -> pd.DataFrame:
    """Helper: filter the membership table to one universe (if requested) and
    fill open-ended ``end_date`` with a sentinel future date for use in joins."""
    df = raw.copy()
    if universe is not None:
        df = df[df["universe"] == universe]
    df = df.copy()
    df["start_date"] = pd.to_datetime(df["start_date"]).dt.date
    df["end_date"] = pd.to_datetime(
        df["end_date"].fillna(pd.Timestamp.max.date())
    ).dt.date
    return df.reset_index(drop=True)


def in_universe_on_date(
    intervals: pd.DataFrame, symbol: str, on: date
) -> bool:
    """Convenience predicate for tests / debugging."""
    sub = intervals[intervals["symbol"] == symbol]
    if sub.empty:
        return False
    return bool(((sub["start_date"] <= on) & (on <= sub["end_date"])).any())
