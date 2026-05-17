"""Compare strategy (Pure ML, no FVG, no regime gate) vs SPY buy-and-hold
on an AFTER-TAX basis over the 11-year walk-forward window (2014-2024).

Tax assumptions:
  Strategy: 25% blended STCG (applied annually to net P&L each year).
            5-day holds → all gains short-term → ordinary income tax.
            Losses get the symmetric 25% credit (assumes ability to offset
            against other gains or carry forward).

  SPY B&H:  15% LTCG, paid ONLY at the end when sold. Capital gains
            compound tax-deferred until then. Dividends not modeled
            explicitly — SPY's per-year returns from yfinance already
            include the dividend stream (close-price approximation), so
            we treat the entire return as appreciation. Tiny over-
            estimate of B&H performance vs explicitly modeling the
            ~1.5%/yr qualified-dividend tax drag.

Output:
  - Per-year + cumulative table to stdout
  - Chart at data/processed/tax_adjusted_comparison.png (log scale)
"""

from __future__ import annotations

import math
import sqlite3
import sys
from datetime import date
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from packages.common.config import settings

WF_PAPER = "data/processed/walkforward_10yr/analysis.sqlite"
PURE_ML_RUN_ID = "a_0_0_1"  # pure ML config from analyze_10yr_walkforward.py

STRATEGY_TAX_RATE = 0.25  # 25% blended STCG
SPY_LTCG_RATE = 0.15

OUT_PNG = "docs/tax_adjusted_comparison.png"  # docs/ is git-tracked; data/processed is gitignored


def strategy_per_year_returns() -> pd.DataFrame:
    """Pull the pure-ML strategy's per-year returns from the WF analysis."""
    conn = sqlite3.connect(WF_PAPER)
    try:
        df = pd.read_sql_query(
            "SELECT trade_date, equity FROM paper_equity "
            "WHERE run_id = ? AND snapshot_kind = 'close_5pm_ct' "
            "ORDER BY trade_date",
            conn, params=[PURE_ML_RUN_ID],
        )
    finally:
        conn.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["year"] = pd.to_datetime(df["trade_date"]).dt.year
    rows = []
    for year, g in df.groupby("year"):
        eqs = g["equity"].to_numpy()
        rows.append({"year": int(year), "return": eqs[-1] / eqs[0] - 1})
    return pd.DataFrame(rows)


def spy_per_year_returns(start: date, end: date) -> pd.DataFrame:
    duck = duckdb.connect(settings.duckdb_path, read_only=True)
    try:
        rows = duck.execute(
            "SELECT bar_date, close FROM ohlcv_daily WHERE symbol='SPY' "
            "AND bar_date BETWEEN ? AND ? ORDER BY bar_date",
            [start, end],
        ).fetchall()
    finally:
        duck.close()
    df = pd.DataFrame(rows, columns=["trade_date", "close"])
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["year"] = pd.to_datetime(df["trade_date"]).dt.year
    out = []
    for year, g in df.groupby("year"):
        closes = g["close"].to_numpy()
        out.append({"year": int(year), "return": closes[-1] / closes[0] - 1})
    return pd.DataFrame(out)


def compound_after_tax_annual(
    yearly_returns: pd.DataFrame, tax_rate: float, starting: float = 1000.0,
) -> pd.DataFrame:
    """For strategies that realize gains annually (STCG model)."""
    eq = starting
    rows = []
    for _, row in yearly_returns.iterrows():
        r = row["return"]
        after_tax_r = r * (1 - tax_rate)
        eq = eq * (1 + after_tax_r)
        rows.append({
            "year": int(row["year"]),
            "pretax_return": r * 100,
            "aftertax_return": after_tax_r * 100,
            "equity_after_tax": eq,
        })
    return pd.DataFrame(rows)


def compound_deferred(
    yearly_returns: pd.DataFrame, ltcg_rate: float, starting: float = 1000.0,
) -> pd.DataFrame:
    """For B&H: gains compound tax-deferred; LTCG applied only at terminal sell.
    Per-year `equity_after_tax` shows the MARK-TO-MARKET liquidation value
    (what you'd net IF you sold that year)."""
    eq = starting
    rows = []
    for _, row in yearly_returns.iterrows():
        r = row["return"]
        eq = eq * (1 + r)
        liquidation_value = starting + (eq - starting) * (1 - ltcg_rate)
        rows.append({
            "year": int(row["year"]),
            "pretax_return": r * 100,
            "equity_pretax": eq,
            "equity_after_tax": liquidation_value,
        })
    return pd.DataFrame(rows)


def main() -> int:
    strat_yr = strategy_per_year_returns()
    if strat_yr.empty:
        print(f"ERROR: no equity data for run_id {PURE_ML_RUN_ID} in {WF_PAPER}")
        return 1
    start = date(int(strat_yr["year"].min()), 1, 1)
    end = date(int(strat_yr["year"].max()), 12, 31)
    spy_yr = spy_per_year_returns(start, end)

    strat_after = compound_after_tax_annual(strat_yr, STRATEGY_TAX_RATE)
    spy_after = compound_deferred(spy_yr, SPY_LTCG_RATE)

    merged = strat_after.merge(spy_after, on="year", suffixes=("_strat", "_spy"))

    print()
    print("=" * 110)
    print(f"AFTER-TAX comparison: Strategy (STCG @ {STRATEGY_TAX_RATE:.0%}) vs SPY B&H (LTCG @ {SPY_LTCG_RATE:.0%} at end)")
    print("=" * 110)
    print(
        f"{'year':>6} | {'STRAT pre':>10} {'STRAT post':>10} {'STRAT $':>11} | "
        f"{'SPY pre':>10} {'SPY pre $':>11} {'SPY post $':>11}"
    )
    print("-" * 110)
    for _, r in merged.iterrows():
        print(
            f"{int(r['year']):>6} | "
            f"{r['pretax_return_strat']:>+9.2f}% {r['aftertax_return']:>+9.2f}% "
            f"${r['equity_after_tax_strat']:>10,.0f} | "
            f"{r['pretax_return_spy']:>+9.2f}% ${r['equity_pretax']:>10,.0f} "
            f"${r['equity_after_tax_spy']:>10,.0f}"
        )

    final_strat = merged["equity_after_tax_strat"].iloc[-1]
    final_spy_pre = merged["equity_pretax"].iloc[-1]
    final_spy_post = merged["equity_after_tax_spy"].iloc[-1]
    print()
    print(f"Strategy pre-tax final:  ${final_strat / (1 - 0):,.0f}  (sum of post-tax annual compounds is the same)")
    print(f"Strategy after-tax final: ${final_strat:,.0f}  (annual STCG drag)")
    print(f"SPY pre-tax final:       ${final_spy_pre:,.0f}")
    print(f"SPY after-tax final:     ${final_spy_post:,.0f}  (LTCG at end)")
    print()
    print(f"Strategy / SPY (after-tax): {final_strat / final_spy_post:.2f}x")
    print()

    # === Chart ===
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2.5, 1]})
    fig.patch.set_facecolor("#0f1015")

    # Build daily-ish equity curves by interpolating annual values across the year
    # For aesthetics, plot at each year-end + add starting point at year-0.
    years = [int(merged["year"].iloc[0]) - 1] + [int(y) for y in merged["year"]]
    strat_eq = [1000.0] + list(merged["equity_after_tax_strat"])
    spy_pre_eq = [1000.0] + list(merged["equity_pretax"])
    spy_post_eq = [1000.0] + list(merged["equity_after_tax_spy"])

    ax = axes[0]
    ax.set_facecolor("#0f1015")
    ax.plot(years, strat_eq, color="#34d399", linewidth=2.5, marker="o", markersize=5,
            label=f"Strategy AFTER-TAX (25% STCG annually) → ${final_strat:,.0f}")
    ax.plot(years, spy_pre_eq, color="#9ca3af", linewidth=1.5, linestyle="--",
            label=f"SPY B&H pre-tax → ${final_spy_pre:,.0f}")
    ax.plot(years, spy_post_eq, color="#38bdf8", linewidth=2.0, marker="s", markersize=5,
            label=f"SPY B&H AFTER-TAX (15% LTCG at end) → ${final_spy_post:,.0f}")
    ax.set_yscale("log")
    ax.set_xlabel("Year", color="#d1d5db")
    ax.set_ylabel("Equity ($), log scale", color="#d1d5db")
    ax.set_title(
        f"After-tax equity curves: Strategy (Pure ML, no FVG/regime gate) vs SPY B&H\n"
        f"$1,000 starting capital, {years[0]+1}–{years[-1]} walk-forward",
        color="#f3f4f6", fontsize=12, pad=10,
    )
    ax.grid(True, which="both", alpha=0.15, color="#9ca3af")
    ax.tick_params(colors="#d1d5db")
    for spine in ax.spines.values():
        spine.set_color("#374151")
    ax.legend(loc="upper left", facecolor="#1f2937", edgecolor="#374151",
              labelcolor="#e5e7eb", framealpha=0.9)

    # Lower panel: ratio
    ax2 = axes[1]
    ax2.set_facecolor("#0f1015")
    ratio = [s / sp for s, sp in zip(strat_eq, spy_post_eq, strict=True)]
    ax2.plot(years, ratio, color="#f59e0b", linewidth=2.0, marker="o", markersize=4)
    ax2.axhline(y=1.0, color="#9ca3af", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Year", color="#d1d5db")
    ax2.set_ylabel("Strategy / SPY (after-tax)", color="#d1d5db")
    ax2.set_title("Outperformance ratio over time", color="#f3f4f6", fontsize=11)
    ax2.grid(True, alpha=0.15, color="#9ca3af")
    ax2.tick_params(colors="#d1d5db")
    for spine in ax2.spines.values():
        spine.set_color("#374151")
    ax2.set_ylim(bottom=0.5)

    plt.tight_layout()
    Path(OUT_PNG).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=120, bbox_inches="tight", facecolor="#0f1015")
    print(f"chart: {OUT_PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Silence unused-import linter.
_ = math
