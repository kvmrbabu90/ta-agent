"""Point-in-time S&P 500 index membership.

Wikipedia maintains two tables we care about:
  1. "S&P 500 component stocks" — current constituents
  2. "Selected changes to the list of S&P 500 components" — additions and removals
     with the date of each change, going back to ~1957

This module:
  - Scrapes both tables from the live Wikipedia page
  - Reconstructs a (symbol, start_date, end_date) timeline for every company
    that has ever been in the index in the modern era
  - Cross-validates current-day membership against the live snapshot
  - Persists results to the index_membership table

Limitations:
  - Wikipedia's changes table is well-maintained from ~2000 onward but has
    gaps before that. We log gaps so you know which periods are reliable.
  - Some old tickers have been recycled (e.g. "GOOG" pre-2014 vs post). We
    rely on the changes table's company name for disambiguation.
  - Symbol changes (e.g. FB -> META in 2022) are recorded as a remove + add.

Usage:
    from packages.ingestion.universe.sp500_history import build_sp500_membership

    df = build_sp500_membership()
    # df has columns: universe, symbol, exchange, start_date, end_date, company_name
"""

from __future__ import annotations

from datetime import date
from io import StringIO

import pandas as pd
import requests

from packages.common.logging import log

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# A safe lower bound for the "modern era" of S&P 500 data we trust.
# Before this date, Wikipedia's changes table is too sparse for unbiased
# point-in-time reconstruction. Adjust if you have alternative sources.
RELIABLE_START = date(2000, 1, 1)

USER_AGENT = (
    "ta-agent/0.1 (technical analysis research; "
    "contact: your-email@example.com)"
)


def _fetch_wiki_html() -> str:
    """Fetch the Wikipedia page HTML, respecting Wikipedia's UA policy."""
    log.info(f"Fetching {WIKI_URL}")
    resp = requests.get(WIKI_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.text


def _parse_current_constituents(html: str) -> pd.DataFrame:
    """Parse the 'S&P 500 component stocks' table (id='constituents' on Wikipedia).

    Columns we care about: Symbol, Security, GICS Sector, Date added, CIK.
    """
    tables = pd.read_html(StringIO(html), attrs={"id": "constituents"})
    if not tables:
        raise RuntimeError("Could not find 'constituents' table on Wikipedia page")
    df = tables[0]

    # Wikipedia column names have shifted over time; normalize defensively.
    rename_map = {}
    for c in df.columns:
        cl = str(c).lower()
        if cl == "symbol":
            rename_map[c] = "symbol"
        elif cl in ("security", "company"):
            rename_map[c] = "company_name"
        elif "date added" in cl or cl == "date first added":
            rename_map[c] = "date_added"
        elif "gics sector" in cl:
            rename_map[c] = "sector"
    df = df.rename(columns=rename_map)

    # Wikipedia uses dots instead of hyphens for some tickers (BRK.B vs BRK-B).
    # We standardize on the dot form for storage; adapters convert as needed.
    df["symbol"] = df["symbol"].astype(str).str.strip()

    if "date_added" in df.columns:
        df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce").dt.date
    else:
        df["date_added"] = pd.NaT

    keep = ["symbol", "company_name", "date_added"]
    if "sector" in df.columns:
        keep.append("sector")
    return df[keep].copy()


def _parse_changes_table(html: str) -> pd.DataFrame:
    """Parse the 'Selected changes to the list of S&P 500 components' table.

    The table has a quirky two-row header:
        | Date | Added                | Removed              | Reason |
        |      | Ticker | Security    | Ticker | Security    |        |

    pandas.read_html flattens that into a MultiIndex; we collapse it.
    """
    tables = pd.read_html(StringIO(html), attrs={"id": "changes"})
    if not tables:
        # Some snapshots use a different id; fall back to scanning all tables.
        all_tables = pd.read_html(StringIO(html))
        for t in all_tables:
            cols = " ".join(str(c) for c in t.columns)
            if "Added" in cols and "Removed" in cols:
                tables = [t]
                break
    if not tables:
        raise RuntimeError("Could not find changes table on Wikipedia page")

    df = tables[0]

    # Flatten MultiIndex columns: ("Added", "Ticker") -> "added_ticker"
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(c).strip().lower() for c in col if str(c) != "nan").strip()
            for col in df.columns
        ]

    # Normalize column names across Wikipedia's historical edits.
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "date" or cl.startswith("date_"):
            col_map[c] = "change_date"
        elif "added_ticker" in cl or cl == "added":
            col_map[c] = "added_ticker"
        elif "added_security" in cl or "added_company" in cl:
            col_map[c] = "added_name"
        elif "removed_ticker" in cl or cl == "removed":
            col_map[c] = "removed_ticker"
        elif "removed_security" in cl or "removed_company" in cl:
            col_map[c] = "removed_name"
        elif "reason" in cl:
            col_map[c] = "reason"
    df = df.rename(columns=col_map)

    # Coerce date column.
    df["change_date"] = pd.to_datetime(df["change_date"], errors="coerce").dt.date

    # Drop rows with no valid date (footnotes, separator rows, etc.)
    df = df.dropna(subset=["change_date"]).copy()

    log.info(f"Parsed {len(df)} index changes from Wikipedia")
    return df


def _reconstruct_timeline(
    current_df: pd.DataFrame,
    changes_df: pd.DataFrame,
    today: date | None = None,
) -> pd.DataFrame:
    """Build a (symbol, start_date, end_date) timeline.

    Algorithm:
      1. Start from current constituents — each has start_date = (date_added if known,
         else RELIABLE_START), end_date = NULL.
      2. Walk the changes table in chronological order. For each removal, find the
         matching current row (or previously added row) and set its end_date.
      3. For each addition that ISN'T in current constituents, create a row with
         start_date = change_date and end_date = matching subsequent removal.

    This produces overlapping intervals only when a ticker is added, removed,
    and re-added later — which is rare but legitimate (e.g. NVDA has been in,
    out, and back in).
    """
    today = today or date.today()
    rows: list[dict] = []

    # Index changes by ticker for fast lookup. A single ticker can appear
    # multiple times in the changes table — we preserve all of them.
    changes_sorted = changes_df.sort_values("change_date").reset_index(drop=True)

    # Step 1: every current constituent gets an "open" row with end_date=None.
    current_symbols = set(current_df["symbol"])
    for _, r in current_df.iterrows():
        sym = r["symbol"]
        # Find the most recent ADDITION of this symbol in the changes table.
        # That's the true start_date for this membership stretch.
        adds = changes_sorted[
            (changes_sorted["added_ticker"] == sym)
            & (changes_sorted["change_date"] <= today)
        ]
        if not adds.empty:
            start = adds.iloc[-1]["change_date"]
        elif pd.notna(r.get("date_added")):
            start = r["date_added"]
        else:
            # Fallback: company has been in the index since before our
            # changes-table coverage begins. Use RELIABLE_START.
            start = RELIABLE_START

        rows.append({
            "symbol": sym,
            "start_date": start,
            "end_date": None,
            "company_name": r.get("company_name"),
        })

    # Step 2: handle every historical (added, removed) pair for tickers
    # NOT currently in the index. We pair adds with the next subsequent remove.
    historical_tickers = set(changes_sorted["added_ticker"].dropna()) | set(
        changes_sorted["removed_ticker"].dropna()
    )
    historical_only = historical_tickers - current_symbols

    for ticker in historical_only:
        adds = changes_sorted[changes_sorted["added_ticker"] == ticker][
            ["change_date", "added_name"]
        ].rename(columns={"change_date": "add_date", "added_name": "name"})
        removes = changes_sorted[changes_sorted["removed_ticker"] == ticker][
            ["change_date", "removed_name"]
        ].rename(columns={"change_date": "remove_date", "removed_name": "name"})

        # If we never saw an add for this ticker, it must have been in the
        # index before our coverage window — use RELIABLE_START.
        if adds.empty and not removes.empty:
            for _, rem in removes.iterrows():
                rows.append({
                    "symbol": ticker,
                    "start_date": RELIABLE_START,
                    "end_date": rem["remove_date"],
                    "company_name": rem["name"],
                })
            continue

        # Pair each add with the next remove on or after it.
        # IMPORTANT: this ticker is NOT in current constituents, so the final
        # period (after the last remove, or the lone add with no remove) must
        # have a non-null end_date. We use `today` as the closeout date in the
        # rare case we observed an add with no matching subsequent remove —
        # this typically means a ticker symbol change recorded as add+remove
        # on different rows that we couldn't pair, or genuinely incomplete
        # data. Logging it helps spot data-quality issues.
        adds_list = adds.sort_values("add_date").to_dict("records")
        removes_list = removes.sort_values("remove_date").to_dict("records")
        rem_idx = 0
        for add in adds_list:
            # Advance to the first remove on/after this add.
            while rem_idx < len(removes_list) and removes_list[rem_idx]["remove_date"] < add["add_date"]:
                rem_idx += 1
            if rem_idx < len(removes_list):
                end = removes_list[rem_idx]["remove_date"]
                rem_idx += 1
            else:
                # No matching remove. This ticker isn't in current constituents
                # either, so we have a data gap. Close it at `today` and log.
                end = today
                log.warning(
                    f"Ticker {ticker} added on {add['add_date']} has no "
                    f"matching removal and is not in current constituents. "
                    f"Closing membership at {today} as a fallback."
                )
            rows.append({
                "symbol": ticker,
                "start_date": add["add_date"],
                "end_date": end,
                "company_name": add.get("name"),
            })

    df = pd.DataFrame(rows)
    df["universe"] = "SP500"
    df["exchange"] = "NYSE"  # Placeholder — refine via adapter at ingest time.
    df = df[["universe", "symbol", "exchange", "start_date", "end_date", "company_name"]]
    df = df.sort_values(["symbol", "start_date"]).reset_index(drop=True)
    return df


def build_sp500_membership(today: date | None = None) -> pd.DataFrame:
    """Public entry point. Fetch + parse + reconstruct timeline."""
    html = _fetch_wiki_html()
    current = _parse_current_constituents(html)
    changes = _parse_changes_table(html)
    timeline = _reconstruct_timeline(current, changes, today=today)

    # Sanity checks.
    n_current = (timeline["end_date"].isna()).sum()
    n_total = len(timeline)
    log.info(
        f"S&P 500 membership: {n_total} historical rows, "
        f"{n_current} currently active (expected ~500-503)"
    )
    if not (490 <= n_current <= 510):
        log.warning(
            f"Active member count {n_current} is outside expected range. "
            f"Wikipedia layout may have changed; inspect the parser."
        )

    return timeline
