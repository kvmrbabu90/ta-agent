"""Point-in-time NIFTY 100 index membership.

This is significantly harder than the S&P 500 case because:
  - NSE does not publish a clean machine-readable history of constituents.
  - niftyindices.com publishes only the current snapshot in CSV form.
  - Reconstitutions happen semi-annually (March, September) and are announced
    in PDF press releases.

Practical strategy (to be implemented across phases):

  Phase A (initial): seed with current NIFTY 100 constituents from the
    niftyindices.com CSV. Every symbol gets start_date=2014-01-01 and
    end_date=NULL. This DOES introduce survivorship bias for older training
    data, but is a starting point.

  Phase B: parse the semi-annual reconstitution PDFs from NSE archives
    (https://www.niftyindices.com/Daily_Snapshot) to reconstruct historical
    additions/removals. This requires PDF parsing and is best done as a
    one-time manual + scripted effort.

  Phase C: cross-reference with NSE's bhavcopy archives to detect when
    historical symbols stopped trading (delistings, mergers).

For v1 we ship Phase A and clearly mark training periods before our reliable
start as biased. That keeps us honest.
"""

from __future__ import annotations

from datetime import date
from io import StringIO

import pandas as pd
import requests

from packages.common.logging import log

# Live current-constituents CSV from niftyindices.com
NIFTY100_CSV_URL = (
    "https://niftyindices.com/IndexConstituent/ind_nifty100list.csv"
)

# We trust point-in-time membership from this date forward only.
# Before this date we use the current snapshot as a proxy and accept
# the survivorship bias (or skip training before this date).
RELIABLE_START = date(2014, 1, 1)

USER_AGENT = "ta-agent/0.1 (research)"


def _fetch_current_csv() -> pd.DataFrame:
    log.info(f"Fetching {NIFTY100_CSV_URL}")
    resp = requests.get(NIFTY100_CSV_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))

    # The CSV typically has columns: Company Name, Industry, Symbol, Series,
    # ISIN Code. We standardize.
    rename = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "symbol":
            rename[c] = "symbol"
        elif "company" in cl:
            rename[c] = "company_name"
        elif "industry" in cl or "sector" in cl:
            rename[c] = "sector"
    df = df.rename(columns=rename)
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    return df[[c for c in ["symbol", "company_name", "sector"] if c in df.columns]]


def build_nifty100_membership(today: date | None = None) -> pd.DataFrame:
    """Phase A implementation: current constituents only.

    This deliberately does NOT attempt historical reconstruction yet.
    Models trained against this universe before the Phase B work is done
    have survivorship bias for any backtest period before today.
    """
    today = today or date.today()
    current = _fetch_current_csv()

    rows = []
    for _, r in current.iterrows():
        rows.append({
            "universe": "NIFTY100",
            "symbol": r["symbol"],
            "exchange": "NSE",
            "start_date": RELIABLE_START,
            "end_date": None,
            "company_name": r.get("company_name"),
        })

    df = pd.DataFrame(rows)
    log.info(
        f"NIFTY 100 membership: {len(df)} current constituents loaded "
        f"(Phase A: survivorship-biased for periods before {RELIABLE_START})"
    )
    log.warning(
        "NIFTY 100 historical reconstitutions are not yet implemented. "
        "Backtests on this universe should be interpreted with caution. "
        "See packages/ingestion/universe/nifty100_history.py for plan."
    )
    return df
