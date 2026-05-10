"""Sector-residualized features — stock returns minus their sector ETF's return.

Why this beats sector-rank (Step 1, today):
    The earlier sector_relative attempt RANKED stocks within their sector.
    That mostly duplicated the universe-wide cross-sectional rank we already
    have, and the tiny per-sector group sizes (~45 stocks) made the rank
    noisier than the 503-stock universe-wide rank.

    Residualizing instead — subtracting the sector ETF's return from the
    stock's return — extracts the *idiosyncratic* component that's
    orthogonal to sector beta. This is how every long-short equity hedge
    fund actually decomposes signal.

Data dependencies:
    - data/processed/sectors_{universe}.parquet  (symbol -> GICS sector)
      produced by scripts.refresh_sectors
    - ohlcv_daily rows for the 11 SPDR sector ETFs, populated by
      scripts.refresh_sector_etfs

Features (5):
    sector_residual__resid_log_return_5d        stock 5d log-return minus sector-ETF 5d
    sector_residual__resid_log_return_20d       same, 20d
    sector_residual__resid_realized_vol_20d     stock 20d realized vol minus sector-ETF
                                                20d realized vol (positive = stock is
                                                idiosyncratically more vol than sector)
    sector_residual__resid_rsi_14_minus_sector  stock RSI(14) minus sector ETF RSI(14)
    sector_residual__resid_zscore_60d           60d trailing z-score of resid_log_return_5d
                                                (against the same stock's own history —
                                                "is today's residual unusual for THIS stock?")

When sector data or sector-ETF data is missing for a symbol, all features are NaN.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from packages.common.config import PROCESSED_DIR
from packages.common.logging import log
from packages.features.base import PanelFeatureGroup
from packages.features.extensions import FeatureExtension, register_extension
from packages.ingestion.storage import get_conn, get_ohlcv

# GICS sector name -> ticker for SPDR Select Sector ETF
_SECTOR_TO_ETF: dict[str, str] = {
    "Technology":             "XLK",
    "Financial Services":     "XLF",   # newer GICS naming
    "Financials":             "XLF",
    "Energy":                 "XLE",
    "Healthcare":             "XLV",
    "Health Care":            "XLV",
    "Consumer Cyclical":      "XLY",
    "Consumer Discretionary": "XLY",
    "Consumer Defensive":     "XLP",
    "Consumer Staples":       "XLP",
    "Industrials":            "XLI",
    "Utilities":              "XLU",
    "Real Estate":            "XLRE",
    "Basic Materials":        "XLB",
    "Materials":              "XLB",
    "Communication Services": "XLC",
}

_FEATURE_COLS = (
    "resid_log_return_5d",
    "resid_log_return_20d",
    "resid_realized_vol_20d",
    "resid_rsi_14_minus_sector",
    "resid_zscore_60d",
)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI on a close series, returning a series the same shape as input."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def _load_sector_map(universe: str) -> pd.DataFrame:
    path = Path(PROCESSED_DIR) / f"sectors_{universe.lower()}.parquet"
    if not path.exists():
        log.warning(
            f"sector_residual: no sector map at {path}; "
            "all symbols will be treated as UNKNOWN sector"
        )
        return pd.DataFrame(columns=["symbol", "sector"])
    return pd.read_parquet(path, columns=["symbol", "sector"])


def _load_sector_etf_features(
    etf_tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    duckdb_path: str | None = None,
) -> pd.DataFrame:
    """Return a long DataFrame with derived ETF features per (etf, bar_date)."""
    if not etf_tickers:
        return pd.DataFrame(
            columns=["etf", "bar_date", "etf_log_return_5d", "etf_log_return_20d",
                     "etf_realized_vol_20d", "etf_rsi_14"]
        )

    with get_conn(duckdb_path) as conn:
        raw = get_ohlcv(
            etf_tickers,
            start=start.date(),
            end=end.date(),
            conn=conn,
        )
    if raw.empty:
        log.warning(f"sector_residual: no OHLCV rows found for ETFs {etf_tickers}")
        return pd.DataFrame(
            columns=["etf", "bar_date", "etf_log_return_5d", "etf_log_return_20d",
                     "etf_realized_vol_20d", "etf_rsi_14"]
        )

    pieces: list[pd.DataFrame] = []
    for etf, grp in raw.groupby("symbol"):
        grp = grp.sort_values("bar_date").reset_index(drop=True)
        log_close = np.log(grp["close"].astype(float).where(lambda s: s > 0))
        log_ret_1d = log_close.diff()
        out = pd.DataFrame({
            "etf": etf,
            "bar_date": grp["bar_date"].values,
            "etf_log_return_5d": log_close.diff(5).values,
            "etf_log_return_20d": log_close.diff(20).values,
            "etf_realized_vol_20d": log_ret_1d.rolling(20, min_periods=20).std().values,
            "etf_rsi_14": _rsi(grp["close"], 14).values,
        })
        pieces.append(out)
    return pd.concat(pieces, axis=0, ignore_index=True)


class SectorResidualFeatures(PanelFeatureGroup):
    """Subtracts sector ETF behaviour from per-stock features (residualization)."""

    name = "sector_residual"

    def __init__(
        self,
        universe: str = "SP500",
        *,
        duckdb_path: str | None = None,
    ) -> None:
        self._universe = universe
        self._duckdb_path = duckdb_path

    def compute(self, panel: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(panel)
        out = panel[["symbol", "bar_date"]].copy()

        if panel.empty:
            for c in _FEATURE_COLS:
                out[f"{self.name}__{c}"] = pd.NA
            return out

        sector_map = _load_sector_map(self._universe)
        if sector_map.empty:
            for c in _FEATURE_COLS:
                out[f"{self.name}__{c}"] = pd.NA
            return out

        # Map each symbol to a sector ETF ticker; drop symbols with no mapping.
        sym_to_etf = (
            sector_map.assign(etf=sector_map["sector"].map(_SECTOR_TO_ETF))
                      .dropna(subset=["etf"])
                      .set_index("symbol")["etf"]
                      .to_dict()
        )

        # Load ETF features once for all sector ETFs touched in the panel.
        relevant_etfs = sorted({sym_to_etf[s] for s in panel["symbol"].unique() if s in sym_to_etf})
        date_min = pd.to_datetime(panel["bar_date"]).min()
        date_max = pd.to_datetime(panel["bar_date"]).max()
        # Pad the window so 20d return / vol windows have full warmup.
        etf_features = _load_sector_etf_features(
            relevant_etfs,
            start=date_min - timedelta(days=60),
            end=date_max,
            duckdb_path=self._duckdb_path,
        )

        if etf_features.empty:
            for c in _FEATURE_COLS:
                out[f"{self.name}__{c}"] = pd.NA
            return out

        # Bring per-symbol inputs onto the work frame. We need:
        #   price__log_return_5d, price__log_return_20d, price__realized_vol_20d,
        #   momentum__rsi_14
        needed_inputs = [
            "price__log_return_5d", "price__log_return_20d",
            "price__realized_vol_20d", "momentum__rsi_14",
        ]
        missing = [c for c in needed_inputs if c not in panel.columns]
        if missing:
            log.warning(f"sector_residual: panel missing inputs {missing}; emitting NaN")
            for c in _FEATURE_COLS:
                out[f"{self.name}__{c}"] = pd.NA
            return out

        # Wide work frame: panel rows + the etf assignment + etf features.
        work = panel[["symbol", "bar_date", *needed_inputs]].copy()
        work["etf"] = work["symbol"].map(sym_to_etf)
        work["bar_date_dt"] = pd.to_datetime(work["bar_date"])

        etf_features = etf_features.copy()
        etf_features["bar_date_dt"] = pd.to_datetime(etf_features["bar_date"])
        joined = work.merge(
            etf_features[["etf", "bar_date_dt", "etf_log_return_5d",
                          "etf_log_return_20d", "etf_realized_vol_20d", "etf_rsi_14"]],
            on=["etf", "bar_date_dt"], how="left",
        )

        # Residual computations.
        joined[f"{self.name}__resid_log_return_5d"] = (
            joined["price__log_return_5d"] - joined["etf_log_return_5d"]
        )
        joined[f"{self.name}__resid_log_return_20d"] = (
            joined["price__log_return_20d"] - joined["etf_log_return_20d"]
        )
        joined[f"{self.name}__resid_realized_vol_20d"] = (
            joined["price__realized_vol_20d"] - joined["etf_realized_vol_20d"]
        )
        joined[f"{self.name}__resid_rsi_14_minus_sector"] = (
            joined["momentum__rsi_14"] - joined["etf_rsi_14"]
        )

        # 60d trailing z-score of resid_log_return_5d, per symbol.
        # Causal: shift(1) before rolling to exclude the current row.
        resid_5d = joined[f"{self.name}__resid_log_return_5d"]
        joined["__resid_5d_shifted"] = joined.groupby("symbol", observed=True, sort=False)[
            f"{self.name}__resid_log_return_5d"
        ].shift(1)
        roll = joined.groupby("symbol", observed=True, sort=False)["__resid_5d_shifted"].rolling(
            60, min_periods=20
        )
        joined["__roll_mean"] = roll.mean().reset_index(level=0, drop=True)
        joined["__roll_std"] = roll.std().reset_index(level=0, drop=True)
        joined[f"{self.name}__resid_zscore_60d"] = (
            (resid_5d - joined["__roll_mean"]) / joined["__roll_std"].replace(0.0, np.nan)
        )

        # Assemble output keyed by (symbol, bar_date) preserving panel order.
        feature_cols = [f"{self.name}__{c}" for c in _FEATURE_COLS]
        out = out.merge(
            joined[["symbol", "bar_date", *feature_cols]],
            on=["symbol", "bar_date"], how="left",
        )
        return out


# --- Extension registration --------------------------------------------------


class _SectorResidualExtension(FeatureExtension):
    """Registers when both the sector mapping parquet AND at least one sector
    ETF row is in ohlcv_daily — otherwise the residuals would all be NaN."""

    name = "sector_residual"
    kind = "panel"

    def is_available(self, *, duckdb_path: str | None = None) -> bool:
        # Sector map present?
        if not (Path(PROCESSED_DIR) / "sectors_sp500.parquet").exists():
            return False
        # At least one ETF in OHLCV?
        try:
            with get_conn(duckdb_path) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM ohlcv_daily WHERE symbol IN ('XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLU','XLRE','XLB','XLC')"
                ).fetchone()
                return bool(row and row[0] > 0)
        except Exception:  # noqa: BLE001
            return False

    def make_group(
        self, *, duckdb_path: str | None = None
    ) -> SectorResidualFeatures:
        return SectorResidualFeatures(universe="SP500", duckdb_path=duckdb_path)


register_extension(_SectorResidualExtension())
