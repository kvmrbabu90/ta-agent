"""Macro features broadcast to (symbol, bar_date) panel.

Reads from the ``macro_daily`` table. If no macro data is present, the
feature group should not be registered (see pipeline conditional logic);
when it is registered with empty data, all features come back NaN.

Features (v1):
    macro__vix_level_z_252  — z-score of VIX close vs. trailing 252-day window
                              (excluding today, so causal)
    macro__vix_chg_5d       — 5-day change in VIX (absolute, not %)
    macro__fx_ret_5d        — 5-day log return of USD/INR

Step-4 additions (regime / cross-asset):
    macro__treasury_10y_z_252        — z-score of 10y yield
    macro__treasury_10y_chg_5d       — 5-day change in 10y yield
    macro__yield_curve_slope_z_252   — z-score of (10y - 5y) yield spread
    macro__credit_spread_z_252       — z-score of LQD/HYG ratio (proxy for credit spread:
                                       higher = wider spread = risk-off)
    macro__credit_spread_chg_5d      — 5-day change in LQD/HYG ratio
    macro__dxy_chg_5d                — 5-day log return of US Dollar Index
    macro__gold_copper_ratio_z_252   — z-score of gold/copper ratio (deflation/risk-off proxy)

Each is computed once per date and broadcast to every (symbol, date) row.

NOTE: a controlled CV after Step-4 added these 7 features showed a -38%
rank-IC regression (seed 42) and -74% on seed 43. Universe-constant features
have zero per-day cross-sectional discrimination; the only way they help is
via interactions with per-symbol features, which our current LightGBM
hyperparams don't reliably learn. The features are still computed (they're
cheap and may help a future regime-conditional model), but their net effect
on flat training is negative. Step 5 is exploring the regime-gating path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from packages.features.base import PanelFeatureGroup
from packages.features.extensions import FeatureExtension, register_extension
from packages.ingestion.macro import has_macro_data, load_macro_series


def _trailing_zscore(s: pd.Series, window: int) -> pd.Series:
    """Causal z-score: exclude the current observation when computing
    the rolling mean and std so today's value isn't in its own baseline."""
    prior = s.shift(1)
    mean = prior.rolling(window=window, min_periods=max(20, window // 4)).mean()
    std = prior.rolling(window=window, min_periods=max(20, window // 4)).std()
    return (s - mean) / std.replace(0.0, np.nan)


class MacroFeatures(PanelFeatureGroup):
    name = "macro"

    def __init__(self, duckdb_path: str | None = None) -> None:
        self._duckdb_path = duckdb_path

    def compute(self, panel: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(panel)
        out = panel[["symbol", "bar_date"]].copy()

        if panel.empty:
            for col in ("vix_level_z_252", "vix_chg_5d", "fx_ret_5d"):
                out[f"{self.name}__{col}"] = pd.NA
            return out

        # Window we need from macro_daily: full panel range plus 252-day warmup.
        bar_dates = pd.to_datetime(panel["bar_date"]).dt.date
        start = bar_dates.min() - pd.Timedelta(days=400)
        end = bar_dates.max()

        # Load all series; missing series come back as empty Series.
        load = lambda name: load_macro_series(name, start, end, duckdb_path=self._duckdb_path)  # noqa: E731
        vix = load("vix")
        fx = load("usd_inr")
        t10 = load("treasury_10y")
        t5 = load("treasury_2y")  # 5-year proxy (see macro.SERIES_TICKERS)
        hyg = load("hyg")
        lqd = load("lqd")
        dxy = load("dxy")
        gold = load("gold")
        copper = load("copper")
        # Phase A2: VIX term structure
        vix9d = load("vix9d")
        vix3m = load("vix3m")
        vvix = load("vvix")
        skew = load("skew")

        # Union of all dates we have any macro on.
        all_idx = sorted(set().union(*(s.index for s in [
            vix, fx, t10, t5, hyg, lqd, dxy, gold, copper,
            vix9d, vix3m, vvix, skew,
        ])))
        idx = pd.DatetimeIndex(all_idx)
        macro = pd.DataFrame(index=idx)

        # v1 features
        if not vix.empty:
            v = vix.reindex(idx).ffill()
            macro["vix_level_z_252"] = _trailing_zscore(v, 252).values
            macro["vix_chg_5d"] = v.diff(5).values
        else:
            macro["vix_level_z_252"] = np.nan
            macro["vix_chg_5d"] = np.nan
        if not fx.empty:
            log_fx = np.log(fx.reindex(idx).ffill().where(lambda s: s > 0))
            macro["fx_ret_5d"] = log_fx.diff(5).values
        else:
            macro["fx_ret_5d"] = np.nan

        # Step-4 additions
        if not t10.empty:
            t = t10.reindex(idx).ffill()
            macro["treasury_10y_z_252"] = _trailing_zscore(t, 252).values
            macro["treasury_10y_chg_5d"] = t.diff(5).values
        else:
            macro["treasury_10y_z_252"] = np.nan
            macro["treasury_10y_chg_5d"] = np.nan

        if not t10.empty and not t5.empty:
            slope = t10.reindex(idx).ffill() - t5.reindex(idx).ffill()
            macro["yield_curve_slope_z_252"] = _trailing_zscore(slope, 252).values
        else:
            macro["yield_curve_slope_z_252"] = np.nan

        if not hyg.empty and not lqd.empty:
            # Higher LQD/HYG ratio = HYG underperforming LQD = wider credit
            # spreads = risk-off. Forward-fill within asset before ratio.
            hyg_f = hyg.reindex(idx).ffill().where(lambda s: s > 0)
            lqd_f = lqd.reindex(idx).ffill().where(lambda s: s > 0)
            credit = lqd_f / hyg_f
            macro["credit_spread_z_252"] = _trailing_zscore(credit, 252).values
            macro["credit_spread_chg_5d"] = credit.diff(5).values
        else:
            macro["credit_spread_z_252"] = np.nan
            macro["credit_spread_chg_5d"] = np.nan

        if not dxy.empty:
            log_dxy = np.log(dxy.reindex(idx).ffill().where(lambda s: s > 0))
            macro["dxy_chg_5d"] = log_dxy.diff(5).values
        else:
            macro["dxy_chg_5d"] = np.nan

        if not gold.empty and not copper.empty:
            gold_f = gold.reindex(idx).ffill().where(lambda s: s > 0)
            copper_f = copper.reindex(idx).ffill().where(lambda s: s > 0)
            ratio = gold_f / copper_f
            macro["gold_copper_ratio_z_252"] = _trailing_zscore(ratio, 252).values
        else:
            macro["gold_copper_ratio_z_252"] = np.nan

        # Phase A2: VIX term-structure features
        # vix9d/vix ratio < 1 typically = stress is concentrated in near-term;
        # vix9d/vix > 1 = unusual short-end stress (event risk)
        if not vix9d.empty and not vix.empty:
            v9 = vix9d.reindex(idx).ffill()
            v30 = vix.reindex(idx).ffill().where(lambda s: s > 0)
            macro["vix9d_to_vix_ratio"] = (v9 / v30).values
        else:
            macro["vix9d_to_vix_ratio"] = np.nan

        # vix/vix3m: < 1 = contango (calm regime); > 1 = backwardation (stress regime)
        if not vix.empty and not vix3m.empty:
            v30 = vix.reindex(idx).ffill()
            v3m = vix3m.reindex(idx).ffill().where(lambda s: s > 0)
            macro["vix_to_vix3m_ratio"] = (v30 / v3m).values
        else:
            macro["vix_to_vix3m_ratio"] = np.nan

        # VVIX z-score: vol-of-vol level vs trailing 252d
        if not vvix.empty:
            vv = vvix.reindex(idx).ffill()
            macro["vvix_z_252"] = _trailing_zscore(vv, 252).values
        else:
            macro["vvix_z_252"] = np.nan

        # SKEW level (raw — typically 100-150) and 5-day change
        if not skew.empty:
            sk = skew.reindex(idx).ffill()
            macro["skew_level"] = sk.values
            macro["skew_chg_5d"] = sk.diff(5).values
        else:
            macro["skew_level"] = np.nan
            macro["skew_chg_5d"] = np.nan

        macro = macro.reset_index().rename(columns={"index": "bar_date"})
        # Forward-fill macro values across stock-holiday gaps before merge.
        macro = macro.sort_values("bar_date").ffill()

        # Coerce both sides to datetime64 to dodge merge dtype mismatches when
        # the panel uses datetime64[us] but macro_daily was stored as DATE.
        macro["bar_date"] = pd.to_datetime(macro["bar_date"])
        out_join = out.copy()
        out_join["bar_date"] = pd.to_datetime(out_join["bar_date"])

        merged = out_join.merge(macro, on="bar_date", how="left")
        merged["bar_date"] = out["bar_date"].values  # restore original dtype
        feature_cols = (
            "vix_level_z_252", "vix_chg_5d", "fx_ret_5d",
            "treasury_10y_z_252", "treasury_10y_chg_5d",
            "yield_curve_slope_z_252",
            "credit_spread_z_252", "credit_spread_chg_5d",
            "dxy_chg_5d",
            "gold_copper_ratio_z_252",
            # Phase A2: VIX term structure
            "vix9d_to_vix_ratio", "vix_to_vix3m_ratio", "vvix_z_252",
            "skew_level", "skew_chg_5d",
        )
        rename = {c: f"{self.name}__{c}" for c in feature_cols}
        return merged.rename(columns=rename)


# --- Extension registration -----------------------------------------------
# Picked up automatically by pipeline._resolve_panel_groups via the extension
# registry. v2 (earnings, news_sentiment) will register the same way.


class _MacroExtension(FeatureExtension):
    name = "macro"
    kind = "panel"

    def is_available(self, *, duckdb_path: str | None = None) -> bool:
        return has_macro_data(duckdb_path=duckdb_path)

    def make_group(self, *, duckdb_path: str | None = None) -> MacroFeatures:
        return MacroFeatures(duckdb_path=duckdb_path)


register_extension(_MacroExtension())
