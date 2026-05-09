"""Unit tests for InteractionFeatures (per-symbol × macro products)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from packages.features.interactions import InteractionFeatures


def _toy_panel_with_macro() -> pd.DataFrame:
    """Two stocks, two dates, with the per-symbol + macro inputs the
    interaction group depends on already populated."""
    rows = []
    for sym, base_ret, base_rsi, base_macd in (("A", 0.05, 70.0, 0.5), ("B", -0.02, 30.0, -0.3)):
        for d, vix_z, credit_z, yc_z, dxy, t10 in (
            (date(2025, 1, 6),  +1.0, +0.5, -0.2, +0.001, +0.05),
            (date(2025, 1, 7),  -0.5, -0.8, +1.0, -0.002, -0.10),
        ):
            rows.append({
                "symbol": sym, "bar_date": d, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1,
                "price__log_return_5d": base_ret,
                "price__log_return_20d": base_ret * 4,
                "price__realized_vol_20d": 0.02,
                "momentum__rsi_14": base_rsi,
                "trend__macd_histogram": base_macd,
                "macro__vix_level_z_252": vix_z,
                "macro__credit_spread_z_252": credit_z,
                "macro__yield_curve_slope_z_252": yc_z,
                "macro__dxy_chg_5d": dxy,
                "macro__treasury_10y_chg_5d": t10,
            })
    return pd.DataFrame(rows)


def test_interactions_emit_8_features() -> None:
    panel = _toy_panel_with_macro()
    out = InteractionFeatures().compute(panel)
    cols = [c for c in out.columns if c.startswith("interactions__")]
    assert len(cols) == 8


def test_interactions_compute_correct_products() -> None:
    panel = _toy_panel_with_macro()
    out = InteractionFeatures().compute(panel)
    out = out.merge(panel[["symbol", "bar_date", "momentum__rsi_14", "macro__vix_level_z_252",
                            "price__log_return_5d", "macro__credit_spread_z_252"]],
                    on=["symbol", "bar_date"])
    # Spot-check a few products against hand calculation
    row_a_d1 = out[(out["symbol"] == "A") & (out["bar_date"] == date(2025, 1, 6))].iloc[0]
    assert row_a_d1["interactions__rsi14_x_vix_z"] == np.float64(70.0 * 1.0)
    row_b_d2 = out[(out["symbol"] == "B") & (out["bar_date"] == date(2025, 1, 7))].iloc[0]
    assert row_b_d2["interactions__rsi14_x_vix_z"] == np.float64(30.0 * -0.5)


def test_interactions_missing_input_columns_yield_na() -> None:
    """If a per-symbol or macro input column is absent (e.g. macro extension
    not loaded), the corresponding interaction is NaN — not a crash."""
    panel = _toy_panel_with_macro().drop(columns=["macro__vix_level_z_252",
                                                  "trend__macd_histogram"])
    out = InteractionFeatures().compute(panel)
    # Two interactions depend on vix_z, one depends on macd_hist → all NaN here
    for c in ("interactions__rsi14_x_vix_z",
              "interactions__log_ret_5d_x_vix_z",
              "interactions__realized_vol_x_vix_z",
              "interactions__macd_hist_x_dxy_chg_5d"):
        assert out[c].isna().all(), c
    # Other interactions whose inputs are present should still compute.
    assert out["interactions__log_ret_20d_x_credit_z"].notna().any()


def test_interactions_nan_propagates_naturally() -> None:
    """If one input is NaN at a row, the interaction at that row is NaN."""
    panel = _toy_panel_with_macro()
    panel.loc[0, "macro__vix_level_z_252"] = np.nan  # nuke vix_z for first row
    out = InteractionFeatures().compute(panel)
    # Row 0's vix_z products should be NaN
    assert pd.isna(out.iloc[0]["interactions__rsi14_x_vix_z"])
    assert pd.isna(out.iloc[0]["interactions__log_ret_5d_x_vix_z"])
    # Row 1's same products should NOT be NaN (vix_z still populated there)
    assert pd.notna(out.iloc[1]["interactions__rsi14_x_vix_z"])
