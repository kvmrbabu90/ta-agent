"""Interaction features — products of per-symbol features and macro/regime
features broadcast to the panel.

Why this exists:
    Universe-broadcast macro features (vix_z, credit_spread_z, etc.) added flat
    to LightGBM training showed -38% to -74% rank-IC in controlled CV (Step 4).
    Universe-constant features have zero per-day cross-sectional discrimination.
    The only way they can help is via INTERACTIONS with per-symbol features —
    "in high-VIX regimes, momentum reverses; in low-VIX regimes, momentum
    continues." LightGBM CAN learn these interactions through deep enough trees,
    but with our hyperparams (num_leaves=117, feature_fraction=0.86) it doesn't
    reliably do so. Pre-engineering the interactions explicitly gives the model
    direct access to the regime-conditioned signal.

Features (8):
    interactions__rsi14_x_vix_z               — high RSI in high-VIX → reversal candidate
    interactions__log_ret_5d_x_vix_z          — recent winners in high-VIX → reversal candidate
    interactions__log_ret_20d_x_credit_z      — medium-term winners in widening credit spreads
    interactions__rsi14_x_credit_z            — overbought when credit is risk-off
    interactions__realized_vol_x_vix_z        — high-vol stock in high-VIX regime (vol clustering)
    interactions__log_ret_5d_x_yield_curve_z  — recent winners with yield-curve regime
    interactions__macd_hist_x_dxy_chg_5d      — trend strength × dollar momentum (FX-sensitive sectors)
    interactions__rsi14_x_treasury_chg_5d     — overbought when rates spiking (rate-sensitive reversion)

All are products of (already existing) per-symbol and macro panel columns.
NaN propagates naturally — when either input is NaN, the interaction is NaN.
"""

from __future__ import annotations

import pandas as pd

# Side-effect import: ensure MacroFeatures registers BEFORE
# InteractionFeatures, so that when the pipeline runs panel groups in
# registration order, macro__* columns are already merged onto the panel
# by the time InteractionFeatures.compute() is called. Without this,
# alphabetical import-sorting (ruff/isort) flips the order and
# interaction columns come out 100% NaN.
import packages.features.macro  # noqa: F401  (must be before our own register_extension)
from packages.features.base import PanelFeatureGroup
from packages.features.extensions import FeatureExtension, register_extension
from packages.ingestion.macro import has_macro_data

# (per_symbol_col, macro_col, output_name)
_INTERACTIONS: tuple[tuple[str, str, str], ...] = (
    ("momentum__rsi_14",            "macro__vix_level_z_252",         "rsi14_x_vix_z"),
    ("price__log_return_5d",        "macro__vix_level_z_252",         "log_ret_5d_x_vix_z"),
    ("price__log_return_20d",       "macro__credit_spread_z_252",     "log_ret_20d_x_credit_z"),
    ("momentum__rsi_14",            "macro__credit_spread_z_252",     "rsi14_x_credit_z"),
    ("price__realized_vol_20d",     "macro__vix_level_z_252",         "realized_vol_x_vix_z"),
    ("price__log_return_5d",        "macro__yield_curve_slope_z_252", "log_ret_5d_x_yield_curve_z"),
    ("trend__macd_histogram",       "macro__dxy_chg_5d",              "macd_hist_x_dxy_chg_5d"),
    ("momentum__rsi_14",            "macro__treasury_10y_chg_5d",     "rsi14_x_treasury_chg_5d"),
)


class InteractionFeatures(PanelFeatureGroup):
    """Pre-engineered per-symbol × macro interaction features.

    Must run AFTER MacroFeatures (which produces the macro__* columns) — the
    extension registry handles ordering by appending to the panel-groups list,
    so as long as macro registers before interactions, this works.
    """

    name = "interactions"

    def compute(self, panel: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(panel)
        out = panel[["symbol", "bar_date"]].copy()
        for left, right, name in _INTERACTIONS:
            col = f"{self.name}__{name}"
            if left not in panel.columns or right not in panel.columns:
                out[col] = pd.NA
                continue
            out[col] = (panel[left] * panel[right]).values
        return out


# --- Extension registration --------------------------------------------------


class _InteractionsExtension(FeatureExtension):
    """Only registers when macro_daily has data (otherwise the macro side of
    every interaction is NaN, making the whole group dead weight)."""

    name = "interactions"
    kind = "panel"

    def is_available(self, *, duckdb_path: str | None = None) -> bool:
        return has_macro_data(duckdb_path=duckdb_path)

    def make_group(self, *, duckdb_path: str | None = None) -> InteractionFeatures:
        del duckdb_path  # InteractionFeatures has no DB access — just multiplies columns
        return InteractionFeatures()


register_extension(_InteractionsExtension())
