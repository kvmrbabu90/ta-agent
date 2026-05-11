"""Paper-trading engine: simulates trading the model's predictions starting
with a fixed cash balance, generating equity curves and position snapshots."""

from packages.paper_trading.engine import (
    DEFAULT_CONFIG,
    StrategyConfig,
    backtest,
    init_paper_db,
)

__all__ = ["DEFAULT_CONFIG", "StrategyConfig", "backtest", "init_paper_db"]
