"""Alpaca connection wrapper with paper/live mode selection.

Unlike IBKR (which uses local Gateway + auto-detect ports), Alpaca is a
pure-HTTPS REST broker. The mode (paper vs live) is determined by which
API key pair you use:

  - paper: https://paper-api.alpaca.markets   keys from the paper dashboard
  - live : https://api.alpaca.markets         keys from the live dashboard

We keep BOTH key pairs in env so you can flip modes without re-entering
credentials. The `ALPACA_MODE` env var (paper|live, default paper) selects
which pair to use. Live mode additionally requires an opt-in flag (see
RiskConfig.confirm_live_account_id in orders.py).

Env vars:
  ALPACA_MODE              "paper" or "live" (default: paper)
  ALPACA_PAPER_KEY         paper API key id
  ALPACA_PAPER_SECRET      paper API secret
  ALPACA_LIVE_KEY          live API key id
  ALPACA_LIVE_SECRET       live API secret

Override mode programmatically by passing `mode=` to KuberaAlpaca().
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.models import TradeAccount

from packages.common.config import settings

log = logging.getLogger(__name__)

Mode = Literal["paper", "live"]

PAPER_REST_BASE = "https://paper-api.alpaca.markets"
LIVE_REST_BASE = "https://api.alpaca.markets"
PAPER_STREAM_URL = "wss://paper-api.alpaca.markets/stream"
LIVE_STREAM_URL = "wss://api.alpaca.markets/stream"


@dataclass(frozen=True)
class AlpacaSession:
    """Snapshot of the connected Alpaca session."""
    mode: Mode
    account_id: str            # the Alpaca-internal UUID
    account_number: str        # the human-facing account number (e.g. "PA12345678" for paper)
    status: str                # "ACTIVE" | "ACCOUNT_CLOSED" | ...
    currency: str              # always "USD" today
    portfolio_value: float
    cash: float
    buying_power: float
    pattern_day_trader: bool
    trading_blocked: bool
    connected_at: float        # unix seconds


def _resolve_mode(mode: Optional[Mode]) -> Mode:
    if mode is not None:
        return mode
    raw = (settings.alpaca_mode or "paper").strip().lower()
    if raw not in ("paper", "live"):
        raise ValueError(f"ALPACA_MODE must be 'paper' or 'live', got {raw!r}")
    return raw  # type: ignore[return-value]


def _load_keys(mode: Mode) -> tuple[str, str]:
    if mode == "paper":
        key = (settings.alpaca_paper_key or "").strip()
        sec = (settings.alpaca_paper_secret or "").strip()
        if not key or not sec:
            raise RuntimeError(
                "ALPACA_PAPER_KEY / ALPACA_PAPER_SECRET not set. "
                "Get them from https://app.alpaca.markets/paper/dashboard/overview "
                "and set them in your .env."
            )
        return key, sec
    key = (settings.alpaca_live_key or "").strip()
    sec = (settings.alpaca_live_secret or "").strip()
    if not key or not sec:
        raise RuntimeError(
            "ALPACA_LIVE_KEY / ALPACA_LIVE_SECRET not set. "
            "Get them from https://app.alpaca.markets/brokerage/dashboard/overview "
            "and set them in your .env."
        )
    return key, sec


def _classify_account_mode(account_number: str) -> Mode:
    """Alpaca paper account_numbers begin with 'PA'; live do not."""
    return "paper" if account_number.upper().startswith("PA") else "live"


class KuberaAlpaca:
    """One-process wrapper around alpaca-py TradingClient.

    Selects the API key pair based on `mode` (or ALPACA_MODE), connects,
    and verifies the account_number prefix matches the requested mode —
    a hard guard against accidentally pointing paper keys at the live URL
    or vice versa.
    """

    def __init__(self, mode: Optional[Mode] = None) -> None:
        self._mode: Mode = _resolve_mode(mode)
        self._client: Optional[TradingClient] = None
        self._session: Optional[AlpacaSession] = None

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def client(self) -> TradingClient:
        if self._client is None:
            raise RuntimeError("not connected; call .connect() first")
        return self._client

    @property
    def session(self) -> Optional[AlpacaSession]:
        return self._session

    def connect(self) -> AlpacaSession:
        key, secret = _load_keys(self._mode)
        client = TradingClient(api_key=key, secret_key=secret, paper=(self._mode == "paper"))
        # get_account() doubles as the connection check — invalid keys raise APIError here.
        acct: TradeAccount = client.get_account()  # type: ignore[assignment]
        account_number = str(acct.account_number)
        inferred_mode = _classify_account_mode(account_number)
        if inferred_mode != self._mode:
            raise RuntimeError(
                f"Alpaca account_number {account_number!r} looks like a "
                f"{inferred_mode.upper()} account but mode={self._mode!r} was "
                "requested. Check which key pair you put in which env var — "
                "Kubera refuses to proceed when the mode and the keys disagree."
            )
        sess = AlpacaSession(
            mode=self._mode,
            account_id=str(acct.id),
            account_number=account_number,
            status=str(acct.status),
            currency=str(acct.currency),
            portfolio_value=float(acct.portfolio_value or 0.0),
            cash=float(acct.cash or 0.0),
            buying_power=float(acct.buying_power or 0.0),
            pattern_day_trader=bool(acct.pattern_day_trader),
            trading_blocked=bool(acct.trading_blocked),
            connected_at=time.time(),
        )
        self._client = client
        self._session = sess
        log.info(
            "Alpaca connected: mode=%s account=%s status=%s NAV=%.2f %s",
            sess.mode, sess.account_number, sess.status,
            sess.portfolio_value, sess.currency,
        )
        return sess

    def disconnect(self) -> None:
        # TradingClient is stateless HTTP; just drop refs.
        self._client = None
        self._session = None

    def is_connected(self) -> bool:
        return self._client is not None and self._session is not None


# Convenience for one-off scripts / CLI: a global lazy singleton.
_singleton: Optional[KuberaAlpaca] = None


def get_session(mode: Optional[Mode] = None) -> AlpacaSession:
    """Get-or-create the singleton connection (for short-lived scripts)."""
    global _singleton
    if _singleton is None or not _singleton.is_connected() or (mode is not None and _singleton.mode != mode):
        _singleton = KuberaAlpaca(mode=mode)
        _singleton.connect()
    assert _singleton._session is not None
    return _singleton._session
