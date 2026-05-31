"""IB Gateway / TWS auto-detect connection.

Probes the four standard ports in order — Gateway-paper 4002, Gateway-live
4001, TWS-paper 7497, TWS-live 7496 — and connects to whichever responds.
Identifies paper vs live from the account-number prefix (`DU*` = paper,
anything else = live). Refuses to proceed if both a paper and a live
gateway are running on the same host (ambiguous).
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

from ib_insync import IB

log = logging.getLogger(__name__)

# Probe order: paper Gateway first since that's the common case during
# initial deployment. The 4-port set covers both Gateway and TWS, both
# modes. User just runs whatever they want; this finds it.
PROBE_PORTS: tuple[tuple[int, str, str], ...] = (
    (4002, "gateway", "paper"),
    (4001, "gateway", "live"),
    (7497, "tws",     "paper"),
    (7496, "tws",     "live"),
)

HOST = "127.0.0.1"
KUBERA_CLIENT_ID_SYNC = 10
KUBERA_CLIENT_ID_ORDER = 11
KUBERA_CLIENT_ID_CLI = 12


@dataclass(frozen=True)
class IbSession:
    """Snapshot of the connected IB session."""
    mode: str            # "paper" or "live"
    surface: str         # "gateway" or "tws"
    host: str
    port: int
    account_id: str      # e.g. "DU1234567" (paper) or "U1234567" (live)
    server_version: int
    connected_at: float  # unix seconds


def _classify_mode(account_id: str) -> str:
    """Account-number convention: DU* = paper, anything else = live.

    IBKR has used this convention since paper accounts were introduced; the
    `D` stands for "demo" in old IBKR docs. There is no documented case
    where a live account starts with `DU`.
    """
    return "paper" if account_id.startswith("DU") else "live"


class KuberaIB:
    """One-process wrapper around ib_insync.IB with auto-detect on connect."""

    def __init__(self, client_id: int = KUBERA_CLIENT_ID_SYNC, timeout: float = 4.0) -> None:
        self._client_id = client_id
        self._timeout = timeout
        self._ib: Optional[IB] = None
        self._session: Optional[IbSession] = None

    @property
    def ib(self) -> IB:
        if self._ib is None or not self._ib.isConnected():
            raise RuntimeError("not connected; call .connect() first")
        return self._ib

    @property
    def session(self) -> Optional[IbSession]:
        return self._session

    def connect(self) -> IbSession:
        """Probe ports in order and connect to the first one that responds.

        Raises ConnectionError if no port responds, or RuntimeError if more
        than one mode is detected on the same host (e.g. user is running
        both a paper Gateway and a live Gateway — refuse rather than guess).
        """
        # ib_insync uses asyncio under the hood; in some hosts (notebooks,
        # certain FastAPI workers) the event loop is already running. The
        # library's nest_asyncio dependency takes care of nested loops.
        candidates: list[tuple[IB, IbSession]] = []
        for port, surface, _expected in PROBE_PORTS:
            ib = IB()
            try:
                ib.connect(HOST, port, clientId=self._client_id, timeout=self._timeout)
            except Exception:
                continue
            try:
                managed = ib.managedAccounts()
                if not managed:
                    ib.disconnect()
                    continue
                # Pick the first managed account on this connection.
                acct = managed[0]
                sess = IbSession(
                    mode=_classify_mode(acct),
                    surface=surface,
                    host=HOST,
                    port=port,
                    account_id=acct,
                    server_version=ib.client.serverVersion() or -1,
                    connected_at=time.time(),
                )
                candidates.append((ib, sess))
            except Exception:
                try:
                    ib.disconnect()
                except Exception:
                    pass
                continue

        if not candidates:
            raise ConnectionError(
                "No IB Gateway / TWS reachable on 127.0.0.1 ports "
                "4002/4001/7497/7496. Start Gateway and log in (paper or live)."
            )

        # If more than one connection succeeded AND they report different
        # modes, that's ambiguous — refuse rather than guess.
        modes = {s.mode for (_, s) in candidates}
        if len(modes) > 1:
            for ib, _ in candidates:
                try:
                    ib.disconnect()
                except Exception:
                    pass
            raise RuntimeError(
                "Both paper AND live IB sessions detected on this host. "
                "Stop one before continuing — Kubera refuses to guess which "
                "to use."
            )

        # Keep the first successful connection; close any others.
        chosen_ib, chosen_session = candidates[0]
        for ib, _ in candidates[1:]:
            try:
                ib.disconnect()
            except Exception:
                pass

        self._ib = chosen_ib
        self._session = chosen_session
        log.info(
            "IBKR connected: mode=%s account=%s %s:%d (%s, server v%d)",
            chosen_session.mode,
            chosen_session.account_id,
            chosen_session.host,
            chosen_session.port,
            chosen_session.surface,
            chosen_session.server_version,
        )
        return chosen_session

    def disconnect(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            try:
                self._ib.disconnect()
            except Exception:
                pass
        self._ib = None
        self._session = None

    def is_connected(self) -> bool:
        return self._ib is not None and self._ib.isConnected()


# Convenience for one-off scripts / CLI: a global lazy singleton.
_singleton: Optional[KuberaIB] = None


def get_session(client_id: int = KUBERA_CLIENT_ID_SYNC) -> IbSession:
    """Get-or-create the singleton connection (for short-lived scripts)."""
    global _singleton
    if _singleton is None or not _singleton.is_connected():
        _singleton = KuberaIB(client_id=client_id)
        _singleton.connect()
    assert _singleton._session is not None
    return _singleton._session
