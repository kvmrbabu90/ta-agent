"""Admin operations: convenience writes for the local user.

The API is otherwise read-only. The single exception lives here: a Kite
login-token exchange so the user can re-auth from the frontend instead
of running `scripts.kite_login` in a terminal each morning.

Tokens are persisted to ``settings.kite_session_path`` (a small JSON file
under ``data/processed/``). The kite adapter falls back to that file when
the env-var token is empty, so a fresh login picks up automatically on
the next ingest / predict run — no API restart required.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from kiteconnect import KiteConnect

from packages.common.config import settings
from packages.common.logging import log


class KiteLoginError(RuntimeError):
    """Raised when Kite login URL or token exchange cannot be performed."""


def kite_login_url() -> str:
    if not settings.kite_api_key:
        raise KiteLoginError(
            "KITE_API_KEY is not set in .env — populate it and restart the API."
        )
    kite = KiteConnect(api_key=settings.kite_api_key)
    return kite.login_url()


def kite_exchange_token(request_token: str) -> dict:
    """Exchange a Zerodha redirect ``request_token`` for an access token.

    Persists the access_token to ``settings.kite_session_path`` so the
    ingest / predict flows pick it up on next call. Returns a sanitized
    summary suitable for the API response — never the raw token.
    """
    rt = (request_token or "").strip()
    if not rt:
        raise KiteLoginError("request_token is empty")
    if not settings.kite_api_key or not settings.kite_api_secret:
        raise KiteLoginError(
            "KITE_API_KEY and KITE_API_SECRET must both be set in .env."
        )

    kite = KiteConnect(api_key=settings.kite_api_key)
    try:
        session = kite.generate_session(rt, api_secret=settings.kite_api_secret)
    except Exception as exc:  # noqa: BLE001 — surface the SDK error
        raise KiteLoginError(f"generate_session failed: {exc!r}") from exc

    access_token = session.get("access_token")
    if not access_token:
        raise KiteLoginError(f"no access_token in session response: keys={list(session)}")

    payload = {
        "access_token": access_token,
        "user_id": session.get("user_id"),
        "user_name": session.get("user_name"),
        "exchanged_at": datetime.now(UTC).isoformat(),
    }
    p = Path(settings.kite_session_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info(f"persisted Kite session for {payload.get('user_id')} → {p}")

    # Sanitized response — return user info but never the token itself.
    return {
        "ok": True,
        "user_id": payload.get("user_id"),
        "user_name": payload.get("user_name"),
        "exchanged_at": payload["exchanged_at"],
    }


def kite_session_status() -> dict:
    """Report whether a token is currently available + when it was minted.

    Never returns the token itself.
    """
    p = Path(settings.kite_session_path)
    has_env = bool(settings.kite_access_token)
    has_file = p.exists()
    info: dict = {
        "configured_api_key": bool(settings.kite_api_key),
        "has_token_env": has_env,
        "has_token_file": has_file,
        "session_path": str(p),
    }
    if has_file:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            info["user_id"] = data.get("user_id")
            info["user_name"] = data.get("user_name")
            info["exchanged_at"] = data.get("exchanged_at")
        except Exception as exc:  # noqa: BLE001
            info["file_error"] = repr(exc)
    return info
