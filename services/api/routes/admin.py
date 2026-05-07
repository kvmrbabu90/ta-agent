"""Admin / settings routes.

Single exception to the API's read-only posture: Kite session refresh.
The endpoint writes a local file under ``data/processed/`` that the
kite_adapter falls back to. No remote calls except the SDK exchange
itself.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.api.services.admin_service import (
    KiteLoginError,
    kite_exchange_token,
    kite_login_url,
    kite_session_status,
)

router = APIRouter(prefix="/admin", tags=["admin"])


class KiteLoginUrlResponse(BaseModel):
    url: str


class KiteExchangeRequest(BaseModel):
    request_token: str = Field(..., min_length=4)


class KiteExchangeResponse(BaseModel):
    ok: bool
    user_id: str | None = None
    user_name: str | None = None
    exchanged_at: str


class KiteStatusResponse(BaseModel):
    configured_api_key: bool
    has_token_env: bool
    has_token_file: bool
    session_path: str
    user_id: str | None = None
    user_name: str | None = None
    exchanged_at: str | None = None
    file_error: str | None = None


@router.get("/kite/login-url", response_model=KiteLoginUrlResponse)
def get_kite_login_url() -> KiteLoginUrlResponse:
    try:
        return KiteLoginUrlResponse(url=kite_login_url())
    except KiteLoginError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/kite/exchange", response_model=KiteExchangeResponse)
def post_kite_exchange(body: KiteExchangeRequest) -> KiteExchangeResponse:
    try:
        result = kite_exchange_token(body.request_token)
    except KiteLoginError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return KiteExchangeResponse(**result)


@router.get("/kite/status", response_model=KiteStatusResponse)
def get_kite_status() -> KiteStatusResponse:
    return KiteStatusResponse(**kite_session_status())
