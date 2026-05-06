"""Universe and membership routes."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, Query

from services.api.deps import get_duckdb_conn
from services.api.schemas import MemberInfo, UniverseInfo
from services.api.services.predictions_service import list_members, list_universes

router = APIRouter(tags=["universes"])


@router.get("/universes", response_model=list[UniverseInfo])
def universes(duck=Depends(get_duckdb_conn)) -> list[UniverseInfo]:
    return list_universes(duck)


@router.get("/universes/{universe}/members", response_model=list[MemberInfo])
def members(
    universe: str,
    as_of: date | None = Query(default=None, description="Defaults to today"),
    duck=Depends(get_duckdb_conn),
) -> list[MemberInfo]:
    return list_members(duck, universe, as_of or date.today())
