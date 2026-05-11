"""FastAPI application entry point.

    uvicorn services.api.main:app --reload

OpenAPI docs at /docs (Swagger UI) and /redoc.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from packages.common.config import settings
from packages.common.logging import log
from packages.inference.db import init_predictions_db
from services.api.routes import (
    admin as admin_routes,
)
from services.api.routes import (
    explain as explain_routes,
)
from services.api.routes import (
    paper as paper_routes,
)
from services.api.routes import (
    performance as performance_routes,
)
from services.api.routes import (
    predictions as predictions_routes,
)
from services.api.routes import (
    stocks as stocks_routes,
)
from services.api.routes import (
    universe as universe_routes,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info("ta-agent API starting")

    if not Path(settings.duckdb_path).exists():
        log.warning(
            f"DuckDB not found at {settings.duckdb_path} — read endpoints "
            "depending on OHLCV / membership will return empty"
        )

    # Make sure the predictions DB exists with the right schema even before
    # the first prediction is logged (avoids 500s on /predictions/top).
    try:
        init_predictions_db()
    except Exception as exc:  # noqa: BLE001 — never block startup on DB init
        log.warning(f"failed to init predictions SQLite: {exc!r}")

    yield
    log.info("ta-agent API shutting down")


app = FastAPI(
    title="ta-agent API",
    description="Read-mostly REST API for ta-agent predictions and performance.",
    version="0.1.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default
        "http://localhost:5174",  # alt Vite port if 5173 is taken
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


app.include_router(universe_routes.router)
app.include_router(predictions_routes.router)
app.include_router(stocks_routes.router)
app.include_router(performance_routes.router)
app.include_router(explain_routes.router)
app.include_router(admin_routes.router)
app.include_router(paper_routes.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}
