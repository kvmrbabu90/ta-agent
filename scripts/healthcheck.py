"""End-to-end stack diagnostic.

Run::

    python -m scripts.healthcheck                # checks local DBs + models
    python -m scripts.healthcheck --check-api    # also pings local FastAPI
    python -m scripts.healthcheck --check-network  # pings yfinance

Exit codes:
    0  all green
    1  any yellow (warnings, system likely usable but degraded)
    2  any red (system likely broken)
"""

from __future__ import annotations

import enum
import sqlite3
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import click

from packages.common.config import MODELS_DIR, settings
from packages.common.logging import log


class Status(enum.Enum):
    GREEN = 0
    YELLOW = 1
    RED = 2


# Color escape sequences: bright bg + black fg, padded.
_BADGES = {
    Status.GREEN: "\x1b[42m\x1b[30m  OK  \x1b[0m",
    Status.YELLOW: "\x1b[43m\x1b[30m WARN \x1b[0m",
    Status.RED: "\x1b[41m\x1b[30m FAIL \x1b[0m",
}


@dataclass
class CheckResult:
    name: str
    status: Status
    message: str


CheckFn = Callable[[], CheckResult]


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_duckdb_exists() -> CheckResult:
    p = Path(settings.duckdb_path)
    if not p.exists():
        return CheckResult("DuckDB file", Status.RED, f"missing at {p}")
    try:
        import duckdb

        with duckdb.connect(str(p)) as conn:
            n = conn.execute("SELECT COUNT(*) FROM ohlcv_daily").fetchone()[0]
    except Exception as exc:
        return CheckResult("DuckDB file", Status.RED, f"connect/query failed: {exc!r}")
    if n == 0:
        return CheckResult("DuckDB file", Status.YELLOW, "ohlcv_daily is empty")
    return CheckResult("DuckDB file", Status.GREEN, f"{n:,} OHLCV rows")


def check_ohlcv_freshness() -> CheckResult:
    try:
        import duckdb

        with duckdb.connect(settings.duckdb_path) as conn:
            row = conn.execute("SELECT MAX(bar_date) FROM ohlcv_daily").fetchone()
    except Exception as exc:
        return CheckResult("OHLCV freshness", Status.RED, f"query failed: {exc!r}")
    if not row or row[0] is None:
        return CheckResult("OHLCV freshness", Status.YELLOW, "no bars stored yet")
    latest = row[0] if isinstance(row[0], date) else date.fromisoformat(str(row[0]))
    age = (date.today() - latest).days
    if age <= 5:
        return CheckResult("OHLCV freshness", Status.GREEN, f"latest bar {latest} ({age}d ago)")
    if age <= 14:
        return CheckResult("OHLCV freshness", Status.YELLOW, f"latest bar {latest} ({age}d ago)")
    return CheckResult("OHLCV freshness", Status.RED, f"latest bar {latest} ({age}d ago)")


def check_membership_present() -> CheckResult:
    try:
        import duckdb

        with duckdb.connect(settings.duckdb_path) as conn:
            rows = conn.execute(
                "SELECT universe, COUNT(*) FROM index_membership GROUP BY universe"
            ).fetchall()
    except Exception as exc:
        return CheckResult("Universe membership", Status.RED, f"query failed: {exc!r}")
    counts = {u: int(n) for u, n in rows}
    missing = [u for u in ("SP500", "NIFTY100") if counts.get(u, 0) == 0]
    if missing:
        return CheckResult(
            "Universe membership", Status.RED,
            f"empty for: {', '.join(missing)} (run scripts.refresh_universes)",
        )
    return CheckResult(
        "Universe membership",
        Status.GREEN,
        ", ".join(f"{u}={counts[u]}" for u in counts),
    )


def check_models_present() -> CheckResult:
    base = Path(MODELS_DIR)
    if not base.exists():
        return CheckResult("Trained models", Status.RED, f"models dir does not exist: {base}")
    needs = [(u, t) for u in ("SP500", "NIFTY100") for t in ("regression", "classification")]
    missing: list[str] = []
    for u, t in needs:
        candidates = sorted(
            d for d in base.iterdir()
            if d.is_dir() and d.name.startswith(f"{u}_{t}_")
            and (d / "metadata.json").exists()
        )
        if not candidates:
            missing.append(f"{u}/{t}")
    if missing:
        sev = Status.YELLOW if len(missing) < len(needs) else Status.RED
        return CheckResult("Trained models", sev, f"missing: {', '.join(missing)}")
    return CheckResult("Trained models", Status.GREEN, "all 4 (universe × target) present")


def check_predictions_recent() -> CheckResult:
    p = Path(settings.predictions_sqlite_path)
    if not p.exists():
        return CheckResult(
            "Predictions log",
            Status.YELLOW,
            f"sqlite file missing at {p} — never run jobs.daily_predict?",
        )
    cutoff = (date.today() - timedelta(days=7)).isoformat()
    try:
        with sqlite3.connect(str(p)) as conn:
            row = conn.execute(
                "SELECT COUNT(*), MAX(as_of) FROM predictions_log WHERE as_of >= ?",
                [cutoff],
            ).fetchone()
    except sqlite3.OperationalError as exc:
        return CheckResult(
            "Predictions log",
            Status.YELLOW,
            f"table not yet created (jobs.daily_predict has not run): {exc}",
        )
    if not row or row[0] == 0:
        return CheckResult(
            "Predictions log",
            Status.YELLOW,
            "no predictions in the last 7 days",
        )
    return CheckResult(
        "Predictions log",
        Status.GREEN,
        f"{row[0]} predictions since {cutoff} (latest as_of {row[1]})",
    )


def check_no_orphan_unsettled() -> CheckResult:
    p = Path(settings.predictions_sqlite_path)
    if not p.exists():
        return CheckResult("Settlement freshness", Status.YELLOW, "no predictions DB yet")
    cutoff = (date.today() - timedelta(days=14)).isoformat()
    try:
        with sqlite3.connect(str(p)) as conn:
            n = conn.execute(
                """
                SELECT COUNT(*) FROM predictions_log
                WHERE as_of <= ? AND realized_return IS NULL
                """,
                [cutoff],
            ).fetchone()[0]
    except sqlite3.OperationalError:
        return CheckResult("Settlement freshness", Status.YELLOW, "table missing")
    if n == 0:
        return CheckResult("Settlement freshness", Status.GREEN, "no orphans older than 14d")
    return CheckResult(
        "Settlement freshness",
        Status.YELLOW,
        f"{n} predictions older than 14d still unsettled (run settle_predictions)",
    )


def check_api(url: str) -> CheckResult:
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=3) as resp:
            if resp.status == 200:
                return CheckResult("FastAPI /health", Status.GREEN, f"reachable at {url}")
            return CheckResult(
                "FastAPI /health", Status.RED, f"HTTP {resp.status} from {url}"
            )
    except Exception as exc:
        return CheckResult(
            "FastAPI /health",
            Status.RED,
            f"could not reach {url}: {exc!r}",
        )


def check_yfinance() -> CheckResult:
    try:
        import yfinance as yf  # noqa: F401

        df = yf.download("AAPL", period="5d", progress=False, threads=False)
        if df is None or df.empty:
            return CheckResult("yfinance reachability", Status.YELLOW, "empty response")
        return CheckResult(
            "yfinance reachability",
            Status.GREEN,
            f"got {len(df)} rows for AAPL/5d",
        )
    except Exception as exc:
        return CheckResult("yfinance reachability", Status.YELLOW, f"failed: {exc!r}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _print_results(results: list[CheckResult]) -> None:
    name_w = max(len(r.name) for r in results)
    print()
    for r in results:
        badge = _BADGES[r.status]
        print(f"{badge}  {r.name.ljust(name_w)}   {r.message}")
    print()


@click.command()
@click.option("--check-api", is_flag=True, help="Also check FastAPI /health endpoint.")
@click.option(
    "--api-url",
    default="http://localhost:8000/health",
    show_default=True,
)
@click.option("--check-network", is_flag=True, help="Also ping yfinance (network egress).")
def main(check_api: bool, api_url: str, check_network: bool) -> None:
    log.info("running healthcheck")
    checks: list[CheckFn] = [
        check_duckdb_exists,
        check_ohlcv_freshness,
        check_membership_present,
        check_models_present,
        check_predictions_recent,
        check_no_orphan_unsettled,
    ]
    if check_api:
        checks.append(lambda: check_api_url_wrapper(api_url))
    if check_network:
        checks.append(check_yfinance)

    results: list[CheckResult] = []
    for fn in checks:
        try:
            results.append(fn())
        except Exception as exc:  # noqa: BLE001 — never let a check itself crash
            results.append(
                CheckResult(getattr(fn, "__name__", "<check>"), Status.RED, f"check raised: {exc!r}")
            )
    _print_results(results)

    worst = max((r.status.value for r in results), default=0)
    print(f"healthcheck completed at {datetime.utcnow().isoformat()}Z (worst={Status(worst).name})\n")
    sys.exit(int(worst))


def check_api_url_wrapper(url: str) -> CheckResult:
    return check_api(url)


if __name__ == "__main__":
    main()
