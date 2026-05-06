"""Unit tests for jobs.daily_ingest orchestration."""

from __future__ import annotations

from datetime import date

import pytest

from jobs import daily_ingest


def _result(**kw):
    base = {
        "requested": kw.get("requested", 0),
        "succeeded": kw.get("succeeded", 0),
        "failed": kw.get("failed", 0),
        "rows_written": kw.get("rows_written", 0),
        "failures": kw.get("failures", []),
    }
    return base


def test_run_aggregates_ib_and_kite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(daily_ingest, "_is_any_market_open", lambda as_of=None: True)
    monkeypatch.setattr(
        daily_ingest, "ib_daily_update",
        lambda u: _result(requested=400, succeeded=395, failed=5, rows_written=395),
    )
    monkeypatch.setattr(
        daily_ingest, "kite_daily_update",
        lambda u: _result(requested=100, succeeded=98, failed=2, rows_written=98),
    )
    yf_calls: list[dict] = []

    def _yf_backfill(universe, start, end, symbols=None):
        yf_calls.append(
            {"universe": universe, "n_symbols": len(symbols or [])}
        )
        return _result(
            requested=len(symbols or []), succeeded=len(symbols or []),
            rows_written=len(symbols or []),
        )

    monkeypatch.setattr(daily_ingest, "yf_backfill_universe", _yf_backfill)

    summary = daily_ingest.run(today=date(2024, 1, 5))
    assert summary["requested"] == 500
    assert summary["succeeded"] == 493
    assert summary["coverage"] > 0.98
    # No fallbacks because failure lists were empty.
    assert yf_calls == []


def test_run_falls_back_to_yfinance_for_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(daily_ingest, "_is_any_market_open", lambda as_of=None: True)

    monkeypatch.setattr(
        daily_ingest, "ib_daily_update",
        lambda u: _result(
            requested=400, succeeded=399, failed=1, rows_written=399,
            failures=[{"symbol": "AAPL", "error": "boom"}],
        ),
    )
    monkeypatch.setattr(
        daily_ingest, "kite_daily_update",
        lambda u: _result(
            requested=100, succeeded=99, failed=1, rows_written=99,
            failures=[{"symbol": "RELIANCE", "error": "boom"}],
        ),
    )

    seen: list[tuple[str, list[str]]] = []

    def _yf_backfill(universe, start, end, symbols=None):
        seen.append((universe, list(symbols or [])))
        return _result(
            requested=len(symbols or []), succeeded=len(symbols or []),
            rows_written=len(symbols or []),
        )

    monkeypatch.setattr(daily_ingest, "yf_backfill_universe", _yf_backfill)

    summary = daily_ingest.run(today=date(2024, 1, 5))

    universes_called = {u for u, _ in seen}
    assert universes_called == {"SP500", "NIFTY100"}
    syms_called = {s for _, syms in seen for s in syms}
    assert syms_called == {"AAPL", "RELIANCE"}
    # Both fallbacks succeeded → totals reflect that.
    assert summary["succeeded"] == 399 + 99 + 1 + 1


def test_run_skips_on_non_trading_day(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(daily_ingest, "_is_any_market_open", lambda as_of=None: False)
    called: dict = {"ib": False, "kite": False}

    def _ib(u):
        called["ib"] = True
        return _result()

    def _kite(u):
        called["kite"] = True
        return _result()

    monkeypatch.setattr(daily_ingest, "ib_daily_update", _ib)
    monkeypatch.setattr(daily_ingest, "kite_daily_update", _kite)
    monkeypatch.setattr(daily_ingest, "yf_backfill_universe", lambda *a, **k: _result())

    summary = daily_ingest.run(today=date(2024, 1, 6))  # Saturday
    assert summary.get("skipped") is True
    assert called == {"ib": False, "kite": False}


def test_determine_exit_code_low_coverage_yields_2() -> None:
    summary = {"skipped": False, "requested": 100, "succeeded": 50, "failed": 50, "coverage": 0.5}
    assert daily_ingest.determine_exit_code(summary) == 2


def test_determine_exit_code_too_many_failures_yields_3() -> None:
    summary = {"skipped": False, "requested": 1000, "succeeded": 949, "failed": 51, "coverage": 0.949}
    assert daily_ingest.determine_exit_code(summary) == 3


def test_determine_exit_code_skipped_yields_0() -> None:
    assert daily_ingest.determine_exit_code({"skipped": True}) == 0


def test_determine_exit_code_clean_run_yields_0() -> None:
    summary = {"skipped": False, "requested": 100, "succeeded": 95, "failed": 5, "coverage": 0.95}
    assert daily_ingest.determine_exit_code(summary) == 0


def test_run_recovers_when_ib_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(daily_ingest, "_is_any_market_open", lambda as_of=None: True)

    def _boom(u):
        raise RuntimeError("ib died")

    monkeypatch.setattr(daily_ingest, "ib_daily_update", _boom)
    monkeypatch.setattr(
        daily_ingest, "kite_daily_update",
        lambda u: _result(requested=100, succeeded=100, rows_written=100),
    )
    monkeypatch.setattr(daily_ingest, "yf_backfill_universe", lambda *a, **k: _result())

    summary = daily_ingest.run(today=date(2024, 1, 5))
    # Run completes; ib leg counted as failed, kite leg succeeded.
    assert summary["per_source"]["ib_sp500"]["failed"] == 1
    assert summary["per_source"]["kite_nifty100"]["succeeded"] == 100
