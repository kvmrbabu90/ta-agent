"""Unit tests for the NIFTY 100 PIT reconstruction loader."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from packages.ingestion.universe.nifty100_pit import (
    _load_changes,
    _reconstruct_intervals,
    _validate_and_sort,
    build_nifty100_pit_membership,
)


def _write_yaml(tmp_path: Path, changes: list[dict]) -> Path:
    p = tmp_path / "nifty100_changes.yaml"
    p.write_text(yaml.safe_dump({"changes": changes}), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _load_changes
# ---------------------------------------------------------------------------


def test_load_changes_empty_file_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "nifty100_changes.yaml"
    p.write_text("changes: []\n", encoding="utf-8")
    assert _load_changes(p) == []


def test_load_changes_missing_file_returns_empty(tmp_path: Path) -> None:
    assert _load_changes(tmp_path / "no-such-file.yaml") == []


def test_load_changes_parses_valid_rows(tmp_path: Path) -> None:
    p = _write_yaml(
        tmp_path,
        [
            {"effective_date": "2024-09-30", "action": "add", "symbol": "ZOMATO"},
            {"effective_date": "2024-09-30", "action": "remove", "symbol": "BERGEPAINT"},
        ],
    )
    out = _load_changes(p)
    assert len(out) == 2
    assert out[0]["action"] == "add" and out[0]["symbol"] == "ZOMATO"
    assert isinstance(out[0]["effective_date"], date)


def test_load_changes_skips_invalid_rows(tmp_path: Path) -> None:
    p = _write_yaml(
        tmp_path,
        [
            {"effective_date": "2024-09-30", "action": "add", "symbol": "GOOD"},
            {"effective_date": "not-a-date", "action": "add", "symbol": "BAD1"},
            {"effective_date": "2024-09-30", "action": "BOGUS", "symbol": "BAD2"},
            {"effective_date": "2024-09-30", "action": "remove", "symbol": ""},
        ],
    )
    out = _load_changes(p)
    syms = {r["symbol"] for r in out}
    assert syms == {"GOOD"}


# ---------------------------------------------------------------------------
# _validate_and_sort
# ---------------------------------------------------------------------------


def test_validate_drops_future_dates() -> None:
    today = date(2024, 1, 1)
    events = [
        {"effective_date": date(2023, 6, 1), "action": "add", "symbol": "OK"},
        {"effective_date": date(2025, 1, 1), "action": "add", "symbol": "FUTURE"},
    ]
    out = _validate_and_sort(events, today=today)
    assert {e["symbol"] for e in out} == {"OK"}


def test_validate_sorts_oldest_first() -> None:
    events = [
        {"effective_date": date(2024, 1, 1), "action": "add", "symbol": "B"},
        {"effective_date": date(2023, 1, 1), "action": "add", "symbol": "A"},
        {"effective_date": date(2024, 1, 1), "action": "remove", "symbol": "C"},
    ]
    out = _validate_and_sort(events, today=date(2025, 1, 1))
    # Sorted by (effective_date, action). 'add' < 'remove' alphabetically.
    # Same date → 'add' first.
    assert [e["symbol"] for e in out] == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# _reconstruct_intervals — the algorithm itself
# ---------------------------------------------------------------------------


def test_no_events_yields_open_intervals_for_all_current() -> None:
    current = {"AAA": "Alpha Co", "BBB": "Beta Co"}
    df = _reconstruct_intervals(current, [], today=date(2024, 6, 1))
    assert len(df) == 2
    assert (df["end_date"].isna()).all()
    # All start_date = RELIABLE_START
    assert (df["start_date"] == df["start_date"].min()).all()


def test_add_event_sets_start_date_for_current_member() -> None:
    """If current members include ZOMATO and there's an 'add ZOMATO' event,
    the rebuilt interval has start_date = that event's date and end_date = None."""
    current = {"ZOMATO": "Zomato Ltd", "OTHER": "Other Co"}
    events = [
        {"effective_date": date(2024, 9, 30), "action": "add", "symbol": "ZOMATO"},
    ]
    df = _reconstruct_intervals(current, events, today=date(2025, 1, 1))
    z = df[df["symbol"] == "ZOMATO"].iloc[0]
    assert z["start_date"] == date(2024, 9, 30)
    assert z["end_date"] is None or pd.isna(z["end_date"])
    other = df[df["symbol"] == "OTHER"].iloc[0]
    # OTHER had no event, started at RELIABLE_START
    assert other["start_date"] is not None
    assert other["end_date"] is None or pd.isna(other["end_date"])


def test_remove_event_creates_historical_interval_ending_on_date() -> None:
    """A 'remove BERGEPAINT' event should produce a closed interval ending
    on that date — the symbol is NOT in current_symbols anymore."""
    current = {"OTHER": "Other Co"}
    events = [
        {"effective_date": date(2024, 9, 30), "action": "remove", "symbol": "BERGEPAINT"},
    ]
    df = _reconstruct_intervals(current, events, today=date(2025, 1, 1))
    bp = df[df["symbol"] == "BERGEPAINT"].iloc[0]
    assert bp["end_date"] == date(2024, 9, 30)
    # start_date defaults to RELIABLE_START because no 'add' event seen
    assert bp["start_date"] is not None


def test_add_then_remove_creates_closed_interval() -> None:
    """Symbol joins on D1, leaves on D2, no longer in current → one closed
    interval [D1, D2]."""
    current: dict[str, str] = {}
    events = [
        {"effective_date": date(2022, 3, 31), "action": "add", "symbol": "TRANSIENT"},
        {"effective_date": date(2024, 9, 30), "action": "remove", "symbol": "TRANSIENT"},
    ]
    df = _reconstruct_intervals(current, events, today=date(2025, 1, 1))
    t = df[df["symbol"] == "TRANSIENT"].iloc[0]
    assert t["start_date"] == date(2022, 3, 31)
    assert t["end_date"] == date(2024, 9, 30)


def test_orphan_remove_logged_and_skipped() -> None:
    """A 'remove X' when X is currently a member is inconsistent —
    skip the event without crashing."""
    current = {"X": "X Co"}  # X IS currently a member
    events = [
        {"effective_date": date(2024, 9, 30), "action": "remove", "symbol": "X"},
    ]
    # Should not raise.
    df = _reconstruct_intervals(current, events, today=date(2025, 1, 1))
    # X stayed in the rolling state; produces a single open interval at RELIABLE_START
    rows_for_x = df[df["symbol"] == "X"]
    assert len(rows_for_x) == 1
    assert rows_for_x.iloc[0]["end_date"] is None or pd.isna(rows_for_x.iloc[0]["end_date"])


def test_orphan_add_logged_and_skipped() -> None:
    """A 'add X' when X is NOT currently a member at the rewind point is
    inconsistent — skip the event."""
    current: dict[str, str] = {}  # X NOT currently a member
    events = [
        {"effective_date": date(2024, 9, 30), "action": "add", "symbol": "X"},
    ]
    df = _reconstruct_intervals(current, events, today=date(2025, 1, 1))
    # No interval generated for X.
    assert "X" not in set(df["symbol"]) if not df.empty else True


# ---------------------------------------------------------------------------
# Top-level integration with mocked snapshot
# ---------------------------------------------------------------------------


def test_build_nifty100_pit_membership_end_to_end(tmp_path: Path) -> None:
    fake_snap = pd.DataFrame(
        [
            {"symbol": "ZOMATO", "company_name": "Zomato Ltd"},
            {"symbol": "RELIANCE", "company_name": "Reliance Industries Ltd."},
        ]
    )
    p = _write_yaml(
        tmp_path,
        [
            {"effective_date": "2024-09-30", "action": "add", "symbol": "ZOMATO"},
            {"effective_date": "2024-09-30", "action": "remove", "symbol": "BERGEPAINT"},
        ],
    )
    with patch(
        "packages.ingestion.universe.nifty100_pit._fetch_current_csv",
        return_value=fake_snap,
    ):
        df = build_nifty100_pit_membership(
            changes_path=p, today=date(2025, 1, 1)
        )

    assert set(df["universe"]) == {"NIFTY100"}
    assert (df["exchange"] == "NSE").all()
    by_sym = df.set_index("symbol").to_dict("index")
    assert "ZOMATO" in by_sym and by_sym["ZOMATO"]["start_date"] == date(2024, 9, 30)
    assert "RELIANCE" in by_sym and by_sym["RELIANCE"]["end_date"] is None
    # BERGEPAINT generated a closed historical row.
    assert "BERGEPAINT" in by_sym
    assert by_sym["BERGEPAINT"]["end_date"] == date(2024, 9, 30)
