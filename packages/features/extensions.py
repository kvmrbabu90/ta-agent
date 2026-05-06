"""Pluggable extension registry for optional feature groups.

The pipeline picks up registered extensions automatically. An extension is
"active" only when its data dependency is present — e.g. macro features
appear once you've populated the macro_daily table; earnings features will
appear once you've populated an earnings_dates table; news_sentiment ones
once a news provider is wired in.

This is the integration point for v2 work (earnings, news sentiment) — the
intent is that adding those is a drop-in: write the data adapter, write
the FeatureGroup, register a small extension wrapper, done. No pipeline
edits required.

Pattern (see packages/features/macro.py for the live example):

    from packages.features.extensions import FeatureExtension, register_extension

    class _MyExtension(FeatureExtension):
        name = "my_thing"
        kind = "panel"           # or "per_symbol"

        def is_available(self, *, duckdb_path=None) -> bool:
            return my_data_table_has_rows(duckdb_path=duckdb_path)

        def make_group(self, *, duckdb_path=None):
            return MyFeatureGroup(duckdb_path=duckdb_path)

    register_extension(_MyExtension())

Failure isolation: an extension that raises in is_available or make_group
is silently skipped. The default pipeline must keep working even when an
extension's data layer is half-configured.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from packages.common.logging import log
from packages.features.base import FeatureGroup, PanelFeatureGroup


class FeatureExtension(ABC):
    """A feature group whose registration is gated on data availability."""

    name: str
    kind: Literal["per_symbol", "panel"]

    @abstractmethod
    def is_available(self, *, duckdb_path: str | None = None) -> bool:
        """Cheap probe: does this extension's data dependency have data?"""

    @abstractmethod
    def make_group(
        self, *, duckdb_path: str | None = None
    ) -> FeatureGroup | PanelFeatureGroup:
        """Build the feature group instance. Called only when is_available is True."""


_EXTENSIONS: list[FeatureExtension] = []


def register_extension(ext: FeatureExtension) -> None:
    """Register an extension. Idempotent on (kind, name) — re-registering
    replaces the previous binding so module reloads in tests behave."""
    for i, existing in enumerate(_EXTENSIONS):
        if existing.kind == ext.kind and existing.name == ext.name:
            _EXTENSIONS[i] = ext
            return
    _EXTENSIONS.append(ext)


def clear_extensions() -> None:
    """Test helper: drop all registered extensions."""
    _EXTENSIONS.clear()


def list_extensions() -> list[tuple[str, str]]:
    """Returns the (kind, name) pairs of all currently registered extensions."""
    return [(e.kind, e.name) for e in _EXTENSIONS]


def get_active_extensions(
    *, duckdb_path: str | None = None
) -> list[tuple[str, FeatureGroup | PanelFeatureGroup]]:
    """Return (kind, group) for each extension whose data is currently available.

    Errors during is_available or make_group are logged and skipped — the
    pipeline must remain functional regardless of extension state.
    """
    active: list[tuple[str, FeatureGroup | PanelFeatureGroup]] = []
    for ext in _EXTENSIONS:
        try:
            if not ext.is_available(duckdb_path=duckdb_path):
                continue
        except Exception as exc:  # noqa: BLE001 — never break the pipeline
            log.debug(f"extension {ext.kind}/{ext.name} availability check raised: {exc!r}")
            continue
        try:
            group = ext.make_group(duckdb_path=duckdb_path)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"extension {ext.kind}/{ext.name} make_group failed: {exc!r}")
            continue
        active.append((ext.kind, group))
    return active
