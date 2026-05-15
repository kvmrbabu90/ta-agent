"""Fetch and cache SEC EDGAR filing body text.

The existing `packages.ingestion.sec_filings` adapter pulls filing
METADATA (accession, date, items, primary_doc URL). This module fetches
the actual HTML body on demand and caches the cleaned text in
`news.sqlite`. Lazy by design — we only pay the HTTP round-trip the
first time we look at a filing.

URL pattern (no API key needed):
    https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_no_dashes}/{primary_doc}

SEC throttles at 10 req/sec per IP; we sleep 110ms between fetches.

Forms we care about (in order of signal density for a "structural reset"
classification):
  8-K   — Material events. The single best signal.
  10-Q  — Quarterly report. Long; we extract opening narrative only.
  10-K  — Annual report. Long; same approach.

Exhibit-99 attachments:
  Every quarterly earnings 8-K (item 2.02) references an attached
  Exhibit 99.1 — the actual press release with the beat/miss numbers
  and guidance. The 8-K's primary document is just a cover sheet
  referring to "see Exhibit 99.1 attached". Without fetching the
  exhibit, our LLM can only say "earnings was released" — it can't
  read whether it was a beat or a miss. We now also fetch any file
  in the filing's index whose name matches /ex.?99/i, store it as a
  separate row in news_articles, and surface it to the classifier as
  an additional numbered source.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import date

import duckdb
import requests
from bs4 import BeautifulSoup

from packages.common.config import settings
from packages.common.logging import log
from packages.news.storage import MAX_BODY_CHARS, init_news_db

_USER_AGENT = "ta-agent local research konda@example.com"
_PER_REQUEST_SLEEP_SEC = 0.11
_REQUEST_TIMEOUT_SEC = 30

# 8-K item codes that suggest a structural reset (we surface these
# prominently in the prompt context even when the body text is generic):
_RESET_ITEM_CODES = {
    "1.02": "termination of material agreement",
    "1.03": "bankruptcy or receivership",
    "2.04": "triggering events accelerating obligations",
    "2.05": "costs from exit/disposal",
    "2.06": "material impairment",
    "3.01": "delisting / failure to satisfy listing rule",
    "4.01": "auditor change",
    "4.02": "non-reliance on prior financials",
    "5.02": "officer/director departure or appointment",
    "5.07": "shareholder vote results",
    "8.01": "other events",
}
# Earnings items — neutral on their own; the body text decides.
_EARNINGS_ITEM_CODES = {
    "2.02": "results of operations",
    "9.01": "financial statements / exhibits",
}


def _accession_dir_url(cik: str, accession: str) -> str:
    """The directory URL for an accession — used to fetch index.json."""
    cik_int = int(cik)
    accession_no_dash = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_no_dash}/"


def _accession_to_url(cik: str, accession: str, primary_doc: str) -> str:
    """Build the canonical Archives URL.

    cik comes in as a 10-digit zero-padded string; the URL path uses the
    integer form. Accession needs its dashes removed.
    """
    return _accession_dir_url(cik, accession) + primary_doc


# Matches typical exhibit-99 filenames: 'ex991.htm', 'ex-99.1.htm',
# 'chtrex991earningsrelease.htm', 'exhibit99-1.htm', etc. We deliberately
# also accept the broader 'ex99' (e.g. 99.2 / 99.3) since some companies
# split content across multiple exhibit-99 attachments.
_EX99_PATTERN = re.compile(r"ex.?99", re.IGNORECASE)

# Skip these even if their name matches — XBRL plumbing, not narrative.
_SKIP_EXTENSIONS = {".xml", ".xsd", ".zip", ".jpg", ".jpeg", ".png", ".gif"}


def _list_exhibit99_filenames(cik: str, accession: str) -> list[str]:
    """Fetch the filing's index.json and return any exhibit-99 filenames.

    Returns an empty list on 404 or parse error — exhibit fetching is
    best-effort; the primary doc is enough on its own.
    """
    url = _accession_dir_url(cik, accession) + "index.json"
    try:
        resp = requests.get(
            url, headers={"User-Agent": _USER_AGENT, "Accept-Encoding": "gzip"},
            timeout=_REQUEST_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        items = resp.json().get("directory", {}).get("item", [])
    except Exception as exc:  # noqa: BLE001
        log.debug(f"edgar index.json failed for {accession}: {exc!r}")
        return []
    matches: list[str] = []
    for item in items:
        name = item.get("name") or ""
        if not name:
            continue
        # Reject XBRL & images by extension regardless of name match.
        if any(name.lower().endswith(ext) for ext in _SKIP_EXTENSIONS):
            continue
        if _EX99_PATTERN.search(name):
            matches.append(name)
    return matches


def _clean_html_to_text(html: str) -> str:
    """Strip tags and normalize whitespace. Keeps the visible reading order.

    8-K HTML pages start with a large XBRL/iXBRL header — tag IDs, CIK
    refs, hidden context tags. For an LLM that's 1-3k chars of noise
    before the actual narrative. We try to skip past it by snipping to
    the first 'Item N.NN' or 'Item N.NNN' marker found after the SEC
    boilerplate. If no Item marker exists (10-Q/10-K, press-release
    style 8-Ks), we return the full cleaned text.
    """
    soup = BeautifulSoup(html, "html.parser")
    # Drop scripts/styles. Keep tables (financial data + officer titles).
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)

    # Snip XBRL header. Searches AFTER the SEC form preamble so we
    # don't catch "Item Number" references in the boilerplate itself.
    item_marker = re.search(r"\bItem\s+\d{1,2}\.\d{2,3}\b", text)
    if item_marker and item_marker.start() > 200:
        text = text[item_marker.start():]
    return text


def _fetch_filings_metadata(
    symbol: str,
    since: date,
    duckdb_path: str | None = None,
) -> list[dict]:
    """Pull recent filing metadata from the existing `edgar_filings` table.

    Filters to 8-K / 10-Q / 10-K (forms with body worth showing to the LLM).
    Form 4 (insider transactions), S-8 (share grants), Schedule 13G/A are
    skipped — they're noise for our use case.
    """
    duck_path = duckdb_path or settings.duckdb_path
    conn = duckdb.connect(duck_path, read_only=True)
    try:
        rows = conn.execute(
            """
            SELECT cik, symbol, accession, filing_date, form_type, items, primary_doc
            FROM edgar_filings
            WHERE symbol = ?
              AND filing_date >= ?
              AND form_type IN ('8-K', '10-Q', '10-K')
            ORDER BY filing_date DESC
            """,
            [symbol, since],
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "cik": r[0], "symbol": r[1], "accession": r[2],
            "filing_date": r[3], "form_type": r[4], "items": r[5],
            "primary_doc": r[6],
        }
        for r in rows
    ]


def _cached_body(
    sqlite_conn: sqlite3.Connection, symbol: str, accession: str
) -> dict | None:
    row = sqlite_conn.execute(
        "SELECT form_type, items, filing_date, url, title, body, body_chars "
        "FROM news_articles WHERE symbol = ? AND source = 'edgar' AND source_id = ?",
        (symbol, accession),
    ).fetchone()
    if row is None:
        return None
    return {
        "form_type": row[0], "items": row[1], "filing_date": row[2],
        "url": row[3], "title": row[4], "body": row[5], "body_chars": row[6],
    }


def _persist_body(
    sqlite_conn: sqlite3.Connection,
    *,
    symbol: str,
    accession: str,
    form_type: str,
    items: str | None,
    filing_date: date,
    url: str,
    title: str | None,
    body: str,
    body_chars: int,
) -> None:
    sqlite_conn.execute(
        """
        INSERT OR REPLACE INTO news_articles
            (symbol, source, source_id, form_type, items, filing_date,
             url, title, body, body_chars)
        VALUES (?, 'edgar', ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (symbol, accession, form_type, items, filing_date.isoformat(),
         url, title, body, body_chars),
    )
    sqlite_conn.commit()


def fetch_filings_for_symbol(
    symbol: str,
    since: date,
    *,
    max_filings: int = 8,
    skip_body_fetch: bool = False,
    duckdb_path: str | None = None,
) -> list[dict]:
    """Return recent filings for `symbol` since `since`, with body text.

    Results are sorted newest-first and capped at `max_filings`. Body text
    is fetched lazily — if `news.sqlite` already has it, we don't hit the
    SEC. Pass `skip_body_fetch=True` to return metadata only (the body
    field will be None; useful for prompts that only need item codes).
    """
    init_news_db()
    meta = _fetch_filings_metadata(symbol, since, duckdb_path=duckdb_path)
    if not meta:
        return []
    meta = meta[:max_filings]

    sqlite_path = init_news_db()
    out: list[dict] = []
    conn = sqlite3.connect(sqlite_path)
    conn.execute("PRAGMA busy_timeout = 5000")
    last_http_ts: float | None = None
    try:
        for m in meta:
            accession = m["accession"]
            primary_doc = m["primary_doc"]
            if not primary_doc:
                # Some filings (Form 4s historically) had null primary_doc.
                continue
            url = _accession_to_url(m["cik"], accession, primary_doc)
            cached = _cached_body(conn, symbol, accession)
            if cached is not None:
                out.append({**m, "url": url, **cached})
                # Cached hit on primary doc still needs to surface any
                # exhibits — they live in separate news_articles rows.
                # _maybe_fetch_exhibits reads from the cache without re-
                # hitting SEC if the exhibits are already persisted.
                if m["form_type"] == "8-K":
                    last_http_ts = _maybe_fetch_exhibits(
                        conn=conn, m=m, primary_url=url,
                        last_http_ts=last_http_ts, out=out,
                    )
                continue
            if skip_body_fetch:
                out.append({**m, "url": url, "body": None, "body_chars": 0})
                continue
            # Throttle 10 req/sec.
            if last_http_ts is not None:
                elapsed = time.monotonic() - last_http_ts
                if elapsed < _PER_REQUEST_SLEEP_SEC:
                    time.sleep(_PER_REQUEST_SLEEP_SEC - elapsed)
            try:
                resp = requests.get(
                    url,
                    headers={"User-Agent": _USER_AGENT, "Accept-Encoding": "gzip"},
                    timeout=_REQUEST_TIMEOUT_SEC,
                )
                last_http_ts = time.monotonic()
                if resp.status_code == 404:
                    log.warning(f"edgar fetch 404: {url}")
                    body = ""
                    body_chars = 0
                else:
                    resp.raise_for_status()
                    raw = resp.text
                    body_chars = len(raw)
                    body = _clean_html_to_text(raw)[:MAX_BODY_CHARS]
            except Exception as exc:  # noqa: BLE001
                log.warning(f"edgar fetch failed for {symbol} {accession}: {exc!r}")
                body = ""
                body_chars = 0
            title = _derive_title(m, body)
            _persist_body(
                conn, symbol=symbol, accession=accession,
                form_type=m["form_type"], items=m["items"],
                filing_date=m["filing_date"], url=url, title=title,
                body=body, body_chars=body_chars,
            )
            out.append({
                **m, "url": url, "title": title, "body": body,
                "body_chars": body_chars,
            })
            # Exhibit-99 attachments — fetch only for 8-K filings since
            # 10-Q/10-K bodies already include all material narrative.
            if m["form_type"] == "8-K":
                last_http_ts = _maybe_fetch_exhibits(
                    conn=conn, m=m, primary_url=url,
                    last_http_ts=last_http_ts, out=out,
                )
    finally:
        conn.close()
    return out


def _maybe_fetch_exhibits(
    *,
    conn: sqlite3.Connection,
    m: dict,
    primary_url: str,
    last_http_ts: float | None,
    out: list[dict],
) -> float | None:
    """Fetch ex99 attachments for an 8-K, append filing-shaped rows to `out`,
    and return the updated last_http_ts so the outer loop's throttling is
    consistent.

    Each exhibit is persisted as its own news_articles row with a composite
    source_id of `{accession}__{filename}`. Cached on the same lazy basis as
    the primary doc — second call to the same accession is free.
    """
    accession = m["accession"]
    symbol = m["symbol"]
    # Listing the index.json itself is one HTTP request.
    if last_http_ts is not None:
        elapsed = time.monotonic() - last_http_ts
        if elapsed < _PER_REQUEST_SLEEP_SEC:
            time.sleep(_PER_REQUEST_SLEEP_SEC - elapsed)
    exhibit_names = _list_exhibit99_filenames(m["cik"], accession)
    last_http_ts = time.monotonic()
    if not exhibit_names:
        return last_http_ts

    for fname in exhibit_names:
        source_id = f"{accession}__{fname}"
        cached = _cached_body(conn, symbol, source_id)
        ex_url = _accession_dir_url(m["cik"], accession) + fname
        if cached is not None:
            out.append({
                **m, "url": ex_url, "exhibit": fname,
                "title": cached.get("title") or f"Exhibit 99 ({fname})",
                "body": cached.get("body"),
                "body_chars": cached.get("body_chars", 0),
            })
            continue
        if last_http_ts is not None:
            elapsed = time.monotonic() - last_http_ts
            if elapsed < _PER_REQUEST_SLEEP_SEC:
                time.sleep(_PER_REQUEST_SLEEP_SEC - elapsed)
        try:
            resp = requests.get(
                ex_url,
                headers={"User-Agent": _USER_AGENT, "Accept-Encoding": "gzip"},
                timeout=_REQUEST_TIMEOUT_SEC,
            )
            last_http_ts = time.monotonic()
            if resp.status_code == 404:
                body, body_chars = "", 0
            else:
                resp.raise_for_status()
                raw = resp.text
                body_chars = len(raw)
                body = _clean_html_to_text(raw)[:MAX_BODY_CHARS]
        except Exception as exc:  # noqa: BLE001
            log.warning(f"edgar exhibit fetch failed for {symbol} {source_id}: {exc!r}")
            body, body_chars = "", 0
        ex_title = f"Exhibit 99 attached to 8-K: {fname}"
        _persist_body(
            conn, symbol=symbol, accession=source_id,
            form_type="8-K_ex99", items=m["items"],
            filing_date=m["filing_date"], url=ex_url, title=ex_title,
            body=body, body_chars=body_chars,
        )
        out.append({
            **m, "url": ex_url, "exhibit": fname, "title": ex_title,
            "body": body, "body_chars": body_chars,
        })
    return last_http_ts


def _derive_title(meta: dict, body: str) -> str:
    """Short human-readable title for a filing.

    Examples:
      "8-K (2.02,9.01): results of operations + exhibits"
      "10-Q quarterly report"
      "8-K (5.02): officer/director departure"
    """
    form = meta["form_type"]
    items_raw = meta.get("items") or ""
    if not items_raw or form != "8-K":
        if form == "10-Q":
            return "10-Q quarterly report"
        if form == "10-K":
            return "10-K annual report"
        return form
    item_descs: list[str] = []
    for code in [c.strip() for c in items_raw.split(",") if c.strip()]:
        if code in _RESET_ITEM_CODES:
            item_descs.append(_RESET_ITEM_CODES[code])
        elif code in _EARNINGS_ITEM_CODES:
            item_descs.append(_EARNINGS_ITEM_CODES[code])
        else:
            item_descs.append(f"item {code}")
    return f"8-K ({items_raw}): {' + '.join(item_descs)}"


__all__ = ["fetch_filings_for_symbol"]

if False:  # quiet linter — json is reserved for future title-extraction
    json
