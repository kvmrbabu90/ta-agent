"""Local-LLM news classifier (Gemma 4 via Ollama).

The classifier reads a stock's recent SEC filings + trailing returns and
returns a verdict for whether a recent decline is more likely a temporary
panic (mean-revert candidate) or a structural reset.

Design notes:
  - We pin `think=False`. Gemma 4 spends its entire token budget reasoning
    silently otherwise; `done_reason='length'` with empty content. Even
    a 400-token budget is exhausted by the thinking trace.
  - `format='json'` constrains output. We still parse defensively and
    coerce common drift cases (`confidence: "High"` → 0.85 etc).
  - The prompt explicitly forbids using external knowledge:
    "Use ONLY the provided sources." This reduces hallucination on
    factual claims (e.g. inventing M&A deals that didn't happen).
  - One pass per pick — caller decides how many top picks to classify.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import ollama

from packages.common.config import settings
from packages.common.logging import log

DEFAULT_MODEL = "gemma4:latest"
DEFAULT_HOST = "http://127.0.0.1:11434"

# Generation budget. Gemma 4 in non-think mode emits ~80-200 tokens for
# our prompt; 600 gives plenty of headroom without runaway.
_NUM_PREDICT = 600

# Truncate per-filing body before stuffing into the prompt. Multiple
# filings × 8000 chars each would blow the model's context. We give each
# filing a fixed budget so total prompt stays bounded.
# 3000 chars × ~5 filings ≈ 15k chars ≈ 4k tokens of context — well
# inside Gemma 4's window.
_PER_FILING_BODY_CHARS = 3000


VALID_VERDICTS = {"PANIC", "RESET", "HYPE", "STRENGTH", "UNCLEAR"}

# Verdicts our strategy treats as "agree with the pick" (KEEP):
#   long  → PANIC    (sentiment-driven decline → mean-revert candidate)
#   short → HYPE     (sentiment-driven rally   → fade candidate)
KEEP_VERDICTS = {"PANIC", "HYPE"}
# Verdicts our strategy treats as "contradict the pick" (AVOID):
#   long  → RESET    (real bad news → not a bounce candidate)
#   short → STRENGTH (real good news → not a fade candidate)
AVOID_VERDICTS = {"RESET", "STRENGTH"}


_LONG_SYSTEM = """You are a financial event classifier. You read recent SEC filings for a stock that has DECLINED sharply over the past 5-20 trading days. Your job is to classify whether the decline is more likely a temporary overreaction (PANIC: mean-revert candidate, keep the long) or a structural reset (RESET: avoid the long).

Classification rubric:
- PANIC: sector/macro spillover, analyst chatter, sentiment-driven, NO material structural news in the filings provided
- RESET: any of — earnings miss combined with guidance cut, going-concern doubt, accounting restatement / non-reliance on prior financials, CEO/CFO departure, dividend cut, auditor change, delisting notice, bankruptcy/receivership, material impairment, fraud or SEC enforcement action, large goodwill writedown, strategic review / "exploring alternatives", senior debt covenant breach
- UNCLEAR: insufficient information in the provided filings to decide

Strict rules:
1. Use ONLY the provided filings. Do not invent facts. If a claim is not in a provided source, do not assert it.
2. An earnings release (8-K with item 2.02) is NOT automatically a RESET. Read the body. If guidance was reaffirmed or the miss was small, that's PANIC or UNCLEAR.
3. Cite which numbered source(s) support your verdict in `evidence_sources`. Cite by source number (e.g. "1", "3").
4. confidence is a float in [0,1] reflecting how strongly the filings support the verdict. Never use words like "High"/"Medium".

Output JSON only. No prose."""


_SHORT_SYSTEM = """You are a financial event classifier. You read recent SEC filings for a stock that has RALLIED sharply over the past 5-20 trading days. Your job is to classify whether the rally is more likely temporary overreaction (HYPE: fade candidate, keep the short) or driven by real fundamental strength (STRENGTH: avoid the short).

Classification rubric:
- HYPE: sector/macro tailwind spillover, analyst upgrades on revisions only, social-media or short-squeeze dynamics, NO substantive positive structural news in the filings provided
- STRENGTH: any of — earnings beat combined with guidance raise, large new contract / customer win disclosed, FDA approval or breakthrough designation, M&A announced where this company is the target at a premium, large buyback or dividend increase, debt refinanced at materially better terms, senior management hire signaling turnaround, divestiture of underperforming segment at premium, regulatory tailwind disclosed
- UNCLEAR: insufficient information in the provided filings to decide

Strict rules:
1. Use ONLY the provided filings. Do not invent facts. If a claim is not in a provided source, do not assert it.
2. An earnings release alone (8-K with item 2.02) is NOT automatically STRENGTH. Read the body. If the beat was small or guidance was merely reaffirmed, that's HYPE or UNCLEAR.
3. Cite which numbered source(s) support your verdict in `evidence_sources`. Cite by source number (e.g. "1", "3").
4. confidence is a float in [0,1] reflecting how strongly the filings support the verdict. Never use words like "High"/"Medium".

Output JSON only. No prose."""


def _system_prompt_for(direction: str) -> str:
    if direction == "short":
        return _SHORT_SYSTEM
    return _LONG_SYSTEM


@dataclass
class Verdict:
    verdict: str  # 'PANIC' | 'RESET' | 'UNCLEAR'
    confidence: float
    key_factors: list[str]
    evidence_sources: list[str]
    raw: str  # original LLM output, for debugging


def _coerce_confidence(raw: Any) -> float:
    """Coerce common drift cases to a [0,1] float."""
    if isinstance(raw, (int, float)):
        v = float(raw)
        return max(0.0, min(1.0, v))
    if isinstance(raw, str):
        s = raw.strip().lower()
        word_map = {
            "very low": 0.15, "low": 0.3, "medium-low": 0.4,
            "medium": 0.5, "moderate": 0.5, "medium-high": 0.65,
            "high": 0.85, "very high": 0.95,
        }
        if s in word_map:
            return word_map[s]
        try:
            v = float(s.rstrip("%"))
            if v > 1:
                v = v / 100.0
            return max(0.0, min(1.0, v))
        except ValueError:
            pass
    return 0.5  # default if unparseable


def _coerce_verdict(raw: Any) -> str:
    if not isinstance(raw, str):
        return "UNCLEAR"
    v = raw.strip().upper()
    if v in VALID_VERDICTS:
        return v
    # Common drift: "panic.", "reset!", "Mean reversion candidate"
    for k in VALID_VERDICTS:
        if k in v:
            return k
    return "UNCLEAR"


def _coerce_str_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x)[:200] for x in raw][:10]
    if isinstance(raw, str):
        return [raw[:200]]
    return [str(raw)[:200]]


def _build_user_prompt(
    symbol: str,
    company_name: str | None,
    trail_5d: float | None,
    trail_20d: float | None,
    predicted_return: float | None,
    filings: list[dict],
    direction: str = "long",
) -> str:
    """Assemble the per-pick user prompt.

    Each filing is rendered as a numbered block. Body is truncated to
    `_PER_FILING_BODY_CHARS` chars to keep total prompt bounded.
    """
    header_lines = [
        f"Symbol: {symbol}",
    ]
    if company_name:
        header_lines.append(f"Company: {company_name}")
    if trail_5d is not None:
        header_lines.append(f"5-day return: {trail_5d * 100:+.2f}%")
    if trail_20d is not None:
        header_lines.append(f"20-day return: {trail_20d * 100:+.2f}%")
    if predicted_return is not None:
        header_lines.append(
            f"Model's 5-day forward prediction: {predicted_return * 100:+.2f}%"
        )

    if not filings:
        filings_block = "(No SEC filings in the provided window.)"
    else:
        chunks = []
        for i, f in enumerate(filings, start=1):
            body = (f.get("body") or "").strip()
            body_clip = body[:_PER_FILING_BODY_CHARS]
            if len(body) > _PER_FILING_BODY_CHARS:
                body_clip += " […truncated…]"
            title = f.get("title") or f["form_type"]
            chunks.append(
                f"Source {i}: {f['filing_date']}  {title}\n"
                f"{body_clip if body_clip else '(body unavailable)'}"
            )
        filings_block = "\n\n".join(chunks)

    if direction == "short":
        verdict_options = "HYPE|STRENGTH|UNCLEAR"
    else:
        verdict_options = "PANIC|RESET|UNCLEAR"
    return (
        "\n".join(header_lines)
        + "\n\nFilings (newest first):\n\n"
        + filings_block
        + "\n\nClassify per the rubric. Return JSON with fields: "
        f"`verdict` ({verdict_options}), `confidence` (float 0-1), "
        "`key_factors` (list of short strings, evidence only), "
        '`evidence_sources` (list of source numbers, e.g. ["1","3"]).'
    )


def classify_decline(
    *,
    symbol: str,
    filings: list[dict],
    company_name: str | None = None,
    trail_5d: float | None = None,
    trail_20d: float | None = None,
    predicted_return: float | None = None,
    direction: str = "long",
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    temperature: float = 0.1,
) -> Verdict:
    """Run the LLM on one pick. Returns a Verdict.

    `direction` switches the rubric:
      'long'  — classify as PANIC (keep) / RESET (avoid) / UNCLEAR
      'short' — classify as HYPE  (keep) / STRENGTH (avoid) / UNCLEAR

    On Ollama/network failure, returns an UNCLEAR verdict with confidence
    0 — callers can treat this as "abstain" (don't filter, don't trust).
    """
    client = ollama.Client(host=host)
    user = _build_user_prompt(
        symbol=symbol, company_name=company_name,
        trail_5d=trail_5d, trail_20d=trail_20d,
        predicted_return=predicted_return, filings=filings,
        direction=direction,
    )
    t0 = time.monotonic()
    try:
        resp = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": _system_prompt_for(direction)},
                {"role": "user", "content": user},
            ],
            think=False,
            format="json",
            options={
                "temperature": temperature,
                "num_predict": _NUM_PREDICT,
            },
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(f"classify_decline({symbol}): LLM call failed: {exc!r}")
        return Verdict(
            verdict="UNCLEAR", confidence=0.0, key_factors=[],
            evidence_sources=[], raw=f"error: {exc!r}",
        )
    elapsed = time.monotonic() - t0
    raw = resp.message.content or ""
    log.info(
        f"classify_decline({symbol}): {elapsed:.1f}s "
        f"done={resp.done_reason} eval={resp.eval_count} "
        f"content_len={len(raw)}"
    )

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        log.warning(f"classify_decline({symbol}): non-JSON output: {raw[:200]!r}")
        return Verdict(
            verdict="UNCLEAR", confidence=0.0, key_factors=[],
            evidence_sources=[], raw=raw,
        )

    return Verdict(
        verdict=_coerce_verdict(parsed.get("verdict")),
        confidence=_coerce_confidence(parsed.get("confidence")),
        key_factors=_coerce_str_list(parsed.get("key_factors")),
        evidence_sources=_coerce_str_list(parsed.get("evidence_sources")),
        raw=raw,
    )


def healthcheck(model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST) -> dict:
    """Quick check that Ollama is reachable and the model is loaded."""
    client = ollama.Client(host=host)
    try:
        models = client.list()
        available = [m.model for m in models.models]
        return {
            "ok": model in available,
            "host": host,
            "model": model,
            "available_models": available,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc), "host": host, "model": model}


# Silence the unused-import linter for `settings`; kept for future tunables.
_ = settings
