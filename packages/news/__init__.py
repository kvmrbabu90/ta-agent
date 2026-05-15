"""News + LLM-based pick filter.

Audit-only by default: the daily pipeline calls `classify_top_picks` after
`daily_predict`, the verdicts land in `news.sqlite`, and the dashboard
shows a chip. The paper-trading engine does NOT consume verdicts yet —
that's deliberate: we accumulate paired (verdict, realized return) data
for a few weeks before deciding whether to act on them.

See `packages/news/pipeline.py` for the main entry point.
"""

from packages.news.classifier import Verdict, classify_decline
from packages.news.pipeline import classify_top_picks
from packages.news.storage import init_news_db

__all__ = ["Verdict", "classify_decline", "classify_top_picks", "init_news_db"]
