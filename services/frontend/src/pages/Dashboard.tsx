import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, CalendarDays, TrendingDown, TrendingUp } from 'lucide-react';
import { Link } from 'react-router-dom';
import { useUniverses } from '@/hooks/useUniverses';
import { useTopPicks } from '@/hooks/useTopPicks';
import { useStockOhlcv } from '@/hooks/useStockOhlcv';
import { useNewsVerdicts } from '@/hooks/useNewsVerdicts';
import type { NewsVerdict, TopPick } from '@/api/types';
import { UniverseSelector } from '@/components/UniverseSelector';
import { Sparkline } from '@/components/Sparkline';
import { VerdictChip } from '@/components/VerdictChip';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { ErrorMessage } from '@/components/ErrorMessage';
import { EmptyState } from '@/components/EmptyState';

const PICKS_PER_SIDE = 10;
// Pull a wider candidate pool so strict-mode filtering still leaves enough.
const CANDIDATE_LIMIT = 50;

function pctFmt(value: number, decimals = 2): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/** Direction-agreement: how much more the model thinks this stock is in the
 *  desired tail than the opposite tail.
 *  Long: top_q − bot_q  (positive when model agrees this is a top pick)
 *  Short: bot_q − top_q  (positive when model agrees this is a bottom pick)
 *  Range roughly [-0.6, +0.7]. Random baseline = 0.
 */
function directionAgreement(pick: TopPick, direction: 'long' | 'short'): number {
  const top = pick.top_quintile_proba ?? 0.2;
  const bot = pick.bottom_quintile_proba ?? 0.2;
  return direction === 'long' ? top - bot : bot - top;
}

/** Combined score: rewards both magnitude of expected return AND classifier
 *  agreement on direction. score = predicted_return × (1 + direction_agreement).
 *  - Both signals agree strongly → score amplified
 *  - Signals disagree (negative direction_agreement) → score shrinks toward 0
 *  Used to re-rank picks after strict-mode filtering.
 *
 *  For longs (predicted > 0), we want HIGHER score → better pick.
 *  For shorts (predicted < 0), we want LOWER (more negative) score → better short.
 */
function combinedScore(pick: TopPick, direction: 'long' | 'short'): number {
  const ret = pick.predicted_return_5d;
  const agreement = directionAgreement(pick, direction);
  // For shorts, agreement is bot_q − top_q which is positive when model
  // agrees with shortness; multiplying a negative return by (1 + positive)
  // makes it more negative, which is what we want.
  if (direction === 'long') {
    return ret * (1 + agreement);
  } else {
    // For shorts, formula stays the same but we re-derive `agreement` against
    // the short direction (already done in directionAgreement).
    return ret * (1 + agreement);
  }
}

/** Sign mismatch between the regression head and the classifier:
 *  - Long pick where model thinks it's MORE likely bottom-quintile
 *  - Short pick where model thinks it's MORE likely top-quintile
 *  These are amber-flag picks even if the direction-agreement bar is short.
 */
function isMismatch(pick: TopPick, direction: 'long' | 'short'): boolean {
  return directionAgreement(pick, direction) < 0;
}

/** Strict-mode filter: keep only picks whose predicted return aligns with
 *  the requested direction. Drops the "least bearish" pickup as a long. */
function strictFilter(picks: TopPick[], direction: 'long' | 'short'): TopPick[] {
  return picks.filter((p) =>
    direction === 'long' ? p.predicted_return_5d > 0 : p.predicted_return_5d < 0,
  );
}

export function DashboardPage() {
  const universesQ = useUniverses();
  const universes = useMemo(() => universesQ.data ?? [], [universesQ.data]);
  const [universe, setUniverse] = useState<string>('');

  useEffect(() => {
    if (!universe && universes.length > 0) setUniverse(universes[0].name);
  }, [universe, universes]);

  // Pull a wider candidate pool so strict-mode + re-ranking has enough to work with.
  const longsQ = useTopPicks({ universe, direction: 'long', limit: CANDIDATE_LIMIT }, Boolean(universe));
  const shortsQ = useTopPicks({ universe, direction: 'short', limit: CANDIDATE_LIMIT }, Boolean(universe));
  // LLM verdicts for today's long picks (audit-only — does not drive trading).
  const verdictsQ = useNewsVerdicts(universe, longsQ.data?.as_of);
  const verdictBySymbol = useMemo(() => {
    const m = new Map<string, NewsVerdict>();
    for (const v of verdictsQ.data?.verdicts ?? []) m.set(v.symbol, v);
    return m;
  }, [verdictsQ.data]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-4 rounded-lg border border-gray-800 bg-gray-900/60 p-4">
        <UniverseSelector
          value={universe}
          onChange={setUniverse}
          universes={universes}
          loading={universesQ.isLoading}
        />
        {longsQ.data?.as_of ? (
          <div className="ml-auto flex items-center gap-2 text-sm text-gray-400">
            <CalendarDays className="h-4 w-4 text-gray-500" />
            <span className="font-mono">{longsQ.data.as_of}</span>
          </div>
        ) : null}
      </div>

      {universesQ.isError ? (
        <ErrorMessage error={universesQ.error} onRetry={() => universesQ.refetch()} />
      ) : null}

      {universesQ.isLoading ? <LoadingSpinner label="Loading universes…" /> : null}

      {universe ? (
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
          <PicksColumn
            universe={universe}
            direction="long"
            data={longsQ.data?.picks ?? []}
            isLoading={longsQ.isLoading}
            isError={longsQ.isError}
            error={longsQ.error}
            verdictBySymbol={verdictBySymbol}
          />
          <PicksColumn
            universe={universe}
            direction="short"
            data={shortsQ.data?.picks ?? []}
            isLoading={shortsQ.isLoading}
            isError={shortsQ.isError}
            error={shortsQ.error}
            verdictBySymbol={verdictBySymbol}
          />
        </div>
      ) : null}
    </div>
  );
}

function PicksColumn({
  universe, direction, data, isLoading, isError, error, verdictBySymbol,
}: {
  universe: string;
  direction: 'long' | 'short';
  data: TopPick[];
  isLoading: boolean;
  isError: boolean;
  error: unknown;
  /** LLM verdicts keyed by symbol. Same map for both columns; longs and
   *  shorts are disjoint by sign of predicted_return so they never collide. */
  verdictBySymbol?: Map<string, NewsVerdict>;
}) {
  const colorClass = direction === 'long' ? 'text-emerald-400' : 'text-rose-400';
  const Icon = direction === 'long' ? TrendingUp : TrendingDown;

  // 1) Strict mode: drop picks that don't align with direction's sign.
  // 2) Re-sort the survivors by combined_score (rewards agreement).
  // 3) Re-rank 1..N. Cap at PICKS_PER_SIDE.
  const ranked = useMemo(() => {
    const filtered = strictFilter(data, direction);
    const sorted = [...filtered].sort((a, b) => {
      const sa = combinedScore(a, direction);
      const sb = combinedScore(b, direction);
      // For longs, descending; for shorts, ascending (most-negative = best short).
      return direction === 'long' ? sb - sa : sa - sb;
    });
    return sorted.slice(0, PICKS_PER_SIDE).map((pick, i) => ({
      ...pick,
      rank: i + 1,
    }));
  }, [data, direction]);

  // No-actionable-signal empty state for strict mode.
  const hasUniverseData = data.length > 0;
  const noActionable = hasUniverseData && ranked.length === 0;

  return (
    <section className="space-y-3">
      <header className="flex items-baseline justify-between">
        <h2 className={`flex items-center gap-2 text-base font-semibold ${colorClass}`}>
          <Icon className="h-5 w-5" />
          Top {direction} picks
        </h2>
        <p className="text-xs text-gray-500">
          {ranked.length > 0 ? `${ranked.length} actionable` : ''}
        </p>
      </header>

      {isLoading ? <LoadingSpinner label="Loading picks…" /> : null}
      {isError ? <ErrorMessage error={error} /> : null}

      {!isLoading && !isError && !hasUniverseData ? (
        <EmptyState title="No predictions yet" hint="Run jobs.daily_predict." />
      ) : null}

      {noActionable ? (
        <EmptyState
          title={`No actionable ${direction} signal today`}
          hint={
            direction === 'long'
              ? 'Model has zero stocks with positive expected return. Market view is bearish today.'
              : 'Model has zero stocks with negative expected return. Market view is bullish today.'
          }
        />
      ) : null}

      <div className="space-y-2">
        {ranked.map((pick) => (
          <PickCard
            key={pick.symbol}
            pick={pick}
            universe={universe}
            direction={direction}
            verdict={verdictBySymbol?.get(pick.symbol)}
          />
        ))}
      </div>
    </section>
  );
}

function PickCard({ pick, universe, direction, verdict }: {
  pick: TopPick;
  universe: string;
  direction: 'long' | 'short';
  verdict?: NewsVerdict;
}) {
  // Pull last 30 trading days of OHLCV for the sparkline.
  const today = new Date();
  const start = new Date(today);
  start.setDate(start.getDate() - 60);
  const ohlcvQ = useStockOhlcv(
    pick.symbol,
    start.toISOString().slice(0, 10),
    today.toISOString().slice(0, 10),
  );
  const sparkData = useMemo(() => {
    const bars = ohlcvQ.data?.bars ?? [];
    return bars.slice(-30).map((b) => ({ value: b.close }));
  }, [ohlcvQ.data]);

  const ret = pick.predicted_return_5d;
  const agreement = directionAgreement(pick, direction);
  const mismatch = isMismatch(pick, direction);
  const isPos = ret >= 0;
  const isLong = direction === 'long';

  // Card accent always reflects the column direction (so users can scan Long/Short visually).
  const accentColor = isLong ? 'text-emerald-400' : 'text-rose-400';
  const ringColor = isLong ? 'ring-emerald-500/20' : 'ring-rose-500/20';
  const bgColor = isLong ? 'bg-emerald-500/[0.04]' : 'bg-rose-500/[0.04]';

  // Direction agreement bar:
  //  - Map [-0.4, +0.4] -> [0%, 100%]
  //  - Color RED when agreement < 0 (model disagrees with the requested direction)
  //  - Color GREEN/RED matching the column otherwise
  const agreementBarPct = Math.max(0, Math.min(100, ((agreement + 0.4) / 0.8) * 100));
  const agreementBarColor =
    agreement < 0 ? 'bg-amber-500/80'
    : isLong ? 'bg-emerald-500/80'
    : 'bg-rose-500/80';
  const agreementValueColor =
    agreement < 0 ? 'text-amber-400'
    : isLong ? 'text-emerald-400'
    : 'text-rose-400';

  return (
    <Link
      to={`/stocks/${universe}/${encodeURIComponent(pick.symbol)}`}
      className={[
        'block rounded-lg border border-gray-800/80 px-4 py-3 transition-all',
        'hover:border-gray-700 hover:bg-gray-900/80 ring-1',
        ringColor,
        bgColor,
      ].join(' ')}
    >
      <div className="flex items-center gap-4">
        {/* Symbol + name + LLM verdict chip */}
        <div className="min-w-[140px]">
          <div className="flex items-center gap-2">
            <div className="font-mono text-sm font-semibold text-gray-100">
              #{pick.rank} {pick.symbol}
            </div>
            {verdict ? <VerdictChip verdict={verdict} /> : null}
          </div>
          <div className="truncate text-xs text-gray-500" title={pick.company_name ?? ''}>
            {pick.company_name ?? '—'}
          </div>
        </div>

        {/* Sparkline */}
        <div className="w-[120px] flex-shrink-0">
          <Sparkline data={sparkData} positive={isPos} height={32} />
        </div>

        {/* Predicted return */}
        <div className="ml-auto text-right">
          <div className={`font-mono text-base font-semibold ${accentColor}`}>
            {ret >= 0 ? '+' : ''}{pctFmt(ret)}
          </div>
          <div className="text-[10px] uppercase tracking-wider text-gray-500">
            5d expected
          </div>
        </div>

        {/* Direction agreement bar with sign-aware color + amber warning */}
        <div className="w-[160px] flex-shrink-0">
          <div className="flex items-baseline justify-between">
            <span className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-gray-500">
              dir. agreement
              {mismatch ? (
                <AlertTriangle
                  className="h-3 w-3 text-amber-400"
                  aria-label="Regression and classifier disagree on direction"
                />
              ) : null}
            </span>
            <span className={`font-mono text-xs ${agreementValueColor}`}>
              {agreement >= 0 ? '+' : ''}{pctFmt(agreement, 1)}
            </span>
          </div>
          <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-gray-800">
            <div
              className={`h-full ${agreementBarColor}`}
              style={{ width: `${agreementBarPct}%` }}
            />
          </div>
        </div>
      </div>
    </Link>
  );
}
