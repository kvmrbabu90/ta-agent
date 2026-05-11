import { useEffect, useMemo, useState } from 'react';
import { CalendarDays, TrendingDown, TrendingUp } from 'lucide-react';
import { Link } from 'react-router-dom';
import { useUniverses } from '@/hooks/useUniverses';
import { useTopPicks } from '@/hooks/useTopPicks';
import { useStockOhlcv } from '@/hooks/useStockOhlcv';
import type { TopPick } from '@/api/types';
import { UniverseSelector } from '@/components/UniverseSelector';
import { Sparkline } from '@/components/Sparkline';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { ErrorMessage } from '@/components/ErrorMessage';
import { EmptyState } from '@/components/EmptyState';

const PICKS_PER_SIDE = 10;

function pctFmt(value: number, decimals = 2): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/** Convert quintile probabilities into a single "net confidence" score in [-1, 1].
 *  Long-bias: top-quintile probability (anchored at 0.2 = random); negative if
 *  the model says it's MORE likely to be bottom-quintile than top.
 */
function netConfidence(pick: TopPick, direction: 'long' | 'short'): number {
  const top = pick.top_quintile_proba ?? 0.2;
  const bot = pick.bottom_quintile_proba ?? 0.2;
  return direction === 'long' ? top - bot : bot - top;
}

export function DashboardPage() {
  const universesQ = useUniverses();
  const universes = useMemo(() => universesQ.data ?? [], [universesQ.data]);
  const [universe, setUniverse] = useState<string>('');

  useEffect(() => {
    if (!universe && universes.length > 0) setUniverse(universes[0].name);
  }, [universe, universes]);

  const longsQ = useTopPicks({ universe, direction: 'long', limit: PICKS_PER_SIDE }, Boolean(universe));
  const shortsQ = useTopPicks({ universe, direction: 'short', limit: PICKS_PER_SIDE }, Boolean(universe));

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
          />
          <PicksColumn
            universe={universe}
            direction="short"
            data={shortsQ.data?.picks ?? []}
            isLoading={shortsQ.isLoading}
            isError={shortsQ.isError}
            error={shortsQ.error}
          />
        </div>
      ) : null}
    </div>
  );
}

function PicksColumn({
  universe, direction, data, isLoading, isError, error,
}: {
  universe: string;
  direction: 'long' | 'short';
  data: TopPick[];
  isLoading: boolean;
  isError: boolean;
  error: unknown;
}) {
  const colorClass = direction === 'long' ? 'text-emerald-400' : 'text-rose-400';
  const Icon = direction === 'long' ? TrendingUp : TrendingDown;

  return (
    <section className="space-y-3">
      <header className="flex items-baseline justify-between">
        <h2 className={`flex items-center gap-2 text-base font-semibold ${colorClass}`}>
          <Icon className="h-5 w-5" />
          Top {direction} picks
        </h2>
        <p className="text-xs text-gray-500">
          {data.length > 0 ? `${data.length} predictions` : ''}
        </p>
      </header>

      {isLoading ? <LoadingSpinner label="Loading picks…" /> : null}
      {isError ? <ErrorMessage error={error} /> : null}
      {!isLoading && !isError && data.length === 0 ? (
        <EmptyState title="No predictions yet" hint="Run jobs.daily_predict." />
      ) : null}

      <div className="space-y-2">
        {data.map((pick) => (
          <PickCard key={pick.symbol} pick={pick} universe={universe} direction={direction} />
        ))}
      </div>
    </section>
  );
}

function PickCard({ pick, universe, direction }: {
  pick: TopPick;
  universe: string;
  direction: 'long' | 'short';
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
  const conf = netConfidence(pick, direction);
  const isPos = ret >= 0;
  const isLong = direction === 'long';
  const accentColor = isLong ? 'text-emerald-400' : 'text-rose-400';
  const ringColor = isLong ? 'ring-emerald-500/20' : 'ring-rose-500/20';
  const bgColor = isLong ? 'bg-emerald-500/[0.04]' : 'bg-rose-500/[0.04]';

  // Confidence bar: scale [-0.4, +0.4] -> [0%, 100%]
  const confBarPct = Math.max(0, Math.min(100, ((conf + 0.4) / 0.8) * 100));

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
        {/* Symbol + name (3 cols on grid, but flex here) */}
        <div className="min-w-[100px]">
          <div className="font-mono text-sm font-semibold text-gray-100">
            #{pick.rank} {pick.symbol}
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

        {/* Net confidence bar (compact) */}
        <div className="w-[140px] flex-shrink-0">
          <div className="flex items-baseline justify-between">
            <span className="text-[10px] uppercase tracking-wider text-gray-500">
              confidence
            </span>
            <span className={`font-mono text-xs ${accentColor}`}>
              {conf >= 0 ? '+' : ''}{pctFmt(conf, 1)}
            </span>
          </div>
          <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-gray-800">
            <div
              className={isLong ? 'h-full bg-emerald-500/80' : 'h-full bg-rose-500/80'}
              style={{ width: `${confBarPct}%` }}
            />
          </div>
        </div>
      </div>
    </Link>
  );
}
