import { useEffect, useMemo, useState } from 'react';
import { CalendarDays } from 'lucide-react';
import { useUniverses } from '@/hooks/useUniverses';
import { useTopPicks } from '@/hooks/useTopPicks';
import type { Direction, TopPick } from '@/api/types';
import { UniverseSelector } from '@/components/UniverseSelector';
import { DirectionToggle } from '@/components/DirectionToggle';
import { PicksTable } from '@/components/PicksTable';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { ErrorMessage } from '@/components/ErrorMessage';
import { EmptyState } from '@/components/EmptyState';

const HIGH_CONF_THRESHOLD = 0.5;

function pickHighConfidence(picks: TopPick[], direction: Direction): TopPick[] {
  const field: keyof TopPick =
    direction === 'long' ? 'top_quintile_proba' : 'bottom_quintile_proba';
  return picks
    .filter((p) => {
      const v = p[field];
      return typeof v === 'number' && v > HIGH_CONF_THRESHOLD;
    })
    .sort((a, b) => Number(b[field] ?? 0) - Number(a[field] ?? 0));
}

export function DashboardPage() {
  const universesQ = useUniverses();
  const universes = useMemo(() => universesQ.data ?? [], [universesQ.data]);

  const [universe, setUniverse] = useState<string>('');
  const [direction, setDirection] = useState<Direction>('long');

  // Lock selection to the first universe once we have data.
  useEffect(() => {
    if (!universe && universes.length > 0) {
      setUniverse(universes[0].name);
    }
  }, [universe, universes]);

  const picksQ = useTopPicks(
    { universe, direction, limit: 20 },
    Boolean(universe),
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-4 rounded border border-gray-200 bg-white p-3">
        <UniverseSelector
          value={universe}
          onChange={setUniverse}
          universes={universes}
          loading={universesQ.isLoading}
        />
        <DirectionToggle value={direction} onChange={setDirection} />
        {picksQ.data?.as_of ? (
          <div className="ml-auto flex items-center gap-1 text-sm text-gray-600">
            <CalendarDays className="h-4 w-4" />
            <span className="font-mono">{picksQ.data.as_of}</span>
          </div>
        ) : null}
      </div>

      {universesQ.isError ? (
        <ErrorMessage error={universesQ.error} onRetry={() => universesQ.refetch()} />
      ) : null}

      {universesQ.isLoading ? <LoadingSpinner label="Loading universes…" /> : null}

      {!universesQ.isLoading && universes.length === 0 ? (
        <EmptyState
          title="No universes loaded yet"
          hint={
            <span>
              Run{' '}
              <code className="rounded bg-gray-100 px-1 font-mono">
                python -m scripts.refresh_universes
              </code>{' '}
              to populate the membership table.
            </span>
          }
        />
      ) : null}

      {universe ? <PicksSection universe={universe} direction={direction} /> : null}
    </div>
  );
}

function PicksSection({ universe, direction }: { universe: string; direction: Direction }) {
  const picksQ = useTopPicks({ universe, direction, limit: 20 });

  if (picksQ.isLoading) return <LoadingSpinner label="Loading picks…" />;
  if (picksQ.isError) {
    return <ErrorMessage error={picksQ.error} onRetry={() => picksQ.refetch()} />;
  }
  const data = picksQ.data;
  if (!data || data.picks.length === 0) {
    return (
      <EmptyState
        title="No predictions for this universe yet"
        hint={
          <span>
            Run{' '}
            <code className="rounded bg-gray-100 px-1 font-mono">
              python -m jobs.daily_predict
            </code>{' '}
            to populate predictions for the most recent trading day.
          </span>
        }
      />
    );
  }

  const highConf = pickHighConfidence(data.picks, direction);

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <Section
        title={direction === 'long' ? 'Top long picks' : 'Top short picks'}
        subtitle={`Sorted by predicted 5-day return ${direction === 'long' ? 'descending' : 'ascending'}.`}
      >
        <PicksTable universe={universe} direction={direction} picks={data.picks} />
      </Section>
      <Section
        title={
          direction === 'long' ? 'High-confidence long' : 'High-confidence short'
        }
        subtitle={`Filtered by ${
          direction === 'long' ? 'top' : 'bottom'
        }-quintile probability > ${(HIGH_CONF_THRESHOLD * 100).toFixed(0)}%.`}
      >
        {highConf.length ? (
          <PicksTable universe={universe} direction={direction} picks={highConf} />
        ) : (
          <EmptyState
            title="No high-confidence picks"
            hint="Try the other direction or wait for tomorrow's predictions."
          />
        )}
      </Section>
    </div>
  );
}

function Section({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <section className="space-y-2">
      <header>
        <h2 className="text-base font-semibold text-gray-900">{title}</h2>
        <p className="text-xs text-gray-500">{subtitle}</p>
      </header>
      {children}
    </section>
  );
}
