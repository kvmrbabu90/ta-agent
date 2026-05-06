import { useEffect, useMemo, useState } from 'react';
import { CalibrationTable } from '@/components/CalibrationTable';
import { ErrorMessage } from '@/components/ErrorMessage';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { PerformanceChart } from '@/components/PerformanceChart';
import { UniverseSelector } from '@/components/UniverseSelector';
import { useUniverses } from '@/hooks/useUniverses';
import { usePerformance } from '@/hooks/usePerformance';
import { formatNumber, formatPercent, formatInteger } from '@/utils/format';
import type { PerformanceResponse } from '@/api/types';

const LOOKBACK_OPTIONS = [30, 90, 180, 365];

export function PerformancePage() {
  const universesQ = useUniverses();
  const universes = useMemo(() => universesQ.data ?? [], [universesQ.data]);

  const [universe, setUniverse] = useState<string>('');
  const [lookback, setLookback] = useState<number>(90);

  useEffect(() => {
    if (!universe && universes.length > 0) setUniverse(universes[0].name);
  }, [universe, universes]);

  const perfQ = usePerformance(universe, lookback);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-4 rounded border border-gray-200 bg-white p-3">
        <UniverseSelector
          value={universe}
          onChange={setUniverse}
          universes={universes}
          loading={universesQ.isLoading}
        />
        <label className="flex items-center gap-2 text-sm text-gray-700">
          <span className="font-medium">Lookback</span>
          <select
            className="rounded border border-gray-300 bg-white px-2 py-1 text-sm focus:border-blue-500 focus:outline-none"
            value={lookback}
            onChange={(e) => setLookback(Number(e.target.value))}
          >
            {LOOKBACK_OPTIONS.map((n) => (
              <option key={n} value={n}>
                {n} days
              </option>
            ))}
          </select>
        </label>
      </div>

      {perfQ.isLoading ? <LoadingSpinner label="Loading performance…" /> : null}
      {perfQ.isError ? (
        <ErrorMessage error={perfQ.error} onRetry={() => perfQ.refetch()} />
      ) : null}
      {perfQ.data ? <PerformanceContent data={perfQ.data} /> : null}
    </div>
  );
}

function PerformanceContent({ data }: { data: PerformanceResponse }) {
  return (
    <>
      <KpiGrid data={data} />

      <section className="space-y-2 rounded border border-gray-200 bg-white p-4">
        <header>
          <h2 className="text-base font-semibold text-gray-900">Daily IC over time</h2>
          <p className="text-xs text-gray-500">
            Cross-sectional Pearson correlation of predicted vs realized 5-day return on each
            settled day. Persistent values above zero are the signal we want to see.
          </p>
        </header>
        <PerformanceChart series={data.ic_timeseries} />
      </section>

      <section className="space-y-2 rounded border border-gray-200 bg-white p-4">
        <header>
          <h2 className="text-base font-semibold text-gray-900">
            Classifier calibration
          </h2>
          <p className="text-xs text-gray-500">
            For each predicted top-quintile probability bucket, the share of those predictions
            that did land in the realized top quintile.
          </p>
        </header>
        <CalibrationTable buckets={data.calibration} />
      </section>
    </>
  );
}

function KpiGrid({ data }: { data: PerformanceResponse }) {
  const kpis: Array<{ label: string; value: string; hint?: string }> = [
    {
      label: 'Mean daily IC',
      value: formatNumber(data.mean_daily_ic, 4),
      hint: 'Higher = predictions correlate more with realized returns.',
    },
    {
      label: 'IC t-stat',
      value: formatNumber(data.ic_t_stat, 2),
      hint: '> 2 is the rough threshold for statistical significance.',
    },
    {
      label: 'Hit rate',
      value: formatPercent(data.hit_rate),
      hint: 'Fraction where predicted direction matched realized.',
    },
    {
      label: 'Decile spread (5d)',
      value: formatPercent(data.decile_spread_5d, 2),
      hint: 'Top decile mean minus bottom decile mean of realized return.',
    },
    {
      label: 'Predictions',
      value: formatInteger(data.n_predictions),
      hint: `${formatInteger(data.n_settled)} settled.`,
    },
  ];

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
      {kpis.map((k) => (
        <div
          key={k.label}
          className="rounded border border-gray-200 bg-white p-3"
        >
          <div className="text-xs uppercase tracking-wide text-gray-500">{k.label}</div>
          <div className="mt-1 text-xl font-mono text-gray-900">{k.value}</div>
          {k.hint ? <div className="mt-1 text-[11px] text-gray-500">{k.hint}</div> : null}
        </div>
      ))}
    </div>
  );
}
