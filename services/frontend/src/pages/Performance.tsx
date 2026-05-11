import { useEffect, useMemo, useState } from 'react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from 'recharts';
import { useUniverses } from '@/hooks/useUniverses';
import { usePerformance } from '@/hooks/usePerformance';
import { UniverseSelector } from '@/components/UniverseSelector';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { ErrorMessage } from '@/components/ErrorMessage';
import { EmptyState } from '@/components/EmptyState';
import type { PerformanceResponse } from '@/api/types';

const LOOKBACK_OPTIONS = [
  { label: '30d', value: 30 },
  { label: '90d', value: 90 },
  { label: '180d', value: 180 },
  { label: '365d', value: 365 },
];

function pctFmt(value: number | null | undefined, decimals = 2): string {
  if (value == null) return '—';
  return `${(value * 100).toFixed(decimals)}%`;
}
function numFmt(value: number | null | undefined, decimals = 2): string {
  if (value == null) return '—';
  return value.toFixed(decimals);
}

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
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-4 rounded-lg border border-gray-800 bg-gray-900/60 p-4">
        <UniverseSelector
          value={universe}
          onChange={setUniverse}
          universes={universes}
          loading={universesQ.isLoading}
        />
        <div className="flex items-center gap-2">
          <span className="text-xs uppercase tracking-wider text-gray-500">Lookback</span>
          <div className="flex rounded-md bg-gray-800/60 p-0.5">
            {LOOKBACK_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                onClick={() => setLookback(opt.value)}
                className={[
                  'rounded px-2.5 py-1 text-xs font-medium transition-colors',
                  lookback === opt.value
                    ? 'bg-emerald-500/20 text-emerald-300'
                    : 'text-gray-400 hover:text-gray-100',
                ].join(' ')}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {perfQ.isLoading ? <LoadingSpinner label="Computing metrics…" /> : null}
      {perfQ.isError ? <ErrorMessage error={perfQ.error} onRetry={() => perfQ.refetch()} /> : null}
      {perfQ.data && perfQ.data.n_predictions === 0 ? (
        <EmptyState
          title="No predictions in window"
          hint="Run jobs.daily_predict to populate the predictions log."
        />
      ) : null}

      {perfQ.data && perfQ.data.n_predictions > 0 ? (
        <PerformanceContent data={perfQ.data} />
      ) : null}
    </div>
  );
}

function PerformanceContent({ data }: { data: PerformanceResponse }) {
  return (
    <>
      <section className="space-y-3">
        <h2 className="text-base font-semibold text-gray-100">
          Risk-adjusted performance — long-short decile spread vs SPY
        </h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <MetricCard
            label="Sharpe (strategy)"
            value={numFmt(data.sharpe_ratio)}
            hint="annualized, RF=0"
            tone={(data.sharpe_ratio ?? 0) >= 0 ? 'pos' : 'neg'}
          />
          <MetricCard
            label="Sharpe (SPY)"
            value={numFmt(data.spy_sharpe_ratio)}
            hint="annualized, RF=0"
            tone={(data.spy_sharpe_ratio ?? 0) >= 0 ? 'pos' : 'neg'}
          />
          <MetricCard
            label="Sortino (strategy)"
            value={numFmt(data.sortino_ratio)}
            hint="downside-vol denom"
            tone={(data.sortino_ratio ?? 0) >= 0 ? 'pos' : 'neg'}
          />
          <MetricCard
            label="Sortino (SPY)"
            value={numFmt(data.spy_sortino_ratio)}
            hint="downside-vol denom"
            tone={(data.spy_sortino_ratio ?? 0) >= 0 ? 'pos' : 'neg'}
          />
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="text-base font-semibold text-gray-100">Model-quality metrics</h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <MetricCard
            label="Directional accuracy"
            value={pctFmt(data.directional_accuracy, 1)}
            hint={`excl. neutral · n=${data.n_directional_observations ?? 0}`}
            tone={(data.directional_accuracy ?? 0) > 0.5 ? 'pos' : 'neg'}
          />
          <MetricCard
            label="Hit rate (all)"
            value={pctFmt(data.hit_rate, 1)}
            hint={`includes neutral · n=${data.n_settled}`}
            tone={(data.hit_rate ?? 0) > 0.5 ? 'pos' : 'neg'}
          />
          <MetricCard
            label="Mean daily IC"
            value={numFmt(data.mean_daily_ic, 4)}
            hint={`t-stat ${numFmt(data.ic_t_stat, 1)}`}
            tone={(data.mean_daily_ic ?? 0) > 0 ? 'pos' : 'neg'}
          />
          <MetricCard
            label="Decile spread (5d)"
            value={pctFmt(data.decile_spread_5d, 2)}
            hint="avg top10 − bot10 realized"
            tone={(data.decile_spread_5d ?? 0) > 0 ? 'pos' : 'neg'}
          />
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="text-base font-semibold text-gray-100">
          Cumulative return — strategy vs SPY
        </h2>
        <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-4">
          {data.equity_curve.length > 0 ? (
            <div className="h-[320px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={data.equity_curve.map((p) => ({
                    bar_date: p.bar_date,
                    Strategy: (p.cum_strategy_return ?? 0) * 100,
                    SPY: p.cum_spy_return == null ? null : p.cum_spy_return * 100,
                  }))}
                  margin={{ top: 5, right: 16, bottom: 5, left: 8 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="bar_date" tick={{ fontSize: 11 }} />
                  <YAxis
                    tick={{ fontSize: 11 }}
                    tickFormatter={(v) => `${v.toFixed(1)}%`}
                  />
                  <Tooltip
                    formatter={(v: number) => `${v.toFixed(2)}%`}
                    labelFormatter={(d) => `Date: ${d}`}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="Strategy"
                    stroke="#34d399"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="SPY"
                    stroke="#94a3b8"
                    strokeWidth={2}
                    strokeDasharray="4 3"
                    dot={false}
                    isAnimationActive={false}
                    connectNulls
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyState
              title="Equity curve not available"
              hint="Need ≥10 settled predictions per day for the long-short decile spread."
            />
          )}
        </div>
      </section>

      <div className="text-xs text-gray-500">
        {data.n_predictions.toLocaleString()} predictions logged · {data.n_settled.toLocaleString()} settled · {data.lookback_days}d lookback
      </div>
    </>
  );
}

function MetricCard({ label, value, hint, tone }: {
  label: string;
  value: string;
  hint?: string;
  tone?: 'pos' | 'neg' | 'neutral';
}) {
  const valueColor =
    tone === 'pos' ? 'text-emerald-400' :
    tone === 'neg' ? 'text-rose-400' :
    'text-gray-100';
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900/60 px-4 py-3">
      <div className="text-[11px] uppercase tracking-wider text-gray-500">{label}</div>
      <div className={`mt-1 font-mono text-2xl font-semibold ${valueColor}`}>{value}</div>
      {hint ? <div className="text-[11px] text-gray-500">{hint}</div> : null}
    </div>
  );
}
