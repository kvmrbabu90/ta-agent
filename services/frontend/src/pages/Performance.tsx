import { useEffect, useMemo, useState } from 'react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine,
} from 'recharts';
import { useUniverses } from '@/hooks/useUniverses';
import { useModelInfo, useWalkforward } from '@/hooks/usePerformance';
import { UniverseSelector } from '@/components/UniverseSelector';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { ErrorMessage } from '@/components/ErrorMessage';
import { EmptyState } from '@/components/EmptyState';
import type {
  ModelInfoResponse,
  ModelTargetInfo,
  WalkforwardEquityPoint,
  WalkforwardResponse,
} from '@/api/types';

type Currency = 'USD' | 'INR';

function pctFmt(value: number | null | undefined, decimals = 2): string {
  if (value == null) return '—';
  return `${value.toFixed(decimals)}%`;
}
function numFmt(value: number | null | undefined, decimals = 4): string {
  if (value == null) return '—';
  return value.toFixed(decimals);
}

function moneyFmt(v: number, currency: Currency, d = 0): string {
  if (currency === 'INR') {
    return `₹${v.toLocaleString('en-IN', {
      minimumFractionDigits: d, maximumFractionDigits: d,
    })}`;
  }
  return `$${v.toLocaleString('en-US', {
    minimumFractionDigits: d, maximumFractionDigits: d,
  })}`;
}

export function PerformancePage() {
  const universesQ = useUniverses();
  const universes = useMemo(() => universesQ.data ?? [], [universesQ.data]);
  const [universe, setUniverse] = useState<string>('');

  useEffect(() => {
    if (!universe && universes.length > 0) setUniverse(universes[0].name);
  }, [universe, universes]);

  const modelQ = useModelInfo(universe);
  const wfQ = useWalkforward(universe);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-4 rounded-lg border border-gray-800 bg-gray-900/60 p-4">
        <UniverseSelector
          value={universe}
          onChange={setUniverse}
          universes={universes}
          loading={universesQ.isLoading}
        />
      </div>

      {/* ---- Section 1: Current model information ---- */}
      <section className="space-y-3">
        <h2 className="text-base font-semibold text-gray-100">Current model — {universe || '…'}</h2>
        <p className="text-xs text-gray-500">
          The actual production model that generates today's picks. Trained on the
          full universe history; metrics below come from 5-fold purged walk-forward CV.
        </p>
        {modelQ.isLoading && <LoadingSpinner label="Loading model info…" />}
        {modelQ.isError && <ErrorMessage error={modelQ.error} onRetry={() => modelQ.refetch()} />}
        {modelQ.data && <ModelInfoView data={modelQ.data} />}
      </section>

      {/* ---- Section 2: Tax-adjusted walk-forward ---- */}
      <section className="space-y-3">
        <h2 className="text-base font-semibold text-gray-100">
          Tax-adjusted walk-forward — strategy vs benchmark
        </h2>
        {wfQ.isLoading && <LoadingSpinner label="Loading walk-forward…" />}
        {wfQ.isError && <ErrorMessage error={wfQ.error} onRetry={() => wfQ.refetch()} />}
        {wfQ.data && <WalkforwardView data={wfQ.data} />}
      </section>
    </div>
  );
}

// =============================================================================
// Section 1: Model info
// =============================================================================

function ModelInfoView({ data }: { data: ModelInfoResponse }) {
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Tile label="Universe members" value={String(data.n_members)} />
        <Tile
          label="Training rows"
          value={data.training_rows != null ? data.training_rows.toLocaleString() : '—'}
        />
        <Tile
          label="Training symbols"
          value={data.training_symbols != null ? String(data.training_symbols) : '—'}
        />
        <Tile
          label="Training date range"
          value={
            data.training_date_range
              ? `${data.training_date_range[0]} → ${data.training_date_range[1]}`
              : '—'
          }
        />
      </div>

      {data.targets.length === 0 ? (
        <EmptyState
          title="No trained models found"
          hint={`No metadata.json under data/models/${data.universe}_*`}
        />
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {data.targets.map((t) => (
            <TargetCard key={t.target} target={t} />
          ))}
        </div>
      )}
    </div>
  );
}

function TargetCard({ target }: { target: ModelTargetInfo }) {
  const tgtLabel = target.target.charAt(0).toUpperCase() + target.target.slice(1);
  return (
    <div className="space-y-3 rounded-lg border border-gray-800 bg-gray-900/60 p-4">
      <div className="flex items-baseline justify-between">
        <h3 className="text-sm font-semibold text-gray-100">{tgtLabel} model</h3>
        <code className="text-[11px] text-gray-500">{target.model_id}</code>
      </div>

      <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
        <KV label="Train start" value={target.train_start} />
        <KV label="Train end" value={target.train_end} />
        <KV label="Horizon (days)" value={String(target.horizon_days)} />
        <KV label="Features" value={String(target.n_features)} />
        <KV label="Learning rate" value={numFmt(target.learning_rate, 5)} />
        <KV label="Num leaves" value={String(target.num_leaves ?? '—')} />
        <KV label="Min data/leaf" value={String(target.min_data_in_leaf ?? '—')} />
        <KV label="CV folds" value={String(target.cv_fold_count)} />
      </dl>

      <div className="pt-2 border-t border-gray-800">
        <p className="mb-1 text-[11px] uppercase tracking-wider text-gray-500">CV mean ± std</p>
        <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
          {Object.keys(target.cv_mean_metrics).map((k) => {
            const mean = target.cv_mean_metrics[k];
            const std = target.cv_std_metrics?.[k];
            return (
              <KV
                key={k}
                label={k}
                value={`${numFmt(mean, 4)}${std != null ? ` ± ${numFmt(std, 4)}` : ''}`}
              />
            );
          })}
        </dl>
      </div>
    </div>
  );
}

function Tile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900/60 px-3 py-2">
      <div className="text-[11px] uppercase tracking-wider text-gray-500">{label}</div>
      <div className="mt-1 font-mono text-base text-gray-100">{value}</div>
    </div>
  );
}

function KV({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt className="text-gray-500">{label}</dt>
      <dd className="font-mono text-gray-200 text-right">{value}</dd>
    </>
  );
}

// =============================================================================
// Section 2: Tax-adjusted walk-forward
// =============================================================================

function WalkforwardView({ data }: { data: WalkforwardResponse }) {
  const [view, setView] = useState<'chart' | 'table'>('chart');
  const currency = (data.currency === 'INR' ? 'INR' : 'USD') as Currency;

  return (
    <div className="space-y-3">
      <p className="text-xs text-gray-500">
        Strategy compounds with annual STCG ({(data.summary.strategy_stcg_rate * 100).toFixed(0)}%);
        benchmark ({data.benchmark_label}) compounds tax-deferred with LTCG
        ({(data.summary.benchmark_ltcg_rate * 100).toFixed(1)}%) applied at terminal sale.
        Starting capital {moneyFmt(data.summary.starting_capital, currency, 0)}.
      </p>

      <SummaryTiles summary={data.summary} benchmark={data.benchmark_label} currency={currency} />

      <div className="flex items-center justify-end gap-1">
        <button
          className={[
            'rounded-md px-3 py-1 text-xs',
            view === 'chart'
              ? 'bg-emerald-500/20 text-emerald-200'
              : 'bg-gray-800 text-gray-400 hover:text-gray-200',
          ].join(' ')}
          onClick={() => setView('chart')}
        >
          Chart
        </button>
        <button
          className={[
            'rounded-md px-3 py-1 text-xs',
            view === 'table'
              ? 'bg-emerald-500/20 text-emerald-200'
              : 'bg-gray-800 text-gray-400 hover:text-gray-200',
          ].join(' ')}
          onClick={() => setView('table')}
        >
          Table
        </button>
      </div>

      <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-4">
        {data.years.length === 0 ? (
          <EmptyState
            title="No walk-forward data yet"
            hint="Run a walk-forward backtest to populate."
          />
        ) : view === 'chart' ? (
          <WFEquityChart
            years={data.years}
            currency={currency}
            startingCapital={data.summary.starting_capital}
            benchmarkLabel={data.benchmark_label}
          />
        ) : (
          <WFYearTable years={data.years} currency={currency} benchmarkLabel={data.benchmark_label} />
        )}
      </div>
    </div>
  );
}

function SummaryTiles({
  summary, benchmark, currency,
}: { summary: WalkforwardResponse['summary']; benchmark: string; currency: Currency }) {
  const outperformOk = summary.outperformance_multiple >= 1;
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      <Tile label="Strategy final (after-tax)" value={moneyFmt(summary.strategy_final_aftertax, currency, 0)} />
      <Tile label={`${benchmark.split(' ')[0]} final (after-tax)`} value={moneyFmt(summary.benchmark_final_aftertax, currency, 0)} />
      <Tile label="Strategy pre-tax" value={moneyFmt(summary.strategy_final_pretax, currency, 0)} />
      <div className={[
        'rounded-lg border bg-gray-900/60 px-3 py-2',
        outperformOk ? 'border-emerald-700/50' : 'border-rose-700/50',
      ].join(' ')}>
        <div className="text-[11px] uppercase tracking-wider text-gray-500">Strategy / Benchmark (after-tax)</div>
        <div className={[
          'mt-1 font-mono text-base',
          outperformOk ? 'text-emerald-400' : 'text-rose-400',
        ].join(' ')}>
          {summary.outperformance_multiple.toFixed(2)}×
        </div>
      </div>
    </div>
  );
}

function WFEquityChart({
  years, currency, startingCapital, benchmarkLabel,
}: {
  years: WalkforwardEquityPoint[];
  currency: Currency;
  startingCapital: number;
  benchmarkLabel: string;
}) {
  // X axis = end of year; one prepended starting point for visual anchor.
  const benchKey = benchmarkLabel.split(' ')[0];
  const series = useMemo(() => {
    const first = years[0];
    if (!first) return [];
    return [
      {
        year: first.year - 1,
        Strategy: startingCapital,
        [`${benchKey} (after-tax)`]: startingCapital,
        [`${benchKey} (pre-tax)`]: startingCapital,
      },
      ...years.map((y) => ({
        year: y.year,
        Strategy: y.strategy_equity,
        [`${benchKey} (after-tax)`]: y.benchmark_equity_aftertax,
        [`${benchKey} (pre-tax)`]: y.benchmark_equity_pretax,
      })),
    ];
  }, [years, startingCapital, benchKey]);

  return (
    <div className="h-[400px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={series} margin={{ top: 5, right: 16, bottom: 5, left: 12 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="year" tick={{ fontSize: 11 }} />
          <YAxis
            tick={{ fontSize: 11 }}
            scale="log"
            domain={['auto', 'auto']}
            tickFormatter={(v) => moneyFmt(v, currency, 0)}
          />
          <Tooltip
            formatter={(v: number) => moneyFmt(v, currency, 0)}
            labelFormatter={(y) => `Year: ${y}`}
            contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <ReferenceLine
            y={startingCapital}
            stroke="#6b7280"
            strokeDasharray="3 3"
          />
          <Line
            type="monotone"
            dataKey="Strategy"
            stroke="#34d399"
            strokeWidth={2.5}
            dot
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey={`${benchKey} (after-tax)`}
            stroke="#38bdf8"
            strokeWidth={2}
            dot
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey={`${benchKey} (pre-tax)`}
            stroke="#9ca3af"
            strokeDasharray="4 4"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function WFYearTable({
  years, currency, benchmarkLabel,
}: { years: WalkforwardEquityPoint[]; currency: Currency; benchmarkLabel: string }) {
  const benchKey = benchmarkLabel.split(' ')[0];
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="text-[11px] uppercase tracking-wider text-gray-500">
          <tr>
            <th className="px-2 py-2 text-left">Year</th>
            <th className="px-2 py-2 text-right">Strategy pre-tax</th>
            <th className="px-2 py-2 text-right">Strategy after-tax</th>
            <th className="px-2 py-2 text-right">Strategy equity</th>
            <th className="px-2 py-2 text-right">{benchKey} pre-tax</th>
            <th className="px-2 py-2 text-right">{benchKey} equity (pre-tax)</th>
            <th className="px-2 py-2 text-right">{benchKey} equity (after-tax)</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-800">
          {years.map((y) => (
            <tr key={y.year} className="hover:bg-gray-900/80">
              <td className="px-2 py-2 font-mono text-gray-200">{y.year}</td>
              <td className={`px-2 py-2 text-right font-mono ${y.strategy_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {y.strategy_return_pct >= 0 ? '+' : ''}{pctFmt(y.strategy_return_pct, 2)}
              </td>
              <td className={`px-2 py-2 text-right font-mono ${y.strategy_aftertax_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {y.strategy_aftertax_pct >= 0 ? '+' : ''}{pctFmt(y.strategy_aftertax_pct, 2)}
              </td>
              <td className="px-2 py-2 text-right font-mono text-emerald-300">
                {moneyFmt(y.strategy_equity, currency, 0)}
              </td>
              <td className={`px-2 py-2 text-right font-mono ${y.benchmark_return_pct >= 0 ? 'text-sky-400' : 'text-rose-400'}`}>
                {y.benchmark_return_pct >= 0 ? '+' : ''}{pctFmt(y.benchmark_return_pct, 2)}
              </td>
              <td className="px-2 py-2 text-right font-mono text-gray-400">
                {moneyFmt(y.benchmark_equity_pretax, currency, 0)}
              </td>
              <td className="px-2 py-2 text-right font-mono text-sky-300">
                {moneyFmt(y.benchmark_equity_aftertax, currency, 0)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
