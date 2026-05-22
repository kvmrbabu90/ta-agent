import { useStrictWf } from '@/hooks/usePerformance';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { ErrorMessage } from '@/components/ErrorMessage';
import { EmptyState } from '@/components/EmptyState';
import type { StrictWfResponse, StrictWfYearPoint } from '@/api/types';

function pctFmt(v: number | null | undefined, signed = false, decimals = 2): string {
  if (v == null) return '—';
  const s = v.toFixed(decimals);
  return signed && v >= 0 ? `+${s}%` : `${s}%`;
}

function numFmt(v: number | null | undefined, decimals = 2): string {
  if (v == null) return '—';
  return v.toFixed(decimals);
}

function fmtRelativeUtc(iso: string | null): string {
  if (!iso) return '—';
  const dt = new Date(iso);
  const diffMs = Date.now() - dt.getTime();
  const mins = Math.floor(Math.abs(diffMs) / 60000);
  const ago = diffMs >= 0;
  if (mins < 1) return ago ? 'just now' : 'in <1 min';
  if (mins < 60) return ago ? `${mins} min ago` : `in ${mins} min`;
  const hours = Math.floor(mins / 60);
  if (hours < 48) return ago ? `${hours}h ago` : `in ${hours}h`;
  const days = Math.floor(hours / 24);
  return ago ? `${days}d ago` : `in ${days}d`;
}

export function LiveWFPage() {
  const sp500 = useStrictWf('SP500');
  const nifty = useStrictWf('NIFTY100');

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h1 className="text-2xl font-semibold text-gray-100">Live Walk-Forward</h1>
          <p className="text-sm text-gray-500">
            Strict per-retrain Optuna, look-ahead-free, survivorship-corrected. Auto-refreshes every 60s.
          </p>
        </div>
      </div>

      <UniverseSection title="SP500 (US)" data={sp500.data} loading={sp500.isLoading} error={sp500.error} />
      <UniverseSection title="NIFTY100 (India)" data={nifty.data} loading={nifty.isLoading} error={nifty.error} />
    </div>
  );
}

function UniverseSection({
  title,
  data,
  loading,
  error,
}: {
  title: string;
  data: StrictWfResponse | undefined;
  loading: boolean;
  error: Error | null;
}) {
  if (loading) return <LoadingSpinner label={`Loading ${title}…`} />;
  if (error) return <ErrorMessage error={error} />;
  if (!data) return null;

  return (
    <section className="space-y-3 rounded-lg border border-gray-800 bg-gray-900/40 p-4">
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h2 className="text-base font-semibold text-gray-100">{title}</h2>
          <p className="text-xs text-gray-500">
            Benchmark: {data.benchmark_label} · Currency: {data.currency}
          </p>
        </div>
        <RunningBadge isRunning={data.progress.is_running} />
      </header>

      <ProgressBar data={data} />

      <SummaryTiles data={data} />

      {data.years.length === 0 ? (
        <EmptyState
          title="No retrains complete yet"
          hint="The first retrain (~70-80 min) hasn't finished writing predictions."
        />
      ) : (
        <YearTable years={data.years} benchKey={data.benchmark_symbol} />
      )}
    </section>
  );
}

function RunningBadge({ isRunning }: { isRunning: boolean }) {
  return (
    <span
      className={[
        'inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium',
        isRunning
          ? 'bg-emerald-500/15 text-emerald-300'
          : 'bg-gray-700/30 text-gray-400',
      ].join(' ')}
    >
      <span
        className={[
          'h-1.5 w-1.5 rounded-full',
          isRunning ? 'bg-emerald-400 animate-pulse' : 'bg-gray-500',
        ].join(' ')}
      />
      {isRunning ? 'Running' : 'Idle'}
    </span>
  );
}

function ProgressBar({ data }: { data: StrictWfResponse }) {
  const p = data.progress;
  const pct = p.retrains_total > 0 ? (p.retrains_complete / p.retrains_total) * 100 : 0;
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between text-xs">
        <span className="text-gray-400">
          <span className="font-mono text-gray-200">{p.retrains_complete}</span> /{' '}
          <span className="font-mono text-gray-400">{p.retrains_total}</span> retrains
          {' · '}
          {p.last_retrain_date ? (
            <>
              latest: <span className="font-mono text-gray-200">{p.last_retrain_date}</span>
              {' '}
              <span className="text-gray-500">({fmtRelativeUtc(p.last_retrain_at_utc)})</span>
            </>
          ) : 'no retrains yet'}
        </span>
        <span className="text-gray-500">
          {p.eta_completion_utc ? (
            <>ETA <span className="text-gray-300">{fmtRelativeUtc(p.eta_completion_utc)}</span></>
          ) : '—'}
          {p.avg_retrain_minutes != null && (
            <> · avg <span className="text-gray-300">{p.avg_retrain_minutes.toFixed(0)} min</span>/retrain</>
          )}
        </span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-gray-800">
        <div
          className={[
            'h-full transition-all duration-700',
            p.is_running ? 'bg-emerald-500' : 'bg-gray-600',
          ].join(' ')}
          style={{ width: `${pct.toFixed(1)}%` }}
        />
      </div>
    </div>
  );
}

function SummaryTiles({ data }: { data: StrictWfResponse }) {
  const s = data.summary;
  const ahead = s.strategy_cum_return_pct >= s.benchmark_cum_return_pct;
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      <Tile
        label="Strategy cum"
        value={pctFmt(s.strategy_cum_return_pct, true)}
        tone={s.strategy_cum_return_pct >= 0 ? 'pos' : 'neg'}
      />
      <Tile
        label={`${data.benchmark_symbol} cum`}
        value={pctFmt(s.benchmark_cum_return_pct, true)}
        tone={s.benchmark_cum_return_pct >= 0 ? 'sky' : 'neg'}
      />
      <Tile
        label="Strategy annualized"
        value={pctFmt(s.strategy_annualized_pct, true)}
        tone={s.strategy_annualized_pct >= s.benchmark_annualized_pct ? 'pos' : 'neg'}
      />
      <Tile
        label={ahead ? 'Strategy / Bench' : 'Trailing bench'}
        value={`${s.strategy_multiple.toFixed(2)}×`}
        tone={ahead ? 'pos' : 'neg'}
      />
    </div>
  );
}

function Tile({
  label,
  value,
  tone = 'neutral',
}: {
  label: string;
  value: string;
  tone?: 'pos' | 'neg' | 'sky' | 'neutral';
}) {
  const color =
    tone === 'pos' ? 'text-emerald-400' :
    tone === 'neg' ? 'text-rose-400' :
    tone === 'sky' ? 'text-sky-400' : 'text-gray-100';
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900/60 px-3 py-2">
      <div className="text-[11px] uppercase tracking-wider text-gray-500">{label}</div>
      <div className={`mt-1 font-mono text-base ${color}`}>{value}</div>
    </div>
  );
}

function YearTable({ years, benchKey }: { years: StrictWfYearPoint[]; benchKey: string }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-gray-800 bg-gray-900/60">
      <table className="w-full text-sm">
        <thead className="text-[11px] uppercase tracking-wider text-gray-500">
          <tr>
            <th className="px-3 py-2 text-left">Year</th>
            <th className="px-3 py-2 text-right">Strategy</th>
            <th
              className="px-3 py-2 text-right"
              title="Strategy return after capital-gains tax (30% US, 15% India). Populated only at end of calendar year."
            >
              After Tax
            </th>
            <th className="px-3 py-2 text-right">{benchKey}</th>
            <th className="px-3 py-2 text-right">Excess</th>
            <th className="px-3 py-2 text-right">Sharpe</th>
            <th className="px-3 py-2 text-right">Max DD</th>
            <th className="px-3 py-2 text-right">Days</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-800">
          {years.map((y) => (
            <tr key={y.year} className="hover:bg-gray-900/80">
              <td className="px-3 py-2 font-mono text-gray-200">{y.year}</td>
              <td className={`px-3 py-2 text-right font-mono ${y.strategy_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {pctFmt(y.strategy_return_pct, true)}
              </td>
              <td
                className={`px-3 py-2 text-right font-mono ${
                  y.strategy_return_after_tax_pct == null
                    ? 'text-gray-500'
                    : y.strategy_return_after_tax_pct >= 0
                      ? 'text-emerald-400/80'
                      : 'text-rose-400/80'
                }`}
              >
                {y.strategy_return_after_tax_pct != null
                  ? pctFmt(y.strategy_return_after_tax_pct, true)
                  : '—'}
              </td>
              <td className={`px-3 py-2 text-right font-mono ${(y.benchmark_return_pct ?? 0) >= 0 ? 'text-sky-400' : 'text-rose-400'}`}>
                {y.benchmark_return_pct != null ? pctFmt(y.benchmark_return_pct, true) : '—'}
              </td>
              <td className={`px-3 py-2 text-right font-mono ${(y.excess_pct ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {y.excess_pct != null ? pctFmt(y.excess_pct, true) : '—'}
              </td>
              <td className={`px-3 py-2 text-right font-mono ${(y.sharpe ?? 0) >= 0 ? 'text-gray-200' : 'text-rose-400'}`}>
                {numFmt(y.sharpe, 2)}
              </td>
              <td className="px-3 py-2 text-right font-mono text-gray-400">
                {y.max_dd_pct != null ? pctFmt(y.max_dd_pct, false, 1) : '—'}
              </td>
              <td className="px-3 py-2 text-right font-mono text-gray-500">{y.n_days}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
