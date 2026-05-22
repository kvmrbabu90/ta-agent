import {
  CartesianGrid,
  Label,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useStrictWf } from '@/hooks/usePerformance';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { ErrorMessage } from '@/components/ErrorMessage';
import { EmptyState } from '@/components/EmptyState';
import type {
  StrictWfEquityCurve,
  StrictWfMonthlyExcessCell,
  StrictWfResponse,
  StrictWfYearPoint,
} from '@/api/types';
import { CHART_BLUE, CHART_GREEN } from '@/utils/colors';

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
        <>
          <EquityCurveChart
            curve={data.equity_curve}
            benchKey={data.benchmark_symbol}
            currency={data.currency}
            startingCapital={data.summary.starting_capital}
          />
          <MonthlyExcessHeatmap
            cells={data.monthly_excess}
            benchKey={data.benchmark_symbol}
          />
          <YearTable years={data.years} benchKey={data.benchmark_symbol} />
        </>
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
  const cumAfterTax =
    s.strategy_cum_return_after_tax_pct != null
      ? `after tax: ${pctFmt(s.strategy_cum_return_after_tax_pct, true)}`
      : undefined;
  const annAfterTax =
    s.strategy_annualized_after_tax_pct != null
      ? `after tax: ${pctFmt(s.strategy_annualized_after_tax_pct, true)}`
      : undefined;
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      <Tile
        label="Strategy cum"
        value={pctFmt(s.strategy_cum_return_pct, true)}
        tone={s.strategy_cum_return_pct >= 0 ? 'pos' : 'neg'}
        subValue={cumAfterTax}
      />
      <Tile
        label={`${data.benchmark_symbol} cum`}
        value={pctFmt(s.benchmark_cum_return_pct, true)}
        tone={s.benchmark_cum_return_pct >= 0 ? 'sky' : 'neg'}
        subValue={
          s.benchmark_cum_return_after_tax_pct != null
            ? `after LTCG: ${pctFmt(s.benchmark_cum_return_after_tax_pct, true)}`
            : undefined
        }
      />
      <Tile
        label="Strategy annualized"
        value={pctFmt(s.strategy_annualized_pct, true)}
        tone={s.strategy_annualized_pct >= s.benchmark_annualized_pct ? 'pos' : 'neg'}
        subValue={annAfterTax}
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
  subValue,
}: {
  label: string;
  value: string;
  tone?: 'pos' | 'neg' | 'sky' | 'neutral';
  subValue?: string;
}) {
  const color =
    tone === 'pos' ? 'text-emerald-400' :
    tone === 'neg' ? 'text-rose-400' :
    tone === 'sky' ? 'text-sky-400' : 'text-gray-100';
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900/60 px-3 py-2">
      <div className="text-[11px] uppercase tracking-wider text-gray-500">{label}</div>
      <div className={`mt-1 font-mono text-base ${color}`}>{value}</div>
      {subValue ? (
        <div
          className="mt-0.5 font-mono text-[11px] text-gray-500"
          title="After capital-gains tax (30% US, 15% India), assuming the prior year's tax bill is paid on Jan 1."
        >
          {subValue}
        </div>
      ) : null}
    </div>
  );
}

// Format a monetary amount using the universe's currency convention.
// USD: $1,000 (US grouping). INR: ₹1,00,000 (Indian lakh/crore grouping).
function fmtMoney(amount: number, currency: string): string {
  if (currency === 'INR') {
    // en-IN locale gives "1,00,000" grouping; symbol is rupee.
    return `₹${Math.round(amount).toLocaleString('en-IN')}`;
  }
  // Default USD/most Western: thousands separator.
  return `$${Math.round(amount).toLocaleString('en-US')}`;
}

function EquityCurveChart({
  curve,
  benchKey,
  currency,
  startingCapital,
}: {
  curve: StrictWfEquityCurve;
  benchKey: string;
  currency: string;
  startingCapital: number;
}) {
  if (!curve || !curve.dates || curve.dates.length === 0) {
    return null;
  }
  // Recharts wants an array of row objects. Zip the columnar arrays.
  const hasBench = curve.benchmark_equity && curve.benchmark_equity.length === curve.dates.length;
  const hasPostTax =
    curve.equity_post_tax &&
    curve.equity_post_tax.length === curve.dates.length &&
    // Hide post-tax series until it diverges from pre-tax (no completed
    // year yet → the two lines overlap and the legend is noise).
    curve.equity_post_tax.some((v, i) => Math.abs(v - curve.equity_pre_tax[i]) > 1e-6);
  const benchEndIdx = curve.dates.length - 1;
  const benchLastEq = hasBench ? curve.benchmark_equity[benchEndIdx] : null;
  const benchPostLtcg = curve.benchmark_post_ltcg_endpoint;
  // Show the dot only if (a) we have a benchmark line, (b) the LTCG
  // value is defined, and (c) it's meaningfully below the pre-tax
  // endpoint (i.e. the benchmark gained — losses pass through and the
  // dot would just sit on top of the line, adding noise).
  const showLtcgDot =
    hasBench &&
    benchPostLtcg != null &&
    benchLastEq != null &&
    benchLastEq - benchPostLtcg > 0.01;
  const data = curve.dates.map((d, i) => ({
    date: d,
    pre: curve.equity_pre_tax[i],
    post: hasPostTax ? curve.equity_post_tax[i] : undefined,
    bench: hasBench ? curve.benchmark_equity[i] : undefined,
  }));
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-3">
      <div className="mb-2 flex items-baseline justify-between">
        <div className="text-[11px] uppercase tracking-wider text-gray-500">Equity curve</div>
        <div className="text-[11px] text-gray-500">
          starts at{' '}
          <span className="font-mono text-gray-300">{fmtMoney(startingCapital, currency)}</span> ·{' '}
          <span className="text-emerald-400">pre-tax</span>
          {hasPostTax ? (
            <>
              {' · '}<span className="text-emerald-400/70">post-tax</span>
            </>
          ) : null}
          {hasBench ? (
            <>
              {' · '}<span className="text-sky-400">{benchKey} B&amp;H</span>
            </>
          ) : null}
          {showLtcgDot ? (
            <>
              {' · '}<span className="text-sky-400/70">{benchKey} post-LTCG</span>
            </>
          ) : null}
        </div>
      </div>
      <div className="h-64 w-full">
        <ResponsiveContainer>
          <LineChart data={data} margin={{ left: 8, right: 16, top: 8, bottom: 8 }}>
            <CartesianGrid stroke="#1f2937" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 10, fill: '#6b7280' }}
              minTickGap={40}
            />
            <YAxis
              tick={{ fontSize: 10, fill: '#6b7280' }}
              tickFormatter={(v: number) => v.toFixed(0)}
              domain={['auto', 'auto']}
              width={56}
            />
            <Tooltip
              contentStyle={{
                fontSize: 11,
                backgroundColor: '#0b1220',
                border: '1px solid #1f2937',
                color: '#e5e7eb',
              }}
              formatter={(v: number | string) =>
                typeof v === 'number' ? fmtMoney(v, currency) : v
              }
            />
            <Line
              type="monotone"
              dataKey="pre"
              name="pre-tax"
              stroke={CHART_GREEN}
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
            {hasPostTax && (
              <Line
                type="monotone"
                dataKey="post"
                name="post-tax"
                stroke={CHART_GREEN}
                strokeOpacity={0.55}
                strokeDasharray="3 3"
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
              />
            )}
            {hasBench && (
              <Line
                type="monotone"
                dataKey="bench"
                name={`${benchKey} B&H`}
                stroke={CHART_BLUE}
                strokeWidth={1.25}
                dot={false}
                isAnimationActive={false}
              />
            )}
            {showLtcgDot && benchPostLtcg != null && (
              <ReferenceDot
                x={curve.dates[benchEndIdx]}
                y={benchPostLtcg}
                r={5}
                fill={CHART_BLUE}
                fillOpacity={0.85}
                stroke="#0b1220"
                strokeWidth={2}
                ifOverflow="extendDomain"
              >
                <Label
                  value={`post-LTCG ${fmtMoney(benchPostLtcg, currency)}`}
                  position="left"
                  offset={10}
                  fill={CHART_BLUE}
                  fontSize={10}
                />
              </ReferenceDot>
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

const MONTH_ABBREV = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

function MonthlyExcessHeatmap({
  cells,
  benchKey,
}: {
  cells: StrictWfMonthlyExcessCell[];
  benchKey: string;
}) {
  if (!cells || cells.length === 0) return null;

  // Build a (year, month) → cell index so we can render the grid even
  // when some cells are missing (e.g. partial first/last year).
  const cellByYM = new Map<string, StrictWfMonthlyExcessCell>();
  let minYear = Infinity;
  let maxYear = -Infinity;
  for (const c of cells) {
    cellByYM.set(`${c.year}-${c.month}`, c);
    if (c.year < minYear) minYear = c.year;
    if (c.year > maxYear) maxYear = c.year;
  }
  const years: number[] = [];
  for (let y = minYear; y <= maxYear; y++) years.push(y);

  // Magnitude scale: |excess|=10% → opacity 1.0; clamp.
  const MAX_MAG = 10;

  function cellBg(excess: number | null | undefined): string {
    if (excess == null || Number.isNaN(excess)) return 'transparent';
    const mag = Math.min(Math.abs(excess) / MAX_MAG, 1);
    if (excess > 0) {
      // emerald-500 #10b981, with alpha scaled by mag
      return `rgba(16, 185, 129, ${(0.15 + 0.85 * mag).toFixed(2)})`;
    }
    if (excess < 0) {
      // rose-500 #f43f5e
      return `rgba(244, 63, 94, ${(0.15 + 0.85 * mag).toFixed(2)})`;
    }
    return 'rgba(107, 114, 128, 0.2)';
  }

  function cellText(excess: number | null | undefined): string {
    if (excess == null || Number.isNaN(excess)) return '—';
    const sign = excess >= 0 ? '+' : '';
    return `${sign}${excess.toFixed(1)}`;
  }

  function cellTooltip(cell: StrictWfMonthlyExcessCell | undefined): string {
    if (!cell) return '';
    const s = cell.strategy_pct != null ? `${cell.strategy_pct.toFixed(2)}%` : '—';
    const b = cell.benchmark_pct != null ? `${cell.benchmark_pct.toFixed(2)}%` : '—';
    const e = cell.excess_pct != null ? `${cell.excess_pct.toFixed(2)}%` : '—';
    return `${cell.year}-${String(cell.month).padStart(2, '0')}\nStrategy: ${s}\n${benchKey}: ${b}\nExcess:   ${e}`;
  }

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-3">
      <div className="mb-2 flex items-baseline justify-between">
        <div className="text-[11px] uppercase tracking-wider text-gray-500">
          Monthly excess vs {benchKey} (Strategy − {benchKey}, %)
        </div>
        <div className="flex items-center gap-2 text-[10px] text-gray-500">
          <span>−10%</span>
          <span
            className="h-2 w-24 rounded"
            style={{
              background:
                'linear-gradient(to right, rgba(244,63,94,1), rgba(244,63,94,0.15), rgba(107,114,128,0.2), rgba(16,185,129,0.15), rgba(16,185,129,1))',
            }}
          />
          <span>+10%</span>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full border-separate" style={{ borderSpacing: 2 }}>
          <thead>
            <tr>
              <th className="w-12 text-left text-[10px] font-medium uppercase tracking-wider text-gray-500"></th>
              {MONTH_ABBREV.map((m) => (
                <th
                  key={m}
                  className="px-1 text-center text-[10px] font-medium uppercase tracking-wider text-gray-500"
                >
                  {m}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {years.map((y) => (
              <tr key={y}>
                <td className="w-12 pr-2 text-right font-mono text-xs text-gray-300">
                  {y}
                </td>
                {MONTH_ABBREV.map((_, idx) => {
                  const month = idx + 1;
                  const cell = cellByYM.get(`${y}-${month}`);
                  return (
                    <td
                      key={month}
                      title={cellTooltip(cell)}
                      className="h-7 min-w-[44px] rounded text-center font-mono text-[10px] text-gray-100"
                      style={{
                        backgroundColor: cellBg(cell?.excess_pct),
                      }}
                    >
                      {cellText(cell?.excess_pct)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
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
