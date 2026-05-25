import {
  Brush,
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
import { useEffect, useState, type ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import { useStrictWf, useStrictWfAnalysis, useStrictWfMonth } from '@/hooks/usePerformance';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { ErrorMessage } from '@/components/ErrorMessage';
import { EmptyState } from '@/components/EmptyState';
import type {
  StrictWfEquityCurve,
  StrictWfMonthDetail,
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
  const [selectedCell, setSelectedCell] = useState<{ year: number; month: number } | null>(null);

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
            onCellClick={(c) => setSelectedCell({ year: c.year, month: c.month })}
          />
          <YearTable years={data.years} benchKey={data.benchmark_symbol} />
          <AnalysisPanel universe={data.universe} />
          {selectedCell ? (
            <MonthDetailModal
              universe={data.universe}
              benchKey={data.benchmark_symbol}
              year={selectedCell.year}
              month={selectedCell.month}
              onClose={() => setSelectedCell(null)}
            />
          ) : null}
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
      <NextRetrainCountdown
        lastRetrainAtUtc={p.last_retrain_at_utc}
        avgRetrainMinutes={p.avg_retrain_minutes}
        isRunning={p.is_running}
      />
    </div>
  );
}

// Live countdown to next expected retrain completion. Re-derived from
// last_retrain_at_utc + avg_retrain_minutes; ticks every second.
function NextRetrainCountdown({
  lastRetrainAtUtc,
  avgRetrainMinutes,
  isRunning,
}: {
  lastRetrainAtUtc: string | null;
  avgRetrainMinutes: number | null;
  isRunning: boolean;
}) {
  const [now, setNow] = useState<number>(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);
  if (!lastRetrainAtUtc || avgRetrainMinutes == null) return null;
  // Backend emits naive ISO (utcnow().isoformat()); add 'Z' if it has no zone marker.
  const iso = /[zZ]|[+-]\d{2}:\d{2}$/.test(lastRetrainAtUtc)
    ? lastRetrainAtUtc
    : `${lastRetrainAtUtc}Z`;
  const lastMs = new Date(iso).getTime();
  if (Number.isNaN(lastMs)) return null;
  const nextMs = lastMs + avgRetrainMinutes * 60_000;
  const remainingMs = nextMs - now;
  let label: ReactNode;
  let toneClass = 'text-gray-400';
  if (remainingMs > 0) {
    const totalSec = Math.floor(remainingMs / 1000);
    const m = Math.floor(totalSec / 60);
    const s = totalSec % 60;
    label = (
      <>
        Next retrain in{' '}
        <span className="font-mono text-gray-200">
          {m}m {s.toString().padStart(2, '0')}s
        </span>
      </>
    );
    toneClass = 'text-gray-400';
  } else {
    // We're past the avg-based ETA — show how overdue. Common when a retrain
    // happens to take longer than the rolling average (slow Optuna trial, etc.).
    const overdueSec = Math.floor(-remainingMs / 1000);
    const m = Math.floor(overdueSec / 60);
    const s = overdueSec % 60;
    label = (
      <>
        Retrain due — running{' '}
        <span className="font-mono text-amber-300">
          {m}m {s.toString().padStart(2, '0')}s
        </span>{' '}
        over avg
      </>
    );
    toneClass = 'text-amber-300/80';
  }
  if (!isRunning) {
    return (
      <div className="text-[11px] text-gray-500">
        WF idle — no retrain in flight
      </div>
    );
  }
  return <div className={`text-[11px] ${toneClass}`}>{label}</div>;
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
      {(() => {
        // Strategy / Bench tile as an after-tax ratio. Numerator =
        // strategy multiple after 30% STCG; denominator = SPY multiple
        // after 15% LTCG. Answers "for every dollar a B&H investor
        // walks away with after-tax, how many does the strategy walk
        // away with after-tax".
        //
        // Falls back to pre-tax ratio if after-tax values aren't
        // populated yet (first year not complete) — gracefully avoids
        // showing N/A for fresh runs.
        const stratMult =
          s.strategy_multiple_after_tax != null
            ? s.strategy_multiple_after_tax
            : 1 + s.strategy_cum_return_pct / 100;
        const benchMult =
          s.benchmark_cum_return_after_tax_pct != null
            ? 1 + s.benchmark_cum_return_after_tax_pct / 100
            : 1 + s.benchmark_cum_return_pct / 100;
        const ratio = benchMult > 0 ? stratMult / benchMult : null;
        const aheadAT = ratio != null && ratio >= 1;
        const labelSuffix =
          s.strategy_multiple_after_tax != null && s.benchmark_cum_return_after_tax_pct != null
            ? ' (a/t)'
            : '';
        return (
          <Tile
            label={(aheadAT ? 'Strategy / Bench' : 'Trailing bench') + labelSuffix}
            value={ratio != null ? `${ratio.toFixed(2)}×` : '—'}
            tone={aheadAT ? 'pos' : 'neg'}
            subValue={
              labelSuffix
                ? `pre-tax: ${(s.strategy_multiple / (1 + s.benchmark_cum_return_pct / 100)).toFixed(2)}×`
                : undefined
            }
          />
        );
      })()}
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
          <span className="text-emerald-400/70">pre-tax (dotted)</span>
          {hasPostTax ? (
            <>
              {' · '}<span className="text-emerald-400">post-tax (solid)</span>
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
      <div className="h-72 w-full">
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
              strokeOpacity={0.55}
              strokeDasharray="3 3"
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
            {/* Brush: drag the handles below the chart to zoom into a date
                range. Y-axis auto-scales to the selected window so vertical
                detail isn't compressed by the full-history scale. */}
            <Brush
              dataKey="date"
              height={24}
              stroke="#374151"
              fill="#0b1220"
              travellerWidth={8}
              tickFormatter={(v: string) => v.slice(0, 7)}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

const MONTH_ABBREV = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

type HeatmapMode = 'excess' | 'strategy';

function MonthlyExcessHeatmap({
  cells,
  benchKey,
  onCellClick,
}: {
  cells: StrictWfMonthlyExcessCell[];
  benchKey: string;
  onCellClick: (cell: StrictWfMonthlyExcessCell) => void;
}) {
  const [mode, setMode] = useState<HeatmapMode>('excess');

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

  // Color scale magnitude depends on the metric. Excess cells rarely
  // exceed ±10% (it's a difference). Raw strategy returns can hit
  // ±15% in big months, so loosen the clamp accordingly.
  const MAX_MAG = mode === 'strategy' ? 15 : 10;

  function valueFor(cell: StrictWfMonthlyExcessCell | undefined): number | null {
    if (!cell) return null;
    const v = mode === 'strategy' ? cell.strategy_pct : cell.excess_pct;
    return v ?? null;
  }

  function cellBg(v: number | null | undefined): string {
    if (v == null || Number.isNaN(v)) return 'transparent';
    const mag = Math.min(Math.abs(v) / MAX_MAG, 1);
    if (v > 0) {
      // emerald-500 #10b981, with alpha scaled by mag
      return `rgba(16, 185, 129, ${(0.15 + 0.85 * mag).toFixed(2)})`;
    }
    if (v < 0) {
      // rose-500 #f43f5e
      return `rgba(244, 63, 94, ${(0.15 + 0.85 * mag).toFixed(2)})`;
    }
    return 'rgba(107, 114, 128, 0.2)';
  }

  function cellText(v: number | null | undefined): string {
    if (v == null || Number.isNaN(v)) return '—';
    const sign = v >= 0 ? '+' : '';
    return `${sign}${v.toFixed(1)}`;
  }

  function cellTooltip(cell: StrictWfMonthlyExcessCell | undefined): string {
    if (!cell) return '';
    const s = cell.strategy_pct != null ? `${cell.strategy_pct.toFixed(2)}%` : '—';
    const b = cell.benchmark_pct != null ? `${cell.benchmark_pct.toFixed(2)}%` : '—';
    const e = cell.excess_pct != null ? `${cell.excess_pct.toFixed(2)}%` : '—';
    return `${cell.year}-${String(cell.month).padStart(2, '0')}\nStrategy: ${s}\n${benchKey}: ${b}\nExcess:   ${e}`;
  }

  const headerLabel =
    mode === 'strategy'
      ? 'Monthly strategy return (%)'
      : `Monthly excess vs ${benchKey} (Strategy − ${benchKey}, %)`;
  const scaleLabel = mode === 'strategy' ? '15%' : '10%';

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-3">
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <div className="text-[11px] uppercase tracking-wider text-gray-500">
          {headerLabel}
        </div>
        <div className="flex items-center gap-3">
          {/* Mode toggle */}
          <div
            className="inline-flex overflow-hidden rounded-md border border-gray-700 text-[10px]"
            role="tablist"
            aria-label="Heatmap metric"
          >
            <button
              type="button"
              role="tab"
              aria-selected={mode === 'excess'}
              onClick={() => setMode('excess')}
              className={[
                'px-2 py-0.5 transition-colors',
                mode === 'excess'
                  ? 'bg-emerald-500/15 text-emerald-300'
                  : 'text-gray-400 hover:bg-gray-800/60 hover:text-gray-100',
              ].join(' ')}
            >
              Excess vs {benchKey}
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mode === 'strategy'}
              onClick={() => setMode('strategy')}
              className={[
                'px-2 py-0.5 transition-colors',
                mode === 'strategy'
                  ? 'bg-emerald-500/15 text-emerald-300'
                  : 'text-gray-400 hover:bg-gray-800/60 hover:text-gray-100',
              ].join(' ')}
            >
              Strategy return
            </button>
          </div>
          {/* Color legend */}
          <div className="flex items-center gap-2 text-[10px] text-gray-500">
            <span>−{scaleLabel}</span>
            <span
              className="h-2 w-24 rounded"
              style={{
                background:
                  'linear-gradient(to right, rgba(244,63,94,1), rgba(244,63,94,0.15), rgba(107,114,128,0.2), rgba(16,185,129,0.15), rgba(16,185,129,1))',
              }}
            />
            <span>+{scaleLabel}</span>
          </div>
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
                  const v = valueFor(cell);
                  const clickable = !!cell;
                  return (
                    <td
                      key={month}
                      title={cellTooltip(cell) + (clickable ? '\n(click for details)' : '')}
                      onClick={clickable ? () => onCellClick(cell!) : undefined}
                      className={[
                        'h-7 min-w-[44px] rounded text-center font-mono text-[10px] text-gray-100',
                        clickable
                          ? 'cursor-pointer hover:outline hover:outline-1 hover:outline-gray-300/50'
                          : '',
                      ].join(' ')}
                      style={{
                        backgroundColor: cellBg(v),
                      }}
                    >
                      {cellText(v)}
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
            <th
              className="px-3 py-2 text-right"
              title={`After-tax strategy return − ${benchKey} pre-tax return. ${benchKey} LTCG is intentionally ignored here so the comparison is "what the strategy actually keeps" vs "what the benchmark prints". Populated only at end of calendar year (when after-tax is available).`}
            >
              Excess (a/t)
            </th>
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
              {(() => {
                // After-tax excess: strategy_after_tax − benchmark (pre-tax).
                // Ignored SPY LTCG per spec — answers "what the strategy
                // keeps vs what the benchmark prints". Falls back to em-dash
                // when the year hasn't completed yet (after_tax is null).
                const atExcess =
                  y.strategy_return_after_tax_pct != null &&
                  y.benchmark_return_pct != null
                    ? y.strategy_return_after_tax_pct - y.benchmark_return_pct
                    : null;
                return (
                  <td
                    className={`px-3 py-2 text-right font-mono ${
                      atExcess == null
                        ? 'text-gray-500'
                        : atExcess >= 0
                          ? 'text-emerald-400'
                          : 'text-rose-400'
                    }`}
                  >
                    {atExcess != null ? pctFmt(atExcess, true) : '—'}
                  </td>
                );
              })()}
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

function AnalysisPanel({ universe }: { universe: string }) {
  const { data, isLoading } = useStrictWfAnalysis(universe);
  // If no analysis has been written yet, show a hint pointing at /wf-analysis.
  const hasContent = !!data?.markdown;
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-4">
      <div className="mb-3 flex items-baseline justify-between gap-2">
        <h3 className="text-sm font-semibold text-gray-200">Analysis</h3>
        <div className="text-[11px] text-gray-500">
          {hasContent && data?.covers_through ? (
            <>
              covers through <span className="font-mono text-gray-300">{data.covers_through}</span>
              {data.written_at ? (
                <> · written <span className="font-mono text-gray-300">{data.written_at.slice(0, 16).replace('T', ' ')}</span></>
              ) : null}
            </>
          ) : (
            <span className="italic">no analysis yet — run <code className="rounded bg-gray-800 px-1 py-0.5 font-mono text-gray-300">/wf-analysis</code> in chat</span>
          )}
        </div>
      </div>
      {isLoading ? (
        <div className="text-xs text-gray-500">Loading…</div>
      ) : hasContent ? (
        <div className="prose prose-invert prose-sm max-w-none text-gray-300
          prose-headings:text-gray-100 prose-headings:mt-3 prose-headings:mb-1.5
          prose-h2:text-sm prose-h2:font-semibold
          prose-h3:text-xs prose-h3:font-semibold prose-h3:uppercase prose-h3:tracking-wider prose-h3:text-gray-400
          prose-p:text-sm prose-p:my-1.5
          prose-strong:text-gray-100
          prose-ul:my-1.5 prose-li:my-0.5 prose-li:text-sm
          prose-table:text-xs prose-th:border-gray-700 prose-td:border-gray-800
          prose-code:rounded prose-code:bg-gray-800 prose-code:px-1 prose-code:py-0.5 prose-code:text-gray-300 prose-code:before:content-none prose-code:after:content-none">
          <ReactMarkdown>{data.markdown!}</ReactMarkdown>
        </div>
      ) : (
        <div className="text-xs text-gray-500">
          The dashboard analysis card hasn't been generated yet. From this chat,
          type <code className="rounded bg-gray-800 px-1 py-0.5 font-mono text-gray-300">/wf-analysis</code>
          {' '}to publish a fresh write-up here.
        </div>
      )}
    </div>
  );
}

const MONTH_NAME = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
];

function MonthDetailModal({
  universe,
  benchKey,
  year,
  month,
  onClose,
}: {
  universe: string;
  benchKey: string;
  year: number;
  month: number;
  onClose: () => void;
}) {
  const { data, isLoading, error } = useStrictWfMonth(universe, year, month);
  // ESC key closes
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
    >
      <div
        className="max-h-[90vh] w-full max-w-4xl overflow-y-auto rounded-lg border border-gray-700 bg-gray-950 p-5 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="mb-4 flex items-start justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-100">
              {MONTH_NAME[month - 1]} {year}
            </h3>
            <p className="text-xs text-gray-500">
              {universe} · vs {benchKey} · click outside or press Esc to close
            </p>
          </div>
          <button
            type="button"
            className="rounded p-1 text-gray-400 hover:bg-gray-800 hover:text-gray-200"
            aria-label="Close"
            onClick={onClose}
          >
            ✕
          </button>
        </header>
        {isLoading && <div className="text-sm text-gray-400">Loading…</div>}
        {error && <ErrorMessage error={error as Error} />}
        {data && <MonthDetailBody data={data} benchKey={benchKey} />}
      </div>
    </div>
  );
}

function MonthDetailBody({ data, benchKey }: { data: StrictWfMonthDetail; benchKey: string }) {
  const tone = (v: number | null | undefined): string =>
    v == null
      ? 'text-gray-400'
      : v > 0
        ? 'text-emerald-400'
        : v < 0
          ? 'text-rose-400'
          : 'text-gray-100';

  // Build cumulative-equity series (indexed to 100 at month start) for both
  // strategy and benchmark so the chart shows the path within the month.
  const chartData: { date: string; strat: number; bench: number }[] = [];
  let stratEq = 100;
  let benchEq = 100;
  for (const d of data.daily) {
    if (d.strategy_pct != null) stratEq *= 1 + d.strategy_pct / 100;
    if (d.benchmark_pct != null) benchEq *= 1 + d.benchmark_pct / 100;
    chartData.push({ date: d.date, strat: stratEq, bench: benchEq });
  }

  return (
    <div className="space-y-4">
      {/* Headline tiles */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <DetailTile
          label="Strategy"
          value={data.strategy_pct != null ? pctFmt(data.strategy_pct, true) : '—'}
          toneClass={tone(data.strategy_pct)}
        />
        <DetailTile
          label={benchKey}
          value={data.benchmark_pct != null ? pctFmt(data.benchmark_pct, true) : '—'}
          toneClass={tone(data.benchmark_pct)}
        />
        <DetailTile
          label="Excess"
          value={data.excess_pct != null ? pctFmt(data.excess_pct, true) : '—'}
          toneClass={tone(data.excess_pct)}
        />
        <DetailTile
          label="Sharpe / MaxDD"
          value={`${data.sharpe != null ? data.sharpe.toFixed(2) : '—'} / ${
            data.max_dd_pct != null ? data.max_dd_pct.toFixed(1) + '%' : '—'
          }`}
          toneClass="text-gray-200"
        />
      </div>

      {/* Daily equity chart (rebased to 100) */}
      <div className="rounded border border-gray-800 bg-gray-900/40 p-2">
        <div className="mb-1 flex items-baseline justify-between text-[11px] text-gray-500">
          <span className="uppercase tracking-wider">
            Daily equity path (rebased to 100 at month start)
          </span>
          <span>
            <span className="text-emerald-400">strategy</span>
            {' · '}
            <span className="text-sky-400">{benchKey}</span>
          </span>
        </div>
        <div className="h-48 w-full">
          <ResponsiveContainer>
            <LineChart data={chartData} margin={{ left: 4, right: 8, top: 4, bottom: 4 }}>
              <CartesianGrid stroke="#1f2937" />
              <XAxis dataKey="date" tick={{ fontSize: 9, fill: '#6b7280' }} minTickGap={32} />
              <YAxis
                tick={{ fontSize: 10, fill: '#6b7280' }}
                tickFormatter={(v: number) => v.toFixed(0)}
                domain={['auto', 'auto']}
                width={40}
              />
              <Tooltip
                contentStyle={{
                  fontSize: 11,
                  backgroundColor: '#0b1220',
                  border: '1px solid #1f2937',
                  color: '#e5e7eb',
                }}
                formatter={(v: number | string) =>
                  typeof v === 'number' ? v.toFixed(2) : v
                }
              />
              <Line
                type="monotone"
                dataKey="strat"
                name="strategy"
                stroke="#34d399"
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="bench"
                name={benchKey}
                stroke="#38bdf8"
                strokeWidth={1.25}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Best / worst days */}
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <DayList title="Best days (by excess)" days={data.best_days} benchKey={benchKey} />
        <DayList title="Worst days (by excess)" days={data.worst_days} benchKey={benchKey} />
      </div>

      {/* Top holdings */}
      <div className="rounded border border-gray-800 bg-gray-900/40">
        <div className="px-3 py-2 text-[11px] uppercase tracking-wider text-gray-500">
          Top holdings during the month (by days held)
        </div>
        {data.top_holdings.length === 0 ? (
          <div className="px-3 py-2 text-sm text-gray-500">No position data.</div>
        ) : (
          <table className="w-full text-sm">
            <thead className="text-[10px] uppercase tracking-wider text-gray-500">
              <tr>
                <th className="px-3 py-1.5 text-left">Symbol</th>
                <th className="px-3 py-1.5 text-right">Days held</th>
                <th className="px-3 py-1.5 text-right">Avg weight</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {data.top_holdings.map((h) => (
                <tr key={h.symbol} className="hover:bg-gray-900/60">
                  <td className="px-3 py-1.5 font-mono text-gray-100">{h.symbol}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-gray-300">
                    {h.days_held} / {data.n_days}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-gray-300">
                    {h.avg_weight_pct.toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Footnote */}
      <div className="text-[10px] text-gray-500">
        n_days={data.n_days} · vol (annualized) ={' '}
        {data.vol_pct != null ? `${data.vol_pct.toFixed(1)}%` : '—'} · holdings ranked by days
        held; ties broken by avg position weight.
      </div>
    </div>
  );
}

function DetailTile({
  label,
  value,
  toneClass,
}: {
  label: string;
  value: string;
  toneClass: string;
}) {
  return (
    <div className="rounded border border-gray-800 bg-gray-900/60 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-gray-500">{label}</div>
      <div className={`mt-0.5 font-mono text-base ${toneClass}`}>{value}</div>
    </div>
  );
}

function DayList({
  title,
  days,
  benchKey,
}: {
  title: string;
  days: StrictWfMonthDetail['best_days'];
  benchKey: string;
}) {
  return (
    <div className="rounded border border-gray-800 bg-gray-900/40">
      <div className="px-3 py-2 text-[11px] uppercase tracking-wider text-gray-500">
        {title}
      </div>
      {days.length === 0 ? (
        <div className="px-3 py-2 text-sm text-gray-500">No data.</div>
      ) : (
        <table className="w-full text-sm">
          <thead className="text-[10px] uppercase tracking-wider text-gray-500">
            <tr>
              <th className="px-3 py-1 text-left">Date</th>
              <th className="px-3 py-1 text-right">Strategy</th>
              <th className="px-3 py-1 text-right">{benchKey}</th>
              <th className="px-3 py-1 text-right">Excess</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {days.map((d) => (
              <tr key={d.date}>
                <td className="px-3 py-1 font-mono text-gray-200">{d.date}</td>
                <td
                  className={`px-3 py-1 text-right font-mono ${
                    (d.strategy_pct ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'
                  }`}
                >
                  {d.strategy_pct != null ? pctFmt(d.strategy_pct, true) : '—'}
                </td>
                <td
                  className={`px-3 py-1 text-right font-mono ${
                    (d.benchmark_pct ?? 0) >= 0 ? 'text-sky-400' : 'text-rose-400'
                  }`}
                >
                  {d.benchmark_pct != null ? pctFmt(d.benchmark_pct, true) : '—'}
                </td>
                <td
                  className={`px-3 py-1 text-right font-mono ${
                    (d.excess_pct ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'
                  }`}
                >
                  {d.excess_pct != null ? pctFmt(d.excess_pct, true) : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
