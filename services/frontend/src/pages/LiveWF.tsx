import {
  Bar,
  BarChart,
  Brush,
  CartesianGrid,
  Cell,
  Label,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useEffect, useState, type ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useStrictWf, useStrictWfAnalysis, useStrictWfMonth } from '@/hooks/usePerformance';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { ErrorMessage } from '@/components/ErrorMessage';
import { EmptyState } from '@/components/EmptyState';
import type {
  StrictWfEquityCurve,
  StrictWfGateDecision,
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
  const [variant, setVariant] = useState<string>('baseline');
  const sp500 = useStrictWf('SP500', variant);
  const variants = sp500.data?.available_variants ?? [];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h1 className="text-2xl font-semibold text-gray-100">Live Walk-Forward</h1>
          <p className="text-sm text-gray-500">
            Strict per-retrain Optuna, look-ahead-free, survivorship-corrected. Auto-refreshes every 60s.
          </p>
        </div>
        {variants.length > 1 && (
          <label className="flex items-center gap-2 text-xs text-gray-400">
            WF run
            <select
              value={variant}
              onChange={(e) => setVariant(e.target.value)}
              className="rounded-md border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-gray-200 focus:border-emerald-500 focus:outline-none"
            >
              {variants.map((v) => (
                <option key={v.key} value={v.key}>{v.label}</option>
              ))}
            </select>
          </label>
        )}
      </div>

      {variant !== 'baseline' && (
        <div className="rounded-md border border-amber-900/50 bg-amber-950/20 px-3 py-2 text-xs text-amber-300/90">
          Viewing an <span className="font-semibold">experiment</span> run, not the locked V1 baseline.
          {sp500.data?.progress && !sp500.data.progress.is_running ? '' : ' This run may still be in progress — numbers fill in as retrains complete.'}
        </div>
      )}

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
          <MonthlyHistograms
            cells={data.monthly_excess}
            benchKey={data.benchmark_symbol}
          />
          <MonthlyExcessHeatmap
            cells={data.monthly_excess}
            benchKey={data.benchmark_symbol}
            gateDecisions={data.gate_decisions}
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
  // Persisted user preferences for this chart, survive page reloads via
  // localStorage. Stored as JSON under EQUITY_CHART_PREFS_KEY. The Brush
  // window is saved as DATES (not indices) so it stays anchored to the
  // same points after new retrains extend the time series.
  const EQUITY_CHART_PREFS_KEY = 'wf-equity-chart-prefs-v1';
  type EquityChartPrefs = {
    showPreTax?: boolean;
  };
  const readPrefs = (): EquityChartPrefs => {
    try {
      const raw = localStorage.getItem(EQUITY_CHART_PREFS_KEY);
      return raw ? (JSON.parse(raw) as EquityChartPrefs) : {};
    } catch {
      return {};
    }
  };

  // Show/hide the pre-tax (dotted) series. Pre-tax compounds at the
  // gross strategy rate and runs ~3× higher than the post-tax line in
  // late years (after the STCG drag), which compresses the more
  // actionable post-tax + SPY comparison. Hiding it lets the y-axis
  // auto-scale down to the realistic-account range. Default: visible.
  // Hydrated from localStorage on mount.
  const [showPreTax, setShowPreTax] = useState<boolean>(
    () => readPrefs().showPreTax ?? true,
  );
  // Brush window indices. `undefined` = no brush selection (Recharts
  // shows the full range). Kept in React state only: the zoom survives the
  // 60s auto-refresh (the component stays mounted) but resets to full range
  // on a page reload or variant switch. We deliberately do NOT persist the
  // zoom — a saved window anchored to old dates (or a different variant's
  // shorter series) used to silently hide the whole curve, leaving the
  // chart stuck zoomed into the first few weeks of 2014.
  const [brushStart, setBrushStart] = useState<number | undefined>(undefined);
  const [brushEnd, setBrushEnd] = useState<number | undefined>(undefined);

  // Reset the brush to full range whenever the underlying series changes
  // (variant switch / first load) so a new dataset always opens un-zoomed.
  useEffect(() => {
    setBrushStart(undefined);
    setBrushEnd(undefined);
  }, [curve?.dates?.length]);

  // Persist only the pre-tax toggle (not the transient zoom).
  useEffect(() => {
    try {
      localStorage.setItem(
        EQUITY_CHART_PREFS_KEY, JSON.stringify({ showPreTax } as EquityChartPrefs),
      );
    } catch {
      /* localStorage can throw in private mode / when quota is hit */
    }
  }, [showPreTax]);

  // Reset everything back to defaults (full range, pre-tax visible) and
  // wipe the persisted prefs so a subsequent refresh shows the same
  // clean state.
  const resetChart = () => {
    setShowPreTax(true);
    setBrushStart(undefined);
    setBrushEnd(undefined);
    try {
      localStorage.removeItem(EQUITY_CHART_PREFS_KEY);
    } catch {
      /* ignore */
    }
  };

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
          {/* Click to hide/show the pre-tax dotted line. Greyed +
              strikethrough = hidden; bright emerald = visible. Toggling
              recomputes the y-axis auto-scale via the conditional Line
              below. */}
          <button
            type="button"
            onClick={() => setShowPreTax((v) => !v)}
            className={`cursor-pointer transition-colors ${
              showPreTax
                ? 'text-emerald-400/70 hover:text-emerald-300'
                : 'text-gray-500 line-through hover:text-gray-400'
            }`}
            title={showPreTax ? 'Hide pre-tax line (y-axis will rescale)' : 'Show pre-tax line'}
            aria-pressed={showPreTax}
          >
            pre-tax (dotted)
          </button>
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
          {/* Reset: clears the persisted toggle + brush window so the
              chart returns to default (pre-tax visible, full date range).
              Shown subtle by default, brightens on hover. */}
          {' · '}
          <button
            type="button"
            onClick={resetChart}
            className="cursor-pointer text-gray-500 transition-colors hover:text-gray-300"
            title="Reset chart to defaults (show pre-tax line, full date range)"
          >
            reset
          </button>
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
            {showPreTax && (
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
            )}
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
                detail isn't compressed by the full-history scale.
                Controlled by brushStart/brushEnd so the window survives
                page reloads (persisted via localStorage as dates). */}
            <Brush
              dataKey="date"
              height={24}
              stroke="#374151"
              fill="#0b1220"
              travellerWidth={8}
              tickFormatter={(v: string) => v.slice(0, 7)}
              startIndex={brushStart}
              endIndex={brushEnd}
              onChange={(range: { startIndex?: number; endIndex?: number }) => {
                if (range.startIndex == null || range.endIndex == null) return;
                setBrushStart(range.startIndex);
                setBrushEnd(range.endIndex);
              }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

const MONTH_ABBREV = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

// ---------------------------------------------------------------------------
// MonthlyHistograms — two side-by-side histograms (Excess vs SPY, Strategy
// return) computed from the same monthly_excess cells the heatmap consumes.
//
// Each histogram shows:
//   - Bar = count of months in each ~1% bin
//   - Mean (vertical solid line + label)
//   - Median (vertical dashed line + label)
//   - ±1σ band (dotted vertical lines flanking the mean)
//   - Pareto stat: what % of total |value| comes from the most extreme 20%
//     of months (fat-tailedness gauge — 80% = textbook Pareto; higher = even
//     fatter tails, lower = more uniform)
// Tooltip on each bar shows bin range + count.
// ---------------------------------------------------------------------------

type HistogramStats = {
  n: number;
  mean: number;
  median: number;
  std: number;
  paretoShare: number; // top-20%-by-|x| share of total |x|
  min: number;
  max: number;
};

function computeStats(values: number[]): HistogramStats | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  const mean = sorted.reduce((a, b) => a + b, 0) / n;
  const median =
    n % 2 === 1 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
  const variance =
    n > 1 ? sorted.reduce((a, x) => a + (x - mean) ** 2, 0) / (n - 1) : 0;
  const std = Math.sqrt(variance);
  // Pareto: share of total |x| contributed by the top 20% of months by |x|.
  // For a perfectly-uniform distribution this is exactly 20%; for a textbook
  // Pareto 80/20 it's 80%. Anything above 50% signals fat-tailed outcomes
  // (the headline excess/return is concentrated in a small number of months).
  const absSorted = values.map(Math.abs).sort((a, b) => b - a);
  const topN = Math.max(1, Math.ceil(absSorted.length * 0.2));
  const totalAbs = absSorted.reduce((a, b) => a + b, 0);
  const topAbs = absSorted.slice(0, topN).reduce((a, b) => a + b, 0);
  const paretoShare = totalAbs > 0 ? topAbs / totalAbs : 0;
  return {
    n,
    mean,
    median,
    std,
    paretoShare,
    min: sorted[0],
    max: sorted[sorted.length - 1],
  };
}

function buildHistogramBins(
  values: number[],
  binWidth: number,
): { binCenter: number; count: number; binLo: number; binHi: number }[] {
  if (values.length === 0) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  // Snap bin boundaries to multiples of binWidth around zero so the
  // histogram is symmetric and the zero line falls exactly on a bin edge
  // (not inside a bin). This is what makes the Mean / Median reference
  // lines visually align with the bar grid.
  const lo = Math.floor(min / binWidth) * binWidth;
  const hi = Math.ceil(max / binWidth) * binWidth;
  const nBins = Math.max(1, Math.round((hi - lo) / binWidth));
  const bins: { binCenter: number; count: number; binLo: number; binHi: number }[] = [];
  for (let i = 0; i < nBins; i++) {
    const binLo = lo + i * binWidth;
    const binHi = binLo + binWidth;
    bins.push({ binCenter: binLo + binWidth / 2, count: 0, binLo, binHi });
  }
  for (const v of values) {
    // Right-open intervals [binLo, binHi); clamp the max value into the
    // last bin so it doesn't fall off the end.
    let idx = Math.floor((v - lo) / binWidth);
    if (idx >= nBins) idx = nBins - 1;
    if (idx < 0) idx = 0;
    bins[idx].count += 1;
  }
  return bins;
}

function MonthlyHistograms({
  cells,
  benchKey,
}: {
  cells: StrictWfMonthlyExcessCell[];
  benchKey: string;
}) {
  if (!cells || cells.length === 0) return null;

  const excessValues = cells
    .map((c) => c.excess_pct)
    .filter((v): v is number => v != null && !Number.isNaN(v));
  const strategyValues = cells
    .map((c) => c.strategy_pct)
    .filter((v): v is number => v != null && !Number.isNaN(v));

  const excessStats = computeStats(excessValues);
  const strategyStats = computeStats(strategyValues);
  // 1% bins for both — fine-grained enough to see the shape, coarse
  // enough not to look spiky on ~140 months of data.
  const excessBins = buildHistogramBins(excessValues, 1);
  const strategyBins = buildHistogramBins(strategyValues, 1);

  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
      <HistogramCard
        title={`Monthly excess vs ${benchKey} (%)`}
        bins={excessBins}
        stats={excessStats}
        unit="%"
        positiveTint
      />
      <HistogramCard
        title="Monthly strategy return (%)"
        bins={strategyBins}
        stats={strategyStats}
        unit="%"
        positiveTint
      />
    </div>
  );
}

function HistogramCard({
  title,
  bins,
  stats,
  unit,
  positiveTint,
}: {
  title: string;
  bins: { binCenter: number; count: number; binLo: number; binHi: number }[];
  stats: HistogramStats | null;
  unit: string;
  positiveTint: boolean;
}) {
  if (!stats || bins.length === 0) return null;

  // Color the bars by sign: emerald for ≥0, rose for <0. Matches the
  // heatmap palette below, so the eye carries the same encoding down.
  const data = bins.map((b) => ({
    ...b,
    label: `${b.binLo.toFixed(0)} to ${b.binHi.toFixed(0)}`,
    fill: positiveTint && b.binCenter >= 0 ? '#10b981' : '#f43f5e',
    fillOpacity: 0.55,
  }));

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-3">
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <div className="text-[11px] uppercase tracking-wider text-gray-500">{title}</div>
        <div className="font-mono text-[10px] text-gray-500">n = {stats.n}</div>
      </div>
      {/* Stat strip — Mean, Median, σ, Pareto. Each value is mono so the
          eye can scan the numbers; labels are uppercase-tracking-wider to
          match the section headers elsewhere. */}
      <div className="mb-2 grid grid-cols-4 gap-2 text-[10px]">
        <Stat label="Mean" value={`${stats.mean >= 0 ? '+' : ''}${stats.mean.toFixed(2)}${unit}`} valueClass={stats.mean >= 0 ? 'text-emerald-400' : 'text-rose-400'} />
        <Stat label="Median" value={`${stats.median >= 0 ? '+' : ''}${stats.median.toFixed(2)}${unit}`} valueClass={stats.median >= 0 ? 'text-emerald-400' : 'text-rose-400'} />
        <Stat label="Std dev" value={`${stats.std.toFixed(2)}${unit}`} valueClass="text-gray-200" />
        <Stat
          label="Pareto"
          value={`${(stats.paretoShare * 100).toFixed(0)}%`}
          valueClass={stats.paretoShare >= 0.5 ? 'text-amber-400' : 'text-gray-200'}
          title={`Share of total |value| contributed by the most extreme 20% of months. 20% = perfectly uniform · 80% = textbook Pareto 80/20 · higher = even fatter tails. Current: ${(stats.paretoShare * 100).toFixed(1)}% means the top ${Math.max(1, Math.ceil(stats.n * 0.2))} of ${stats.n} months account for ${(stats.paretoShare * 100).toFixed(0)}% of all the moves the strategy made.`}
        />
      </div>
      <div className="h-44 w-full">
        <ResponsiveContainer>
          <BarChart data={data} margin={{ left: 4, right: 8, top: 4, bottom: 4 }}>
            <CartesianGrid stroke="#1f2937" vertical={false} />
            <XAxis
              dataKey="binCenter"
              type="number"
              domain={[bins[0].binLo, bins[bins.length - 1].binHi]}
              tick={{ fontSize: 10, fill: '#6b7280' }}
              tickFormatter={(v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(0)}`}
              ticks={[
                bins[0].binLo,
                bins[0].binLo + (bins[bins.length - 1].binHi - bins[0].binLo) / 4,
                0,
                bins[0].binLo + (3 * (bins[bins.length - 1].binHi - bins[0].binLo)) / 4,
                bins[bins.length - 1].binHi,
              ].filter((v, i, arr) => i === 0 || v !== arr[i - 1])}
              allowDataOverflow={false}
            />
            <YAxis
              tick={{ fontSize: 10, fill: '#6b7280' }}
              width={28}
              allowDecimals={false}
            />
            <Tooltip
              contentStyle={{
                fontSize: 11,
                backgroundColor: '#0b1220',
                border: '1px solid #1f2937',
                color: '#e5e7eb',
              }}
              labelFormatter={(_v: number, payload: { payload?: { binLo?: number; binHi?: number } }[]) => {
                const p = payload && payload[0]?.payload;
                if (!p || p.binLo == null || p.binHi == null) return '';
                return `Bin ${p.binLo >= 0 ? '+' : ''}${p.binLo.toFixed(0)}${unit} to ${p.binHi >= 0 ? '+' : ''}${p.binHi.toFixed(0)}${unit}`;
              }}
              formatter={(v: number | string) => [v, 'Months']}
            />
            {/* Mean line (solid). Always rendered so the eye can compare
                mean-vs-median at a glance (right-skew = mean > median). */}
            <ReferenceLine
              x={stats.mean}
              stroke="#e5e7eb"
              strokeWidth={1.5}
              ifOverflow="extendDomain"
            >
              <Label value="μ" position="top" fill="#e5e7eb" fontSize={10} />
            </ReferenceLine>
            {/* Median line (dashed). Coincides with mean for symmetric
                distributions; diverges visibly when skewed. */}
            <ReferenceLine
              x={stats.median}
              stroke="#fbbf24"
              strokeDasharray="3 3"
              strokeWidth={1.5}
              ifOverflow="extendDomain"
            >
              <Label value="med" position="top" fill="#fbbf24" fontSize={10} />
            </ReferenceLine>
            {/* ±1σ band (dotted). Encloses ~68% of data for a normal
                distribution — visual gauge of how wide the tails really
                are vs how wide a Gaussian would predict. */}
            <ReferenceLine
              x={stats.mean - stats.std}
              stroke="#9ca3af"
              strokeDasharray="2 4"
              strokeWidth={1}
              ifOverflow="extendDomain"
            >
              <Label value="−1σ" position="top" fill="#9ca3af" fontSize={9} />
            </ReferenceLine>
            <ReferenceLine
              x={stats.mean + stats.std}
              stroke="#9ca3af"
              strokeDasharray="2 4"
              strokeWidth={1}
              ifOverflow="extendDomain"
            >
              <Label value="+1σ" position="top" fill="#9ca3af" fontSize={9} />
            </ReferenceLine>
            {/* Per-bar coloring via <Cell> children: emerald for the
                positive half of the distribution, rose for the negative.
                Matches the heatmap palette below so the eye carries the
                same encoding down to the calendar grid. */}
            <Bar dataKey="count" isAnimationActive={false} fillOpacity={0.7}>
              {data.map((d, idx) => (
                <Cell key={idx} fill={d.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  valueClass,
  title,
}: {
  label: string;
  value: string;
  valueClass?: string;
  title?: string;
}) {
  return (
    <div className="rounded border border-gray-800 bg-gray-900/40 px-2 py-1" title={title}>
      <div className="text-[9px] uppercase tracking-wider text-gray-500">{label}</div>
      <div className={`font-mono text-xs ${valueClass ?? 'text-gray-200'}`}>{value}</div>
    </div>
  );
}

type HeatmapMode = 'excess' | 'strategy';

function MonthlyExcessHeatmap({
  cells,
  benchKey,
  gateDecisions,
  onCellClick,
}: {
  cells: StrictWfMonthlyExcessCell[];
  benchKey: string;
  gateDecisions?: StrictWfGateDecision[];
  onCellClick: (cell: StrictWfMonthlyExcessCell) => void;
}) {
  const [mode, setMode] = useState<HeatmapMode>('excess');

  if (!cells || cells.length === 0) return null;

  // (year, month) → gate decision. Only gated variants populate this; the
  // V1 baseline always deploys, so the map is empty and no markers render.
  const gateByYM = new Map<string, StrictWfGateDecision>();
  for (const g of gateDecisions ?? []) gateByYM.set(`${g.year}-${g.month}`, g);
  const hasGate = gateByYM.size > 0;

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
        <div className="flex items-center gap-3">
          <div className="text-[11px] uppercase tracking-wider text-gray-500">
            {headerLabel}
          </div>
          {hasGate && (
            <div className="flex items-center gap-2 text-[10px] text-gray-500" title="Promote/retain gate decision per month (regression head). Solid = a freshly-trained model was adopted; ring = the gate kept the incumbent (model reused).">
              <span className="inline-flex items-center gap-1">
                <span className="h-[6px] w-[6px] rounded-full" style={{ backgroundColor: '#fcd34d' }} />
                adapted
              </span>
              <span className="inline-flex items-center gap-1">
                <span className="h-[6px] w-[6px] rounded-full" style={{ border: '1px solid rgba(252,211,77,0.7)' }} />
                reused
              </span>
            </div>
          )}
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
                  const gate = gateByYM.get(`${y}-${month}`);
                  // Marker reflects the REGRESSION head (the one that
                  // actually retains often). Solid dot = a fresh model was
                  // deployed that month; hollow ring = the gate kept the
                  // incumbent (model reused).
                  const regDeploy = gate?.reg_decision === 'deploy';
                  const regRetain = gate?.reg_decision === 'retain';
                  const gateTip = gate
                    ? `\nModel: regression ${gate.reg_decision ?? '—'}, classification ${gate.cls_decision ?? '—'}`
                    : '';
                  return (
                    <td
                      key={month}
                      title={cellTooltip(cell) + gateTip + (clickable ? '\n(click for details)' : '')}
                      onClick={clickable ? () => onCellClick(cell!) : undefined}
                      className={[
                        'relative h-7 min-w-[44px] rounded text-center font-mono text-[10px] text-gray-100',
                        clickable
                          ? 'cursor-pointer hover:outline hover:outline-1 hover:outline-gray-300/50'
                          : '',
                      ].join(' ')}
                      style={{
                        backgroundColor: cellBg(v),
                      }}
                    >
                      {(regDeploy || regRetain) && (
                        <span
                          className="pointer-events-none absolute left-[3px] top-[3px] h-[6px] w-[6px] rounded-full"
                          style={
                            regDeploy
                              ? { backgroundColor: '#fcd34d' }                       // solid amber = adapted
                              : { border: '1px solid rgba(252,211,77,0.7)' }         // ring = retained
                          }
                        />
                      )}
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
              title={`${benchKey} intra-year max drawdown over the same window the strategy traded. Companion to the Max DD column — surfaces how stressful the year was for the benchmark, which is when the strategy's defensive alpha has the most room to operate.`}
            >
              {benchKey} MaxDD
            </th>
            <th
              className="px-3 py-2 text-right"
              title="VIX peak (intraday high) during the strategy's trading window for the year. The Wall Street 'fear gauge' — higher peaks correlate with high-excess years."
            >
              VIX peak
            </th>
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
              {/* SPY MaxDD: dim text by default; redden when deep (≥10%)
                  so the eye lands on the stress years. Signed positive. */}
              <td
                className={`px-3 py-2 text-right font-mono ${
                  y.benchmark_max_dd_pct == null
                    ? 'text-gray-500'
                    : y.benchmark_max_dd_pct >= 10
                      ? 'text-rose-400/80'
                      : 'text-gray-400'
                }`}
              >
                {y.benchmark_max_dd_pct != null
                  ? pctFmt(y.benchmark_max_dd_pct, false, 1)
                  : '—'}
              </td>
              {/* VIX peak: amber when peak ≥30 (real stress), red ≥50
                  (crash-level), gray otherwise. */}
              <td
                className={`px-3 py-2 text-right font-mono ${
                  y.vix_peak == null
                    ? 'text-gray-500'
                    : y.vix_peak >= 50
                      ? 'text-rose-400'
                      : y.vix_peak >= 30
                        ? 'text-amber-400'
                        : 'text-gray-400'
                }`}
              >
                {y.vix_peak != null ? y.vix_peak.toFixed(1) : '—'}
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
  const hasContent = !!data?.markdown;
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-5">
      <div className="mb-4 flex items-baseline justify-between gap-2 border-b border-gray-800 pb-3">
        <h3 className="text-lg font-semibold text-gray-100">Analysis</h3>
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
        // All styling done via component overrides — no Tailwind prose
        // plugin needed. Each markdown node gets explicit dark-theme classes
        // tuned for the dashboard's monospace-numeric aesthetic.
        <div className="text-sm leading-relaxed text-gray-300">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              h1: ({ children }) => (
                <h2 className="mt-2 mb-3 text-xl font-bold text-gray-100">{children}</h2>
              ),
              h2: ({ children }) => (
                <h3 className="mt-5 mb-2 text-lg font-bold text-emerald-300">{children}</h3>
              ),
              h3: ({ children }) => (
                <h4 className="mt-4 mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-gray-400">{children}</h4>
              ),
              h4: ({ children }) => (
                <h5 className="mt-3 mb-1 text-xs font-semibold uppercase tracking-wider text-gray-500">{children}</h5>
              ),
              p: ({ children }) => (
                <p className="my-2 text-sm leading-relaxed text-gray-300">{children}</p>
              ),
              ul: ({ children }) => (
                <ul className="my-2 ml-5 list-disc space-y-1 marker:text-gray-600">{children}</ul>
              ),
              ol: ({ children }) => (
                <ol className="my-2 ml-5 list-decimal space-y-1 marker:text-gray-600">{children}</ol>
              ),
              li: ({ children }) => (
                <li className="text-sm leading-relaxed text-gray-300">{children}</li>
              ),
              strong: ({ children }) => (
                <strong className="font-semibold text-gray-100">{children}</strong>
              ),
              em: ({ children }) => (
                <em className="italic text-gray-400">{children}</em>
              ),
              code: ({ children }) => (
                <code className="rounded bg-gray-800 px-1.5 py-0.5 font-mono text-[12px] text-emerald-300">{children}</code>
              ),
              blockquote: ({ children }) => (
                <blockquote className="my-3 border-l-2 border-emerald-500/40 pl-4 italic text-gray-400">{children}</blockquote>
              ),
              hr: () => <hr className="my-4 border-gray-800" />,
              // Tables — render with explicit borders and tighter spacing.
              table: ({ children }) => (
                <div className="my-3 overflow-x-auto rounded border border-gray-800">
                  <table className="w-full border-collapse text-xs">{children}</table>
                </div>
              ),
              thead: ({ children }) => (
                <thead className="bg-gray-900/80 text-[10px] uppercase tracking-wider text-gray-400">{children}</thead>
              ),
              tbody: ({ children }) => (
                <tbody className="divide-y divide-gray-800">{children}</tbody>
              ),
              tr: ({ children }) => <tr>{children}</tr>,
              th: ({ children }) => (
                <th className="px-3 py-2 text-left font-medium">{children}</th>
              ),
              td: ({ children }) => (
                <td className="px-3 py-1.5 font-mono text-gray-300">{children}</td>
              ),
            }}
          >
            {data.markdown!}
          </ReactMarkdown>
        </div>
      ) : (
        <div className="text-sm text-gray-500">
          The dashboard analysis card hasn't been generated yet. From this chat,
          type <code className="rounded bg-gray-800 px-1 py-0.5 font-mono text-emerald-300">/wf-analysis</code>
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
