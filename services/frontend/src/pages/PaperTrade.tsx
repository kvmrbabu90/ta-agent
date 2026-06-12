import { useEffect, useMemo, useState } from 'react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
} from 'recharts';
import { ArrowDown, ArrowUp, Banknote, ChevronsUpDown, Clock, RefreshCw } from 'lucide-react';
import {
  useIntradayMark, useNextDayPicks, usePaperSnapshot, usePaperTrades,
} from '@/hooks/usePaper';
import { useUniverses } from '@/hooks/useUniverses';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { ErrorMessage } from '@/components/ErrorMessage';
import { EmptyState } from '@/components/EmptyState';
import { UniverseSelector } from '@/components/UniverseSelector';
import type { NextDayPicksResponse, PaperEquityPoint, PaperIntradayMark, PaperPosition, PaperTrade, PaperRunSummary } from '@/api/types';

// ---------------------------------------------------------------------------
// Currency formatting — supports USD ($) and INR (₹ with Indian comma format)
// ---------------------------------------------------------------------------

type Currency = 'USD' | 'INR';

function moneyFmt(v: number, currency: Currency, d = 2): string {
  if (currency === 'INR') {
    // Indian numbering system (lakhs, crores)
    return `₹${v.toLocaleString('en-IN', {
      minimumFractionDigits: d,
      maximumFractionDigits: d,
    })}`;
  }
  return `$${v.toFixed(d)}`;
}

function signedMoneyFmt(v: number, currency: Currency, d = 2): string {
  return `${v >= 0 ? '+' : ''}${moneyFmt(v, currency, d)}`;
}

function pctFmt(v: number, d = 2) {
  return `${(v * 100).toFixed(d)}%`;
}

function currencySymbol(currency: Currency): string {
  return currency === 'INR' ? '₹' : '$';
}

/** Current calendar date (YYYY-MM-DD) in the market's timezone — CT for USD
 * runs, IST for INR runs. Used to decide which day "Today's snapshots" shows,
 * independent of where the browser is. */
function marketToday(currency: Currency): string {
  const tz = currency === 'INR' ? 'Asia/Kolkata' : 'America/Chicago';
  // en-CA formats as YYYY-MM-DD, matching the trade_date strings from the API.
  return new Intl.DateTimeFormat('en-CA', { timeZone: tz }).format(new Date());
}

// Universe → run_id + currency mapping. Add more entries here as new live
// runs come online (e.g. a second SP500 run with different sizing).
const UNIVERSE_RUNS: Record<string, { run_id: string; currency: Currency }> = {
  SP500: { run_id: 'default', currency: 'USD' },
  NIFTY100: { run_id: 'live_nifty100', currency: 'INR' },
};

/** Render the run's actual config into prose. */
function describeStrategy(run: PaperRunSummary, currency: Currency): string {
  // Legacy run row — emit the v1 description so we don't lie.
  if (run.holding_days == null) {
    return (
      `Long top ${run.n_long}` +
      (run.n_short > 0
        ? `, short bottom ${run.n_short} (when predicted < -${pctFmt(run.short_threshold, 1)})`
        : '') +
      `. Equal-weight ${moneyFmt(run.position_size, currency, 0)} per position. ` +
      `Rebalanced at the 8:30 CT open each trading day.`
    );
  }
  // v2: overlapping portfolios + conviction weighting + stop-loss.
  const stop = run.stop_loss_enabled
    ? `stop-loss at ${pctFmt(run.stop_buffer_pct ?? 0, 1)} below the ${run.support_lookback_days}-day rolling low`
    : 'no stop-loss';
  const costs =
    run.commission_model === 'ibkr_lite'
      ? 'IBKR Lite costs (sells only)'
      : run.commission_model === 'india_zerodha'
      ? 'Zerodha India retail costs (STT + exchange + GST + stamp)'
      : run.commission_model === 'none'
      ? 'no transaction costs'
      : `costs: ${run.commission_model}`;
  const tz = currency === 'INR' ? '9:20 IST' : '8:35 CT';
  return (
    `Long-only, top ${run.n_long} by conviction (predicted return × direction agreement). ` +
    `Overlapping ${run.holding_days}-day portfolios: open a new slice each trading day, ` +
    `force-close after ${run.holding_days} bars. Slice size = current_equity / ${run.holding_days}, ` +
    `weighted within slice by combined_score / ATR(14) (conviction × inverse-vol). ` +
    `Rebalanced at ${tz} (post-open). ` +
    `${stop[0].toUpperCase()}${stop.slice(1)}; ${costs}.`
  );
}

export function PaperTradePage() {
  const universesQ = useUniverses();
  const universes = useMemo(() => universesQ.data ?? [], [universesQ.data]);
  const [universe, setUniverse] = useState<string>('SP500');

  useEffect(() => {
    if (universes.length > 0 && !universes.some((u) => u.name === universe)) {
      setUniverse(universes[0].name);
    }
  }, [universe, universes]);

  const mapping = UNIVERSE_RUNS[universe] ?? { run_id: 'default', currency: 'USD' as Currency };
  const { run_id: runId, currency } = mapping;

  const snapQ = usePaperSnapshot(runId, 365);
  const picksQ = useNextDayPicks(runId);
  // Recent closes only — OPEN trades are noisy and don't carry realized
  // P&L. Pull the last 100 close events (long_close, short_close,
  // stop_close) so the user sees what's been booked recently.
  const tradesQ = usePaperTrades(runId, 100, true);
  // Manual-trigger mutation. Fires only when the Refresh button is
  // clicked — never auto-polled (yfinance is rate-limit-sensitive and
  // the user only needs this when actively checking mid-day).
  const intradayM = useIntradayMark(runId);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-4 rounded-lg border border-gray-800 bg-gray-900/60 p-4">
        <UniverseSelector
          value={universe}
          onChange={setUniverse}
          universes={universes}
          loading={universesQ.isLoading}
        />
        <span className="text-xs text-gray-500">
          run_id <code className="rounded bg-gray-800 px-1.5 py-0.5 text-gray-300">{runId}</code>
          {' · '}currency <span className="text-gray-300">{currency}</span>
        </span>
      </div>

      {snapQ.isLoading && <LoadingSpinner label={`Loading ${universe} paper-trade snapshot…`} />}
      {snapQ.isError && <ErrorMessage error={snapQ.error} onRetry={() => snapQ.refetch()} />}
      {snapQ.data && (
        <PaperRunView
          data={snapQ.data}
          trades={tradesQ.data?.trades ?? []}
          tradesLoading={tradesQ.isLoading}
          picks={picksQ.data}
          picksLoading={picksQ.isLoading}
          currency={currency}
          intradayMark={intradayM.data}
          intradayLoading={intradayM.isPending}
          intradayError={intradayM.error as Error | null}
          onRefreshIntraday={() => intradayM.mutate()}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main view (separated from page so universe switching unmounts cleanly)
// ---------------------------------------------------------------------------

function PaperRunView({
  data, trades, tradesLoading, picks, picksLoading, currency,
  intradayMark, intradayLoading, intradayError, onRefreshIntraday,
}: {
  data: {
    run: PaperRunSummary;
    equity_curve: PaperEquityPoint[];
    positions: PaperPosition[];
    benchmark_curve?: { trade_date: string; equity: number }[];
    benchmark_symbol?: string | null;
    post_tax_curve?: { trade_date: string; equity: number }[];
    strategy_tax_rate?: number;
  };
  trades: PaperTrade[];
  tradesLoading: boolean;
  picks: NextDayPicksResponse | undefined;
  picksLoading: boolean;
  currency: Currency;
  intradayMark: PaperIntradayMark | undefined;
  intradayLoading: boolean;
  intradayError: Error | null;
  onRefreshIntraday: () => void;
}) {
  const { run, equity_curve, positions } = data;
  const totalReturn = run.final_equity != null
    ? (run.final_equity - run.starting_cash) / run.starting_cash
    : 0;
  const isWinning = totalReturn >= 0;
  const equityLabel = run.final_equity != null ? moneyFmt(run.final_equity, currency) : '—';

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h1 className="text-2xl font-semibold text-gray-100">Paper Trade</h1>
          <p className="text-sm text-gray-500">
            {describeStrategy(run, currency)}
          </p>
        </div>
        <div className="text-right">
          <div className={`font-mono text-3xl font-semibold ${isWinning ? 'text-emerald-400' : 'text-rose-400'}`}>
            {equityLabel}
          </div>
          <div className={`text-sm ${isWinning ? 'text-emerald-400' : 'text-rose-400'}`}>
            {totalReturn >= 0 ? '+' : ''}{pctFmt(totalReturn)} since {run.first_trade_date ?? '—'}
          </div>
        </div>
      </header>

      <SnapshotsToday
        equity_curve={equity_curve}
        starting_cash={run.starting_cash}
        currency={currency}
        intradayMark={intradayMark}
        intradayLoading={intradayLoading}
        intradayError={intradayError}
        onRefreshIntraday={onRefreshIntraday}
      />

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        <section className="space-y-3 xl:col-span-2">
          <div className="flex items-baseline justify-between">
            <h2 className="text-base font-semibold text-gray-100">Equity curve</h2>
            <div className="text-[11px] text-gray-500">
              <span className="text-emerald-400">strategy</span>
              {(data.post_tax_curve?.length ?? 0) > 0 ? (
                <>
                  {' · '}
                  <span className="text-emerald-400/70" title="Estimated post-tax NAV — 30% short-term capital gains applied at end of each fully-elapsed calendar year, reduced-base compounding. This is a forward estimate of the year-end tax bill; actual tax is owed in April and isn't reflected as a cash withdrawal here. Mid-year, this line tracks pre-tax (no haircut applied until Dec 28).">
                    est. after tax
                  </span>
                </>
              ) : null}
              {(data.benchmark_curve?.length ?? 0) > 0 ? (
                <>
                  {' · '}
                  <span className="text-sky-400" title="SPY buy-and-hold rebased to the paper run's starting capital">
                    {data.benchmark_symbol ?? 'SPY'} B&amp;H
                  </span>
                </>
              ) : null}
              <span className="text-gray-500"> · IBKR Lite fees already in strategy</span>
            </div>
          </div>
          <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-4">
            {equity_curve.length > 0 ? (
              <EquityChart
                points={equity_curve}
                benchPoints={data.benchmark_curve}
                postTaxPoints={data.post_tax_curve}
                benchSymbol={data.benchmark_symbol}
                starting={run.starting_cash}
                currency={currency}
                intradayMark={intradayMark}
              />
            ) : (
              <EmptyState
                title="No equity history yet"
                hint="Need at least one trading day in the backtest window."
              />
            )}
          </div>
        </section>
        <NextDayPicksTable picks={picks} loading={picksLoading} currency={currency} />
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <PositionsTable positions={positions} currency={currency} />
        <RecentTrades data={trades} loading={tradesLoading} currency={currency} />
      </div>

      <RunMetadata run={run} currency={currency} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Today's two snapshots: post-open (08:35 CT) and post-close (17:00 CT)
// ---------------------------------------------------------------------------

function SnapshotsToday({
  equity_curve, starting_cash, currency,
  intradayMark, intradayLoading, intradayError, onRefreshIntraday,
}: {
  equity_curve: PaperEquityPoint[];
  starting_cash: number;
  currency: Currency;
  intradayMark: PaperIntradayMark | undefined;
  intradayLoading: boolean;
  intradayError: Error | null;
  onRefreshIntraday: () => void;
}) {
  // Always anchor on the current market date — show "Awaiting" until each
  // run produces its snapshot, rather than falling back to the last day
  // that happens to have data.
  const today = marketToday(currency);
  const todaysPoints = useMemo(
    () => equity_curve.filter((p) => p.trade_date === today),
    [equity_curve, today],
  );

  const open8am = todaysPoints.find((p) => p.snapshot_kind === 'open_8am_ct');
  const close5pm = todaysPoints.find((p) => p.snapshot_kind === 'close_5pm_ct');

  const isIndia = currency === 'INR';
  // 8:30 CT is the NYSE opening auction. Orders fill at that print; the
  // snapshot row is written at the 8:35 CT scheduler tick. The label
  // shows the auction time since that's what the price reflects.
  const openLabel = isIndia ? '9:15 AM IST' : '8:30 AM CT';
  const closeLabel = isIndia ? '3:30 PM IST' : '5:00 PM CT';

  // Local-CT time of the intraday quote, for the card label.
  const intradayLabel = intradayMark
    ? new Date(intradayMark.quoted_at_utc).toLocaleTimeString('en-US', {
        timeZone: isIndia ? 'Asia/Kolkata' : 'America/Chicago',
        hour: '2-digit', minute: '2-digit', hour12: false,
      }) + (isIndia ? ' IST' : ' CT')
    : '—';

  // Intraday P&L only becomes meaningful once the close (post-5pm) snapshot
  // lands. Decompose the open→close equity change into realized and
  // unrealized deltas using the cumulative figures on each snapshot.
  //
  // % returns are computed against the start-of-day equity (8am snapshot),
  // so they answer "what % move did the strategy generate TODAY" — not
  // since-inception. Denominator falls back to starting_cash if equity is
  // somehow 0 (shouldn't happen but defensive).
  const intraday = open8am && close5pm
    ? (() => {
        const totalD = close5pm.equity - open8am.equity;
        const realizedD = close5pm.realized_pnl - open8am.realized_pnl;
        const unrealizedD = close5pm.unrealized_pnl - open8am.unrealized_pnl;
        const denom = open8am.equity > 0 ? open8am.equity : starting_cash;
        return {
          total: totalD,
          realized: realizedD,
          unrealized: unrealizedD,
          totalPct: denom > 0 ? (totalD / denom) * 100 : null,
          realizedPct: denom > 0 ? (realizedD / denom) * 100 : null,
          unrealizedPct: denom > 0 ? (unrealizedD / denom) * 100 : null,
        };
      })()
    : null;

  return (
    <section className="space-y-3">
      <div className="flex items-baseline justify-between gap-2">
        <h2 className="text-base font-semibold text-gray-100">
          Today&apos;s snapshots <span className="text-gray-500">({today})</span>
        </h2>
        <button
          type="button"
          onClick={onRefreshIntraday}
          disabled={intradayLoading}
          className={[
            'inline-flex items-center gap-1.5 rounded-md border border-gray-700 bg-gray-800/60',
            'px-2.5 py-1 text-xs font-medium text-gray-200',
            'hover:bg-gray-800 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed',
          ].join(' ')}
          title="Pull live quotes for held positions and recompute equity. Result is shown in the Intraday mark card and appended to the equity chart for this session only — not persisted."
        >
          <RefreshCw className={`h-3.5 w-3.5 ${intradayLoading ? 'animate-spin' : ''}`} />
          {intradayLoading ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>
      {intradayError && (
        <div className="rounded-md border border-rose-900/60 bg-rose-950/40 px-3 py-2 text-xs text-rose-300">
          Refresh failed: {intradayError.message}
        </div>
      )}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <SnapshotCard
          icon={<Clock className="h-5 w-5 text-gray-400" />}
          time={openLabel}
          label="Post-open mark"
          point={open8am}
          starting={starting_cash}
          currency={currency}
        />
        <IntradayCard
          intraday={intradayMark}
          time={intradayLabel}
          starting={starting_cash}
          currency={currency}
          stale={!intradayMark}
        />
        <SnapshotCard
          icon={<Clock className="h-5 w-5 text-gray-400" />}
          time={closeLabel}
          label="Post-close mark"
          point={close5pm}
          starting={starting_cash}
          currency={currency}
        />
        <div className="rounded-lg border border-gray-800 bg-gray-900/60 px-4 py-3">
          <div className="flex items-center gap-2">
            <Banknote className="h-5 w-5 text-gray-400" />
            <span className="text-sm font-medium text-gray-300">Intraday P&amp;L</span>
          </div>
          {intraday == null ? (
            <>
              <div className="mt-2 font-mono text-2xl font-semibold text-gray-500">
                Awaiting
              </div>
              <div className="text-[11px] text-gray-500">
                updates after the {closeLabel} close run
              </div>
            </>
          ) : (
            <>
              <div className={[
                'mt-2 font-mono text-2xl font-semibold',
                intraday.total >= 0 ? 'text-emerald-400' : 'text-rose-400',
              ].join(' ')}>
                {signedMoneyFmt(intraday.total, currency)}
                {intraday.totalPct != null && (
                  <span className="ml-2 text-base font-normal opacity-80">
                    ({intraday.totalPct >= 0 ? '+' : ''}{intraday.totalPct.toFixed(2)}%)
                  </span>
                )}
              </div>
              <div className="mt-2 grid grid-cols-2 gap-2 text-[11px] text-gray-500">
                <span>
                  realized{' '}
                  <span className={intraday.realized >= 0 ? 'text-emerald-400/80' : 'text-rose-400/80'}>
                    {signedMoneyFmt(intraday.realized, currency)}
                    {intraday.realizedPct != null && (
                      <span className="ml-1 opacity-70">
                        ({intraday.realizedPct >= 0 ? '+' : ''}{intraday.realizedPct.toFixed(2)}%)
                      </span>
                    )}
                  </span>
                </span>
                <span>
                  unrealized{' '}
                  <span className={intraday.unrealized >= 0 ? 'text-emerald-400/80' : 'text-rose-400/80'}>
                    {signedMoneyFmt(intraday.unrealized, currency)}
                    {intraday.unrealizedPct != null && (
                      <span className="ml-1 opacity-70">
                        ({intraday.unrealizedPct >= 0 ? '+' : ''}{intraday.unrealizedPct.toFixed(2)}%)
                      </span>
                    )}
                  </span>
                </span>
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  );
}

function SnapshotCard({ icon, time, label, point, starting, currency }: {
  icon: React.ReactNode;
  time: string;
  label: string;
  point: PaperEquityPoint | undefined;
  starting: number;
  currency: Currency;
}) {
  if (!point) {
    return (
      <div className="rounded-lg border border-gray-800 bg-gray-900/60 px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {icon}
            <span className="text-sm font-medium text-gray-300">{time}</span>
          </div>
          <span className="text-[11px] uppercase tracking-wider text-gray-500">{label}</span>
        </div>
        <div className="mt-2 font-mono text-2xl font-semibold text-gray-500">Awaiting</div>
        <div className="text-[11px] text-gray-500">scheduled {time}</div>
      </div>
    );
  }
  const pnl = point.equity - starting;
  const isPos = pnl >= 0;
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900/60 px-4 py-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {icon}
          <span className="text-sm font-medium text-gray-300">{time}</span>
        </div>
        <span className="text-[11px] uppercase tracking-wider text-gray-500">{label}</span>
      </div>
      <div className={`mt-2 font-mono text-2xl font-semibold ${isPos ? 'text-emerald-400' : 'text-rose-400'}`}>
        {moneyFmt(point.equity, currency)}
      </div>
      <div className={`text-[11px] ${isPos ? 'text-emerald-400/80' : 'text-rose-400/80'}`}>
        {signedMoneyFmt(pnl, currency)} ({isPos ? '+' : ''}{pctFmt(pnl / starting)})
      </div>
      <div className="mt-2 grid grid-cols-3 gap-2 text-[11px] text-gray-500">
        <span>cash <span className="text-gray-300">{moneyFmt(point.cash, currency, 0)}</span></span>
        <span>long <span className="text-emerald-400/80">{moneyFmt(point.long_mv, currency, 0)}</span></span>
        <span>short <span className="text-rose-400/80">{moneyFmt(point.short_mv, currency, 0)}</span></span>
      </div>
    </div>
  );
}

function IntradayCard({ intraday, time, starting, currency, stale }: {
  intraday: PaperIntradayMark | undefined;
  time: string;
  starting: number;
  currency: Currency;
  stale: boolean;
}) {
  if (stale || !intraday) {
    return (
      <div className="rounded-lg border border-gray-800 bg-gray-900/60 px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <RefreshCw className="h-5 w-5 text-gray-400" />
            <span className="text-sm font-medium text-gray-300">Intraday mark</span>
          </div>
          <span className="text-[11px] uppercase tracking-wider text-gray-500">on-demand</span>
        </div>
        <div className="mt-2 font-mono text-2xl font-semibold text-gray-500">—</div>
        <div className="text-[11px] text-gray-500">
          click Refresh to pull live quotes
        </div>
      </div>
    );
  }
  const pnl = intraday.equity - starting;
  const isPos = pnl >= 0;
  const intradayPos = (intraday.intraday_delta ?? 0) >= 0;
  return (
    <div className="rounded-lg border border-amber-900/40 bg-amber-950/10 px-4 py-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <RefreshCw className="h-5 w-5 text-amber-400/70" />
          <span className="text-sm font-medium text-gray-200">{time}</span>
        </div>
        <span className="text-[11px] uppercase tracking-wider text-amber-400/70" title="Computed from live yfinance quotes — not persisted to paper_equity">live mark</span>
      </div>
      <div className={`mt-2 font-mono text-2xl font-semibold ${isPos ? 'text-emerald-400' : 'text-rose-400'}`}>
        {moneyFmt(intraday.equity, currency)}
      </div>
      <div className={`text-[11px] ${isPos ? 'text-emerald-400/80' : 'text-rose-400/80'}`}>
        {signedMoneyFmt(pnl, currency)} ({isPos ? '+' : ''}{pctFmt(pnl / starting)})
      </div>
      {intraday.intraday_delta != null && (
        <div className="mt-1 text-[11px] text-gray-500">
          vs 8:30 open{' '}
          <span className={intradayPos ? 'text-emerald-400/80' : 'text-rose-400/80'}>
            {signedMoneyFmt(intraday.intraday_delta, currency)}
            {intraday.intraday_delta_pct != null && (
              <> ({intraday.intraday_delta_pct >= 0 ? '+' : ''}{intraday.intraday_delta_pct.toFixed(2)}%)</>
            )}
          </span>
        </div>
      )}
      {intraday.quote_failures.length > 0 && (
        <div className="mt-1 text-[10px] text-rose-400/70" title={`Live quote failed for: ${intraday.quote_failures.join(', ')}. Marked at last close instead.`}>
          {intraday.quote_failures.length} quote(s) fell back to last close
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Equity chart
// ---------------------------------------------------------------------------

function EquityChart({ points, benchPoints, postTaxPoints, benchSymbol, starting, currency, intradayMark }: {
  points: PaperEquityPoint[];
  benchPoints?: { trade_date: string; equity: number; snapshot_kind?: 'open_8am_ct' | 'close_5pm_ct' }[];
  postTaxPoints?: { trade_date: string; equity: number }[];
  benchSymbol?: string | null;
  starting: number;
  currency: Currency;
  intradayMark?: PaperIntradayMark | undefined;
}) {
  // Equity-curve series. The leftmost point is the open_8am_ct snapshot
  // of the first trading day — both Strategy and SPY anchored at
  // starting_cash before any trading. All subsequent points are
  // close_5pm_ct (one per trading day). Values are ACTUAL DOLLARS to
  // match the headline NAV / snapshot tiles / intraday P&L (everything
  // else on the page uses actual dollars; normalising just the chart
  // creates the headline-vs-chart mismatch the user noticed).
  //
  // X-axis labels use "Jun 2 AM" / "Jun 2 PM" so the morning baseline
  // doesn't collide with same-day's close on the axis.
  const series = useMemo(() => {
    const sorted = [...points].sort((a, b) => {
      if (a.trade_date !== b.trade_date) return a.trade_date < b.trade_date ? -1 : 1;
      return a.snapshot_kind === 'open_8am_ct' ? -1 : 1;
    });
    const firstOpen = sorted.find((p) => p.snapshot_kind === 'open_8am_ct');
    const closes = sorted.filter((p) => p.snapshot_kind === 'close_5pm_ct');
    const stratPoints = firstOpen ? [firstOpen, ...closes] : closes;

    const benchByKey = new Map(
      (benchPoints ?? []).map((b) => [
        `${b.trade_date}|${b.snapshot_kind ?? 'close_5pm_ct'}`,
        b.equity,
      ]),
    );
    const postByDate = new Map((postTaxPoints ?? []).map((b) => [b.trade_date, b.equity]));

    const basePoints = stratPoints.map((p) => {
      const isAm = p.snapshot_kind === 'open_8am_ct';
      const datePart = p.trade_date;
      const label = `${datePart} ${isAm ? 'AM' : 'PM'}`;
      const benchActual = benchByKey.get(`${datePart}|${p.snapshot_kind}`) ?? null;
      // Post-tax line: morning baseline = starting capital; close = actual value.
      const postActual = isAm ? starting : (postByDate.get(datePart) ?? null);
      return {
        date: label,
        Strategy: p.equity,
        Bench: benchActual,
        'After tax': postActual,
      };
    });

    // Append the intraday mark as the latest point (after the last close).
    // The label uses "LIVE" to make it clear this is the on-demand mark.
    // It's not persisted — refreshing the snapshot wipes it.
    if (intradayMark) {
      basePoints.push({
        date: `${intradayMark.as_of_trade_date} LIVE`,
        Strategy: intradayMark.equity,
        Bench: null,
        'After tax': null,
      });
    }
    return basePoints;
  }, [points, benchPoints, postTaxPoints, starting, intradayMark]);
  if (series.length === 0) return null;
  const sym = currencySymbol(currency);
  // Decide whether to render bench / post-tax. Bench is opt-in; post-tax
  // line only meaningfully differs from Strategy once a calendar year
  // has completed in the run — until then it tracks the pre-tax line.
  const hasBench = (benchPoints?.length ?? 0) > 0;
  const hasPostTax =
    (postTaxPoints?.length ?? 0) > 0 &&
    postTaxPoints!.some((p, i) => {
      const matchIdx = series.findIndex((s) => s.date === p.trade_date);
      if (matchIdx < 0) return false;
      const preTax = series[matchIdx].Strategy;
      return Math.abs(p.equity - preTax) > 0.01;
    });
  return (
    <div className="h-[280px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={series} margin={{ top: 5, right: 16, bottom: 5, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="date" tick={{ fontSize: 11 }} />
          <YAxis
            tick={{ fontSize: 11 }}
            domain={['auto', 'auto']}
            tickFormatter={(v) => `${sym}${v.toFixed(0)}`}
          />
          <Tooltip
            // Show actual dollar value + percentage return vs starting cash.
            formatter={(v: number, name: string) => {
              const pct = starting > 0 ? ((v / starting - 1) * 100) : 0;
              const pctStr = `${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`;
              return [`${moneyFmt(v, currency)} (${pctStr})`, name];
            }}
            labelFormatter={(d) => `Date: ${d}`}
          />
          <ReferenceLine
            y={starting}
            stroke="#6b7280"
            strokeDasharray="3 3"
            label={{ value: `start ${moneyFmt(starting, currency, 0)}`, position: 'right', fill: '#6b7280', fontSize: 10 }}
          />
          {/* Strategy pre-tax (IBKR Lite fees already deducted by the
              paper engine) — primary solid green line. */}
          <Line
            type="monotone"
            dataKey="Strategy"
            stroke="#34d399"
            strokeWidth={2.2}
            dot={false}
            isAnimationActive={false}
          />
          {/* Strategy after 30% STCG — dashed dim green. Only shows
              divergence after a calendar year completes. */}
          {hasPostTax && (
            <Line
              type="monotone"
              dataKey="After tax"
              stroke="#34d399"
              strokeOpacity={0.55}
              strokeDasharray="4 3"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
              connectNulls
            />
          )}
          {/* SPY B&H — sky blue, rebased to starting capital. */}
          {hasBench && (
            <Line
              type="monotone"
              dataKey="Bench"
              name={benchSymbol ?? 'SPY'}
              stroke="#38bdf8"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
              connectNulls
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Positions table + recent trades
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Sortable table helpers — shared by Open positions + Recent closes
// ---------------------------------------------------------------------------

type SortDir = 'asc' | 'desc';
type SortState<K extends string> = { key: K; dir: SortDir } | null;
type SortValue = string | number | null | undefined;

function useSortState<K extends string>(initial: SortState<K> = null) {
  const [sort, setSort] = useState<SortState<K>>(initial);
  const toggle = (key: K) =>
    setSort((cur) =>
      cur && cur.key === key
        ? { key, dir: cur.dir === 'asc' ? 'desc' : 'asc' }
        : { key, dir: 'asc' },
    );
  return { sort, toggle };
}

/** Compare two cells. Nulls always sort last regardless of direction.
 * Numbers compare numerically; everything else compares as a string. */
function compareCells(a: SortValue, b: SortValue, dir: SortDir): number {
  if (a == null && b == null) return 0;
  if (a == null) return 1;
  if (b == null) return -1;
  const r =
    typeof a === 'number' && typeof b === 'number'
      ? a - b
      : String(a).localeCompare(String(b));
  return dir === 'asc' ? r : -r;
}

function sortRows<T, K extends string>(
  rows: T[],
  sort: SortState<K>,
  accessors: Record<K, (row: T) => SortValue>,
): T[] {
  if (!sort) return rows;
  const acc = accessors[sort.key];
  return [...rows].sort((a, b) => compareCells(acc(a), acc(b), sort.dir));
}

function SortableTh<K extends string>({
  label, sortKey, align = 'left', sort, onSort, title,
}: {
  label: string;
  sortKey: K;
  align?: 'left' | 'right';
  sort: SortState<K>;
  onSort: (key: K) => void;
  title?: string;
}) {
  const active = sort?.key === sortKey;
  const dir = active ? sort!.dir : null;
  return (
    <th className={`px-2 py-2 ${align === 'right' ? 'text-right' : 'text-left'}`} title={title}>
      <button
        type="button"
        onClick={() => onSort(sortKey)}
        className={[
          'inline-flex items-center gap-1 uppercase tracking-wider hover:text-gray-300',
          active ? 'text-gray-300' : '',
        ].join(' ')}
      >
        <span>{label}</span>
        {dir === 'asc' ? (
          <ArrowUp className="h-3 w-3" />
        ) : dir === 'desc' ? (
          <ArrowDown className="h-3 w-3" />
        ) : (
          <ChevronsUpDown className="h-3 w-3 opacity-40" />
        )}
      </button>
    </th>
  );
}

type PositionSortKey =
  | 'symbol' | 'qty' | 'entry' | 'entry_date' | 'lots'
  | 'planned_exit' | 'stop' | 'last' | 'pnl' | 'pnl_pct';

// Compute % return from entry to last. For multi-lot positions entry_price
// is the qty-weighted average (API-side). pct = (last - entry) / entry × 100.
function positionPctReturn(p: PaperPosition): number | null {
  if (p.last_price == null || !p.entry_price) return null;
  return ((p.last_price - p.entry_price) / p.entry_price) * 100;
}

const POSITION_ACCESSORS: Record<PositionSortKey, (p: PaperPosition) => SortValue> = {
  symbol: (p) => p.symbol,
  qty: (p) => p.qty,
  entry: (p) => p.entry_price,
  entry_date: (p) => p.entry_date,
  lots: (p) => p.lot_count,
  planned_exit: (p) => p.planned_exit_date,
  stop: (p) => p.stop_level,
  last: (p) => p.last_price,
  pnl: (p) => p.unrealized_pnl,
  pnl_pct: (p) => positionPctReturn(p),
};

function PositionsTable({ positions, currency }: { positions: PaperPosition[]; currency: Currency }) {
  const { sort, toggle } = useSortState<PositionSortKey>();
  const rows = useMemo(
    () => sortRows(positions, sort, POSITION_ACCESSORS),
    [positions, sort],
  );

  if (positions.length === 0) {
    return (
      <section className="space-y-3">
        <h2 className="text-base font-semibold text-gray-100">Open positions</h2>
        <EmptyState title="No open positions" />
      </section>
    );
  }
  const sym = currencySymbol(currency);
  return (
    <section className="space-y-3">
      <h2 className="text-base font-semibold text-gray-100">
        Open positions <span className="text-gray-500">({positions.length})</span>
      </h2>
      <div className="overflow-x-auto rounded-lg border border-gray-800 bg-gray-900/60">
        <table className="w-full text-sm whitespace-nowrap">
          <thead className="bg-gray-900 text-[11px] uppercase tracking-wider text-gray-500">
            <tr>
              <SortableTh label="Sym" sortKey="symbol" sort={sort} onSort={toggle} />
              <SortableTh label="Qty" sortKey="qty" align="right" sort={sort} onSort={toggle} />
              <SortableTh
                label="Entry" sortKey="entry" align="right" sort={sort} onSort={toggle}
                title="Entry price. For multi-lot symbols this is the qty-weighted average across lots."
              />
              <SortableTh
                label="First→Last entry" sortKey="entry_date" align="right" sort={sort} onSort={toggle}
                title="Entry date band: oldest lot → newest lot. For single-lot symbols only the oldest date shows. The NEWEST lot drives the planned exit date."
              />
              <SortableTh
                label="Lots" sortKey="lots" align="right" sort={sort} onSort={toggle}
                title="Number of open lots (entry orders) aggregated into this row."
              />
              <SortableTh
                label="Last exit" sortKey="planned_exit" align="right" sort={sort} onSort={toggle}
                title="Planned forced-close date of the NEWEST lot (= newest entry + holding_days trading days). Once this date is reached the entire symbol is unwound. Earlier lots in the band exit on earlier dates implicitly; this column shows when the last lot leaves."
              />
              <SortableTh
                label="Stop" sortKey="stop" align="right" sort={sort} onSort={toggle}
                title="Stop-loss level. Tightest active stop across guarded lots. If fewer than all lots are guarded (e.g. broken-support skipped the stop on an entry day) the cell shows the count as 'X/Y stopped'."
              />
              <SortableTh label="Last" sortKey="last" align="right" sort={sort} onSort={toggle} />
              <SortableTh label="P&L" sortKey="pnl" align="right" sort={sort} onSort={toggle} />
              <SortableTh
                label="%Ret" sortKey="pnl_pct" align="right" sort={sort} onSort={toggle}
                title="Unrealized return = (last_price − entry_price) / entry_price × 100. For multi-lot symbols entry_price is the qty-weighted average across lots."
              />
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {rows.map((p) => {
              const isPosPnl = p.unrealized_pnl >= 0;
              const pct = positionPctReturn(p);
              return (
                <tr key={p.symbol} className="hover:bg-gray-900/80">
                  <td className="px-2 py-2 font-mono text-gray-100">{p.symbol}</td>
                  <td className="px-2 py-2 text-right font-mono text-gray-300">{p.qty.toFixed(3)}</td>
                  <td
                    className="px-2 py-2 text-right font-mono text-gray-400"
                    title={p.lot_count > 1 ? `qty-weighted avg across ${p.lot_count} lots` : undefined}
                  >
                    {sym}{p.entry_price.toFixed(2)}
                  </td>
                  <td className="px-2 py-2 text-right font-mono text-gray-500">
                    {p.latest_entry_date && p.latest_entry_date !== p.entry_date
                      ? `${p.entry_date} → ${p.latest_entry_date}`
                      : p.entry_date}
                  </td>
                  <td className="px-2 py-2 text-right font-mono text-gray-300">{p.lot_count}</td>
                  <td className="px-2 py-2 text-right font-mono text-gray-500">
                    {p.planned_exit_date ?? '—'}
                  </td>
                  <td
                    className="px-2 py-2 text-right font-mono text-amber-400/80"
                    title={
                      p.stop_lot_count < p.lot_count
                        ? `Only ${p.stop_lot_count}/${p.lot_count} lots have a stop. ${p.lot_count - p.stop_lot_count} lot(s) run naked to the holding-window exit (broken-support guard skipped the stop on entry).`
                        : undefined
                    }
                  >
                    {p.stop_level != null ? (
                      <>
                        {sym}{p.stop_level.toFixed(2)}
                        {p.stop_lot_count < p.lot_count && (
                          <span className="ml-1 text-[10px] text-rose-400/80">
                            ({p.stop_lot_count}/{p.lot_count})
                          </span>
                        )}
                      </>
                    ) : '—'}
                  </td>
                  <td className="px-2 py-2 text-right font-mono text-gray-300">
                    {p.last_price != null ? `${sym}${p.last_price.toFixed(2)}` : '—'}
                  </td>
                  <td
                    className={`px-2 py-2 text-right font-mono ${isPosPnl ? 'text-emerald-400' : 'text-rose-400'}`}
                    title={
                      p.realized_pnl_to_date != null
                        ? `Unrealized on what's still open: ${signedMoneyFmt(p.unrealized_pnl, currency)}. Already-booked realized on prior closes in this cycle: ${signedMoneyFmt(p.realized_pnl_to_date, currency)}.`
                        : undefined
                    }
                  >
                    {signedMoneyFmt(p.unrealized_pnl, currency)}
                    {p.realized_pnl_to_date != null && (
                      <div className="text-[10px] text-gray-500">
                        realized {signedMoneyFmt(p.realized_pnl_to_date, currency)}
                      </div>
                    )}
                  </td>
                  <td
                    className={`px-2 py-2 text-right font-mono ${pct == null ? 'text-gray-500' : pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}
                    title="Unrealized only — does NOT include realized P&L on prior closes of this symbol. See P&L column tooltip for realized."
                  >
                    {pct == null ? '—' : `${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

type TradeSortKey =
  | 'date' | 'symbol' | 'action' | 'entry_date' | 'entry' | 'price' | 'realized' | 'realized_pct';

// % return on a closing trade. Derive from realized_pnl rather than from
// the side label — the simulator emits three close sides ('long_close',
// 'stop_close' for long stop-outs, 'short_close') and the earlier
// implementation incorrectly classified 'stop_close' as a short (the
// startsWith('long') check missed it), flipping the sign on every
// long-side stop. realized_pnl is signed correctly by construction:
//
//   pct = realized_pnl / (qty × entry_price) × 100
//
// Works for any close shape: a long winner has +realized → +pct,
// a long loser (incl. stop-out) has −realized → −pct, a short winner
// has +realized → +pct. No need to know the side label at all.
function tradePctReturn(t: PaperTrade): number | null {
  if (t.side.endsWith('_open')) return null;
  if (t.entry_price == null || !t.entry_price) return null;
  if (t.realized_pnl == null) return null;
  const denom = t.qty * t.entry_price;
  if (denom <= 0) return null;
  return (t.realized_pnl / denom) * 100;
}

const TRADE_ACCESSORS: Record<TradeSortKey, (t: PaperTrade) => SortValue> = {
  date: (t) => t.trade_date,
  symbol: (t) => t.symbol,
  action: (t) => t.side,
  entry_date: (t) => t.entry_date,
  entry: (t) => t.entry_price,
  price: (t) => t.fill_price,
  realized: (t) => t.realized_pnl,
  realized_pct: (t) => tradePctReturn(t),
};

function RecentTrades({ data, loading, currency }: { data: PaperTrade[]; loading: boolean; currency: Currency }) {
  const { sort, toggle } = useSortState<TradeSortKey>();
  // Sort the full set before truncating so a sort reflects all closes, not
  // just the 30 most recent. (Default — no sort — keeps the API's date-desc order.)
  const rows = useMemo(
    () => sortRows(data, sort, TRADE_ACCESSORS).slice(0, 30),
    [data, sort],
  );
  const sym = currencySymbol(currency);
  return (
    <section className="space-y-3">
      <h2 className="text-base font-semibold text-gray-100">
        Recent closes <span className="text-gray-500">({data.length})</span>
      </h2>
      <div className="overflow-x-auto rounded-lg border border-gray-800 bg-gray-900/60">
        {loading ? (
          <div className="px-3 py-6"><LoadingSpinner label="Loading trades…" /></div>
        ) : data.length === 0 ? (
          <div className="px-3 py-6"><EmptyState title="No trades yet" /></div>
        ) : (
          <table className="w-full text-sm whitespace-nowrap">
            <thead className="bg-gray-900 text-[11px] uppercase tracking-wider text-gray-500">
              <tr>
                <SortableTh label="Date" sortKey="date" sort={sort} onSort={toggle} />
                <SortableTh label="Symbol" sortKey="symbol" sort={sort} onSort={toggle} />
                <SortableTh label="Action" sortKey="action" sort={sort} onSort={toggle} />
                <SortableTh label="Entry Date" sortKey="entry_date" align="right" sort={sort} onSort={toggle} />
                <SortableTh label="Entry" sortKey="entry" align="right" sort={sort} onSort={toggle} />
                <SortableTh label="Price" sortKey="price" align="right" sort={sort} onSort={toggle} />
                <SortableTh label="Realized" sortKey="realized" align="right" sort={sort} onSort={toggle} />
                <SortableTh
                  label="% Ret" sortKey="realized_pct" align="right" sort={sort} onSort={toggle}
                  title="Realized return = (exit_price − entry_price) / entry_price × 100. Per-lot — entry_price here is the specific lot's open price, not a portfolio average."
                />
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {rows.map((t, i) => {
                const isLong = t.side.startsWith('long');
                const isOpen = t.side.endsWith('_open');
                const realized = t.realized_pnl ?? 0;
                const pct = tradePctReturn(t);
                return (
                  <tr key={`${t.trade_date}-${t.symbol}-${t.side}-${i}`} className="hover:bg-gray-900/80">
                    <td className="px-2 py-2 font-mono text-gray-400">{t.trade_date}</td>
                    <td className="px-2 py-2 font-mono text-gray-100">{t.symbol}</td>
                    <td className="px-2 py-2 text-xs">
                      <span className={[
                        'rounded px-1.5 py-0.5 font-medium uppercase',
                        isLong && isOpen ? 'bg-emerald-500/15 text-emerald-300' :
                        isLong ? 'bg-emerald-500/10 text-emerald-300/70' :
                        isOpen ? 'bg-rose-500/15 text-rose-300' :
                        'bg-rose-500/10 text-rose-300/70',
                      ].join(' ')}>
                        {t.side.replace('_', ' ')}
                      </span>
                    </td>
                    <td className="px-2 py-2 text-right font-mono text-gray-500">{t.entry_date ?? '—'}</td>
                    <td className="px-2 py-2 text-right font-mono text-gray-400">
                      {t.entry_price != null ? `${sym}${t.entry_price.toFixed(2)}` : '—'}
                    </td>
                    <td className="px-2 py-2 text-right font-mono text-gray-300">{sym}{t.fill_price.toFixed(2)}</td>
                    <td className={[
                      'px-2 py-2 text-right font-mono',
                      isOpen ? 'text-gray-500' :
                        realized >= 0 ? 'text-emerald-400' : 'text-rose-400',
                    ].join(' ')}>
                      {isOpen ? '—' : signedMoneyFmt(realized, currency)}
                    </td>
                    <td className={[
                      'px-2 py-2 text-right font-mono',
                      isOpen || pct == null ? 'text-gray-500' :
                        pct >= 0 ? 'text-emerald-400' : 'text-rose-400',
                    ].join(' ')}>
                      {isOpen || pct == null ? '—' : `${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}

function RunMetadata({ run, currency }: { run: PaperRunSummary; currency: Currency }) {
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900/40 px-4 py-3 text-xs text-gray-500">
      <span className="font-mono">run={run.run_id}</span> · universe={run.universe} · started {run.started_at.slice(0, 10)} ·
      first trade {run.first_trade_date} · last trade {run.last_trade_date} ·
      starting cash {moneyFmt(run.starting_cash, currency, 0)} · realized P&amp;L {run.final_realized_pnl != null ? signedMoneyFmt(run.final_realized_pnl, currency) : '—'}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Next-day picks table — what the engine will trade at tomorrow's open
// ---------------------------------------------------------------------------

function NextDayPicksTable({
  picks,
  loading,
  currency,
}: {
  picks: NextDayPicksResponse | undefined;
  loading: boolean;
  currency: Currency;
}) {
  const sym = currencySymbol(currency);
  return (
    <section className="space-y-3">
      <div className="flex items-baseline justify-between">
        <h2 className="text-base font-semibold text-gray-100">Planned next-day picks</h2>
        {picks?.target_trade_date && (
          <span className="text-[11px] text-gray-500">
            for <span className="font-mono text-gray-300">{picks.target_trade_date}</span>
          </span>
        )}
      </div>
      <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-3">
        {loading ? (
          <div className="px-2 py-6"><LoadingSpinner label="Loading picks…" /></div>
        ) : !picks || picks.picks.length === 0 ? (
          <EmptyState
            title="No picks yet"
            hint="Updates after the 17:00 CT daily_predict step."
          />
        ) : (
          <>
            <div className="mb-2 text-[11px] text-gray-500">
              as_of <span className="font-mono text-gray-300">{picks.as_of}</span>
              {' · '}NAV <span className="font-mono text-gray-300">{moneyFmt(picks.nav, currency, 0)}</span>
              {' · '}slice <span className="font-mono text-gray-300">{moneyFmt(picks.slice_budget, currency, 0)}</span>
            </div>
            {(() => {
              // "Preliminary" = the predictions row for this as_of was
              // last written by the morning 08:35 CT tick. "Final" =
              // last written by the post-close 17:00 CT tick (or any
              // later manual run).
              //
              // Compare write date vs as_of date, NOT raw UTC hour —
              // a late evening run (e.g. 19:00 CDT) writes at 00:00 UTC
              // the NEXT day, so a naive hour check would wrongly flag
              // it as preliminary. Logic:
              //   - write_date > as_of_date           → FINAL (next-day write)
              //   - write_date == as_of_date, h < 20  → PRELIMINARY (morning)
              //   - write_date == as_of_date, h ≥ 20  → FINAL (post-close)
              const written = picks.predictions_written_at;
              if (!written) return null;
              const writeDt = new Date(written + 'Z');
              const writeDateUtc = writeDt.toISOString().slice(0, 10);
              const writeHourUtc = writeDt.getUTCHours();
              const asOfDate = picks.as_of;
              const isPreliminary =
                writeDateUtc === asOfDate && writeHourUtc < 20;
              return isPreliminary ? (
                <div
                  className="mb-2 rounded border border-amber-700/50 bg-amber-900/20 px-2 py-1 text-[10px] text-amber-200"
                  title={`Last write to predictions_log: ${written} UTC. The 08:35 CT pipeline tick generates picks from today's morning data; the 17:00 CT post-close tick overwrites with the canonical end-of-day batch. The post-17:00 CT picks are what actually drive tomorrow's trades.`}
                >
                  preliminary — final picks land after 17:00 CT
                </div>
              ) : (
                <div
                  className="mb-2 rounded border border-emerald-700/50 bg-emerald-900/20 px-2 py-1 text-[10px] text-emerald-200"
                  title={`Last write to predictions_log: ${written} UTC (post-close 17:00 CT tick).`}
                >
                  final — post-close picks
                </div>
              );
            })()}
            <div className="overflow-x-auto">
              <table className="w-full text-sm whitespace-nowrap">
                <thead className="text-[11px] uppercase tracking-wider text-gray-500">
                  <tr>
                    <th className="px-2 py-1 text-left">Sym</th>
                    <th
                      className="px-2 py-1 text-right"
                      title="Conviction score = predicted_return × (1 + (top_q − bot_q)). The ranking signal."
                    >
                      Score
                    </th>
                    <th
                      className="px-2 py-1 text-right"
                      title="Allocation as % of total equity (NAV). At steady-state 25 lots, each lot averages 4% NAV; the inverse-vol weighting concentrates more capital into low-ATR names so individual lot weights range roughly 1-10% NAV. Hover over a row to see the $ value."
                    >
                      %NAV
                    </th>
                    <th
                      className="px-2 py-1 text-right"
                      title="3-day rolling-low × 0.997 — the protective stop. 'brk' = broken support (rolling-low ≥ last close): no stop placed, lot relies on 5-day expiry."
                    >
                      Stop
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800">
                  {picks.picks.map((p) => (
                    <tr key={p.symbol} className={p.broken_support ? 'bg-amber-900/10' : 'hover:bg-gray-900/80'}>
                      <td className="px-2 py-1.5 font-mono text-gray-100">{p.symbol}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-gray-400">
                        {(p.combined_score * 100).toFixed(3)}%
                      </td>
                      <td
                        className="px-2 py-1.5 text-right font-mono text-gray-300"
                        title={`${sym}${Math.round(p.planned_notional).toLocaleString()} of ${sym}${Math.round(picks.nav).toLocaleString()} NAV`}
                      >
                        {picks.nav > 0
                          ? `${(p.planned_notional / picks.nav * 100).toFixed(2)}%`
                          : '—'}
                      </td>
                      <td
                        className={`px-2 py-1.5 text-right font-mono ${p.broken_support ? 'text-amber-400' : 'text-amber-400/80'}`}
                        title={p.broken_support ? 'Broken support — no stop will be placed; lot runs to 5-day expiry only.' : undefined}
                      >
                        {p.broken_support
                          ? 'brk'
                          : p.rolling_low_stop != null
                            ? `${sym}${p.rolling_low_stop.toFixed(2)}`
                            : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </section>
  );
}
