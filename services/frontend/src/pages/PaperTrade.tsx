import { useEffect, useMemo, useState } from 'react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
} from 'recharts';
import { Banknote, Clock, TrendingDown, TrendingUp } from 'lucide-react';
import { usePaperSnapshot, usePaperTrades } from '@/hooks/usePaper';
import { useUniverses } from '@/hooks/useUniverses';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { ErrorMessage } from '@/components/ErrorMessage';
import { EmptyState } from '@/components/EmptyState';
import { UniverseSelector } from '@/components/UniverseSelector';
import type { PaperEquityPoint, PaperPosition, PaperTrade, PaperRunSummary } from '@/api/types';

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
      `Re-rebalanced at 8am CT each trading day.`
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
    `weighted within slice by combined score. Rebalanced at ${tz} (post-open). ` +
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
  const tradesQ = usePaperTrades(runId, 100);

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
          currency={currency}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main view (separated from page so universe switching unmounts cleanly)
// ---------------------------------------------------------------------------

function PaperRunView({
  data, trades, tradesLoading, currency,
}: {
  data: { run: PaperRunSummary; equity_curve: PaperEquityPoint[]; positions: PaperPosition[] };
  trades: PaperTrade[];
  tradesLoading: boolean;
  currency: Currency;
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
      />

      <section className="space-y-3">
        <h2 className="text-base font-semibold text-gray-100">Equity curve</h2>
        <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-4">
          {equity_curve.length > 0 ? (
            <EquityChart points={equity_curve} starting={run.starting_cash} currency={currency} />
          ) : (
            <EmptyState
              title="No equity history yet"
              hint="Need at least one trading day in the backtest window."
            />
          )}
        </div>
      </section>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <PositionsTable positions={positions} currency={currency} />
        <RecentTrades data={trades} loading={tradesLoading} currency={currency} />
      </div>

      <RunMetadata run={run} currency={currency} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Today's two snapshots: 8am CT and 5pm CT
// ---------------------------------------------------------------------------

function SnapshotsToday({ equity_curve, starting_cash, currency }: {
  equity_curve: PaperEquityPoint[];
  starting_cash: number;
  currency: Currency;
}) {
  const lastDate = equity_curve.length > 0
    ? equity_curve[equity_curve.length - 1].trade_date
    : null;
  const latestForDate = useMemo(() => {
    if (!lastDate) return [] as PaperEquityPoint[];
    return equity_curve.filter((p) => p.trade_date === lastDate);
  }, [equity_curve, lastDate]);

  const open8am = latestForDate.find((p) => p.snapshot_kind === 'open_8am_ct');
  const close5pm = latestForDate.find((p) => p.snapshot_kind === 'close_5pm_ct');

  const intradayPnl = open8am && close5pm ? close5pm.equity - open8am.equity : null;

  const isIndia = currency === 'INR';
  const openLabel = isIndia ? '9:15 AM IST' : '8:00 AM CT';
  const closeLabel = isIndia ? '3:30 PM IST' : '5:00 PM CT';

  return (
    <section className="space-y-3">
      <h2 className="text-base font-semibold text-gray-100">
        Today&apos;s snapshots <span className="text-gray-500">({lastDate ?? '—'})</span>
      </h2>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <SnapshotCard
          icon={<Clock className="h-5 w-5 text-gray-400" />}
          time={openLabel}
          label="Pre-market state"
          point={open8am}
          starting={starting_cash}
          currency={currency}
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
          <div className={[
            'mt-2 font-mono text-2xl font-semibold',
            intradayPnl == null
              ? 'text-gray-500'
              : intradayPnl >= 0 ? 'text-emerald-400' : 'text-rose-400',
          ].join(' ')}>
            {intradayPnl == null ? '—' : signedMoneyFmt(intradayPnl, currency)}
          </div>
          <div className="text-[11px] text-gray-500">close − open snapshot today</div>
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
        <div className="flex items-center gap-2">
          {icon}
          <span className="text-sm font-medium text-gray-300">{time}</span>
        </div>
        <div className="mt-2 font-mono text-2xl font-semibold text-gray-500">—</div>
        <div className="text-[11px] text-gray-500">{label}</div>
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

// ---------------------------------------------------------------------------
// Equity chart
// ---------------------------------------------------------------------------

function EquityChart({ points, starting, currency }: {
  points: PaperEquityPoint[];
  starting: number;
  currency: Currency;
}) {
  const series = useMemo(
    () => points
      .filter((p) => p.snapshot_kind === 'close_5pm_ct')
      .map((p) => ({
        date: p.trade_date,
        Equity: p.equity,
      })),
    [points],
  );
  if (series.length === 0) return null;
  const sym = currencySymbol(currency);
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
            formatter={(v: number) => moneyFmt(v, currency)}
            labelFormatter={(d) => `Date: ${d}`}
          />
          <ReferenceLine
            y={starting}
            stroke="#6b7280"
            strokeDasharray="3 3"
            label={{ value: `start ${moneyFmt(starting, currency, 0)}`, position: 'right', fill: '#6b7280', fontSize: 10 }}
          />
          <Line
            type="monotone"
            dataKey="Equity"
            stroke="#34d399"
            strokeWidth={2.2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Positions table + recent trades
// ---------------------------------------------------------------------------

function PositionsTable({ positions, currency }: { positions: PaperPosition[]; currency: Currency }) {
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
      <div className="overflow-hidden rounded-lg border border-gray-800 bg-gray-900/60">
        <table className="w-full text-sm">
          <thead className="bg-gray-900 text-[11px] uppercase tracking-wider text-gray-500">
            <tr>
              <th className="px-3 py-2 text-left">Symbol</th>
              <th className="px-3 py-2 text-left">Side</th>
              <th className="px-3 py-2 text-right">Qty</th>
              <th className="px-3 py-2 text-right">Entry</th>
              <th className="px-3 py-2 text-right">Last</th>
              <th className="px-3 py-2 text-right">P&amp;L</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {positions.map((p) => {
              const isLong = p.side === 'long';
              const isPosPnl = p.unrealized_pnl >= 0;
              return (
                <tr key={p.symbol} className="hover:bg-gray-900/80">
                  <td className="px-3 py-2 font-mono text-gray-100">{p.symbol}</td>
                  <td className="px-3 py-2">
                    <span className={[
                      'inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium uppercase',
                      isLong
                        ? 'bg-emerald-500/15 text-emerald-300'
                        : 'bg-rose-500/15 text-rose-300',
                    ].join(' ')}>
                      {isLong ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                      {p.side}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-gray-300">{p.qty.toFixed(3)}</td>
                  <td className="px-3 py-2 text-right font-mono text-gray-400">{sym}{p.entry_price.toFixed(2)}</td>
                  <td className="px-3 py-2 text-right font-mono text-gray-300">
                    {p.last_price != null ? `${sym}${p.last_price.toFixed(2)}` : '—'}
                  </td>
                  <td className={`px-3 py-2 text-right font-mono ${isPosPnl ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {signedMoneyFmt(p.unrealized_pnl, currency)}
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

function RecentTrades({ data, loading, currency }: { data: PaperTrade[]; loading: boolean; currency: Currency }) {
  const sym = currencySymbol(currency);
  return (
    <section className="space-y-3">
      <h2 className="text-base font-semibold text-gray-100">
        Recent trades <span className="text-gray-500">({data.length})</span>
      </h2>
      <div className="overflow-hidden rounded-lg border border-gray-800 bg-gray-900/60">
        {loading ? (
          <div className="px-3 py-6"><LoadingSpinner label="Loading trades…" /></div>
        ) : data.length === 0 ? (
          <div className="px-3 py-6"><EmptyState title="No trades yet" /></div>
        ) : (
          <table className="w-full text-sm">
            <thead className="bg-gray-900 text-[11px] uppercase tracking-wider text-gray-500">
              <tr>
                <th className="px-3 py-2 text-left">Date</th>
                <th className="px-3 py-2 text-left">Symbol</th>
                <th className="px-3 py-2 text-left">Action</th>
                <th className="px-3 py-2 text-right">Price</th>
                <th className="px-3 py-2 text-right">Realized</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {data.slice(0, 30).map((t, i) => {
                const isLong = t.side.startsWith('long');
                const isOpen = t.side.endsWith('_open');
                const realized = t.realized_pnl ?? 0;
                return (
                  <tr key={`${t.trade_date}-${t.symbol}-${t.side}-${i}`} className="hover:bg-gray-900/80">
                    <td className="px-3 py-2 font-mono text-gray-400">{t.trade_date}</td>
                    <td className="px-3 py-2 font-mono text-gray-100">{t.symbol}</td>
                    <td className="px-3 py-2 text-xs">
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
                    <td className="px-3 py-2 text-right font-mono text-gray-300">{sym}{t.fill_price.toFixed(2)}</td>
                    <td className={[
                      'px-3 py-2 text-right font-mono',
                      isOpen ? 'text-gray-500' :
                        realized >= 0 ? 'text-emerald-400' : 'text-rose-400',
                    ].join(' ')}>
                      {isOpen ? '—' : signedMoneyFmt(realized, currency)}
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
