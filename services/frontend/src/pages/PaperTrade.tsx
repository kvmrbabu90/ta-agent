import { useMemo } from 'react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
} from 'recharts';
import { Banknote, Clock, TrendingDown, TrendingUp } from 'lucide-react';
import { usePaperSnapshot, usePaperTrades } from '@/hooks/usePaper';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { ErrorMessage } from '@/components/ErrorMessage';
import { EmptyState } from '@/components/EmptyState';
import type { PaperEquityPoint, PaperPosition, PaperTrade, PaperRunSummary } from '@/api/types';

function pctFmt(v: number, d = 2) { return `${(v * 100).toFixed(d)}%`; }
function dollarFmt(v: number, d = 2) { return `$${v.toFixed(d)}`; }
function dollarSignedFmt(v: number, d = 2) {
  return `${v >= 0 ? '+' : ''}${dollarFmt(v, d)}`;
}

export function PaperTradePage() {
  const snapQ = usePaperSnapshot('default', 365);
  const tradesQ = usePaperTrades('default', 100);

  if (snapQ.isLoading) return <LoadingSpinner label="Loading paper-trade snapshot…" />;
  if (snapQ.isError) return <ErrorMessage error={snapQ.error} onRetry={() => snapQ.refetch()} />;
  if (!snapQ.data) return null;

  const { run, equity_curve, positions } = snapQ.data;
  const totalReturn = run.final_equity != null
    ? (run.final_equity - run.starting_cash) / run.starting_cash
    : 0;
  const isWinning = totalReturn >= 0;

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h1 className="text-2xl font-semibold text-gray-100">Paper Trade</h1>
          <p className="text-sm text-gray-500">
            Strategy: long top {run.n_long}, short bottom {run.n_short} (when predicted &lt; -{pctFmt(run.short_threshold, 1)}).
            Equal-weight {dollarFmt(run.position_size, 0)} per position. Re-rebalanced at 8am CT each trading day.
          </p>
        </div>
        <div className="text-right">
          <div className={`font-mono text-3xl font-semibold ${isWinning ? 'text-emerald-400' : 'text-rose-400'}`}>
            {run.final_equity != null ? dollarFmt(run.final_equity) : '—'}
          </div>
          <div className={`text-sm ${isWinning ? 'text-emerald-400' : 'text-rose-400'}`}>
            {totalReturn >= 0 ? '+' : ''}{pctFmt(totalReturn)} since {run.first_trade_date ?? '—'}
          </div>
        </div>
      </header>

      <SnapshotsToday equity_curve={equity_curve} starting_cash={run.starting_cash} />

      <section className="space-y-3">
        <h2 className="text-base font-semibold text-gray-100">Equity curve</h2>
        <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-4">
          {equity_curve.length > 0 ? (
            <EquityChart points={equity_curve} starting={run.starting_cash} />
          ) : (
            <EmptyState
              title="No equity history yet"
              hint="Need at least one trading day in the backtest window."
            />
          )}
        </div>
      </section>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <PositionsTable positions={positions} />
        <RecentTrades data={tradesQ.data?.trades ?? []} loading={tradesQ.isLoading} />
      </div>

      <RunMetadata run={run} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Today's two snapshots: 8am CT and 5pm CT
// ---------------------------------------------------------------------------

function SnapshotsToday({ equity_curve, starting_cash }: {
  equity_curve: PaperEquityPoint[];
  starting_cash: number;
}) {
  // Find the latest trade_date that has any snapshot.
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

  return (
    <section className="space-y-3">
      <h2 className="text-base font-semibold text-gray-100">
        Today&apos;s snapshots <span className="text-gray-500">({lastDate ?? '—'})</span>
      </h2>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <SnapshotCard
          icon={<Clock className="h-5 w-5 text-gray-400" />}
          time="8:00 AM CT"
          label="Pre-market state"
          point={open8am}
          starting={starting_cash}
        />
        <SnapshotCard
          icon={<Clock className="h-5 w-5 text-gray-400" />}
          time="5:00 PM CT"
          label="Post-close mark"
          point={close5pm}
          starting={starting_cash}
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
            {intradayPnl == null ? '—' : dollarSignedFmt(intradayPnl)}
          </div>
          <div className="text-[11px] text-gray-500">close − open snapshot today</div>
        </div>
      </div>
    </section>
  );
}

function SnapshotCard({ icon, time, label, point, starting }: {
  icon: React.ReactNode;
  time: string;
  label: string;
  point: PaperEquityPoint | undefined;
  starting: number;
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
        {dollarFmt(point.equity)}
      </div>
      <div className={`text-[11px] ${isPos ? 'text-emerald-400/80' : 'text-rose-400/80'}`}>
        {dollarSignedFmt(pnl)} ({isPos ? '+' : ''}{pctFmt(pnl / starting)})
      </div>
      <div className="mt-2 grid grid-cols-3 gap-2 text-[11px] text-gray-500">
        <span>cash <span className="text-gray-300">{dollarFmt(point.cash, 0)}</span></span>
        <span>long <span className="text-emerald-400/80">{dollarFmt(point.long_mv, 0)}</span></span>
        <span>short <span className="text-rose-400/80">{dollarFmt(point.short_mv, 0)}</span></span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Equity chart
// ---------------------------------------------------------------------------

function EquityChart({ points, starting }: { points: PaperEquityPoint[]; starting: number }) {
  // Use only the close_5pm snapshot for the daily equity curve (single point per day).
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
  return (
    <div className="h-[280px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={series} margin={{ top: 5, right: 16, bottom: 5, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="date" tick={{ fontSize: 11 }} />
          <YAxis
            tick={{ fontSize: 11 }}
            domain={['auto', 'auto']}
            tickFormatter={(v) => `$${v.toFixed(0)}`}
          />
          <Tooltip
            formatter={(v: number) => `$${v.toFixed(2)}`}
            labelFormatter={(d) => `Date: ${d}`}
          />
          <ReferenceLine
            y={starting}
            stroke="#6b7280"
            strokeDasharray="3 3"
            label={{ value: `start $${starting}`, position: 'right', fill: '#6b7280', fontSize: 10 }}
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

function PositionsTable({ positions }: { positions: PaperPosition[] }) {
  if (positions.length === 0) {
    return (
      <section className="space-y-3">
        <h2 className="text-base font-semibold text-gray-100">Open positions</h2>
        <EmptyState title="No open positions" />
      </section>
    );
  }
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
                  <td className="px-3 py-2 text-right font-mono text-gray-400">${p.entry_price.toFixed(2)}</td>
                  <td className="px-3 py-2 text-right font-mono text-gray-300">
                    {p.last_price != null ? `$${p.last_price.toFixed(2)}` : '—'}
                  </td>
                  <td className={`px-3 py-2 text-right font-mono ${isPosPnl ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {dollarSignedFmt(p.unrealized_pnl)}
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

function RecentTrades({ data, loading }: { data: PaperTrade[]; loading: boolean }) {
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
                    <td className="px-3 py-2 text-right font-mono text-gray-300">${t.fill_price.toFixed(2)}</td>
                    <td className={[
                      'px-3 py-2 text-right font-mono',
                      isOpen ? 'text-gray-500' :
                        realized >= 0 ? 'text-emerald-400' : 'text-rose-400',
                    ].join(' ')}>
                      {isOpen ? '—' : dollarSignedFmt(realized)}
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

function RunMetadata({ run }: { run: PaperRunSummary }) {
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900/40 px-4 py-3 text-xs text-gray-500">
      <span className="font-mono">run={run.run_id}</span> · universe={run.universe} · started {run.started_at.slice(0, 10)} ·
      first trade {run.first_trade_date} · last trade {run.last_trade_date} ·
      starting cash {dollarFmt(run.starting_cash, 0)} · realized P&amp;L {run.final_realized_pnl != null ? dollarSignedFmt(run.final_realized_pnl) : '—'}
    </div>
  );
}
