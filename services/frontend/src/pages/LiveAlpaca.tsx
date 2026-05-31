/**
 * Live Alpaca tab — mirrors LiveIbkr but for Alpaca.
 *
 * Unlike IBKR (where mode is auto-detected from Gateway), Alpaca's mode is
 * driven server-side by the ALPACA_MODE env var + which key pair is set.
 * The dashboard reads whichever the sync loop is currently writing.
 *
 * Mode badge contract is identical:
 *   gray DISCONNECTED · emerald PAPER · red+pulsing LIVE.
 * Losing focus on mode is the #1 way to fire real-money orders by accident.
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { API_BASE_URL } from '@/api/client';

const API_BASE = API_BASE_URL;

type Status = {
  connected: boolean;
  mode: 'paper' | 'live' | null;
  account_number: string | null;
  account_id: string | null;
  status: string | null;
  currency: string | null;
  last_seen_at: string | null;
  reason: string | null;
};

type Position = {
  symbol: string;
  qty: number;
  avg_entry_price: number;
  mkt_price: number | null;
  mkt_value: number | null;
  unreal_pnl: number | null;
  unreal_pnl_pct: number | null;
  side: string;
};

type OrderRow = {
  order_id: string;
  symbol: string;
  side: string;
  qty: number;
  order_type: string;
  status: string;
  filled_qty: number;
  filled_avg_price: number | null;
  submitted_at: string;
};

type EquityPoint = {
  snapshot_at: string;
  nav: number;
  cash: number;
  long_mv: number;
  equity: number;
  buying_power: number;
};

type ReconRow = {
  trade_date: string;
  symbol: string;
  side: string;
  sim_price: number;
  actual_price: number;
  qty: number;
  notional: number;
  slippage_bps: number;
  commission_usd: number | null;
  commission_bps: number | null;
};

type ReconResp = {
  rows: ReconRow[];
  n: number;
  mean_slip_bps: number | null;
  median_slip_bps: number | null;
  total_commission_usd: number;
  total_notional: number;
};

type EngineStatus = {
  status: 'stopped' | 'running' | 'error';
  pid: number | null;
  sync_pid: number | null;
  engine_alive: boolean;
  sync_alive: boolean;
  started_at: string | null;
  last_run_at: string | null;
  last_run_date: string | null;
  last_run_status: string | null;
  last_error: string | null;
  heartbeat_at: string | null;
  stopped_at: string | null;
};

type PendingSignal = {
  id: number;
  signal_date: string;
  intended_action: string;
  symbol: string;
  qty: number;
  target_price: number | null;
  status: string;
};

function fmt(n: number | null | undefined, d = 2): string {
  if (n == null) return '—';
  return n.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
}

function fmtUsd(n: number | null | undefined): string {
  if (n == null) return '—';
  return '$' + n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export function LiveAlpacaPage() {
  const queryClient = useQueryClient();
  const [selectedSignals, setSelectedSignals] = useState<Set<number>>(new Set());
  const [confirmAccount, setConfirmAccount] = useState('');

  const status = useQuery<Status>({
    queryKey: ['alpaca', 'status'],
    queryFn: () => fetch(API_BASE + '/live-alpaca/status').then(r => r.json()),
    refetchInterval: 5000,
  });
  const positions = useQuery<Position[]>({
    queryKey: ['alpaca', 'positions'],
    queryFn: () => fetch(API_BASE + '/live-alpaca/positions').then(r => r.json()),
    refetchInterval: 10000,
  });
  const orders = useQuery<OrderRow[]>({
    queryKey: ['alpaca', 'orders'],
    queryFn: () => fetch(API_BASE + '/live-alpaca/orders/today').then(r => r.json()),
    refetchInterval: 10000,
  });
  const equity = useQuery<EquityPoint[]>({
    queryKey: ['alpaca', 'equity'],
    queryFn: () => fetch(API_BASE + '/live-alpaca/equity').then(r => r.json()),
    refetchInterval: 30000,
  });
  const recon = useQuery<ReconResp>({
    queryKey: ['alpaca', 'reconciliation'],
    queryFn: () => fetch(API_BASE + '/live-alpaca/reconciliation').then(r => r.json()),
    refetchInterval: 30000,
  });
  const pending = useQuery<PendingSignal[]>({
    queryKey: ['alpaca', 'pending'],
    queryFn: () => fetch(API_BASE + '/live-alpaca/signals/pending').then(r => r.json()),
    refetchInterval: 5000,
  });
  const engine = useQuery<EngineStatus>({
    queryKey: ['alpaca', 'engine'],
    queryFn: () => fetch(API_BASE + '/live-alpaca/engine/status').then(r => r.json()),
    refetchInterval: 5000,
  });

  const engineStartMut = useMutation({
    mutationFn: async () => {
      const r = await fetch(API_BASE + '/live-alpaca/engine/start', { method: 'POST' });
      if (!r.ok) throw new Error((await r.json()).detail || 'start failed');
      return r.json();
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['alpaca', 'engine'] }),
  });
  const engineStopMut = useMutation({
    mutationFn: async () => {
      const r = await fetch(API_BASE + '/live-alpaca/engine/stop', { method: 'POST' });
      if (!r.ok) throw new Error((await r.json()).detail || 'stop failed');
      return r.json();
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['alpaca', 'engine'] }),
  });

  const approveMut = useMutation({
    mutationFn: async (payload: { signal_ids: number[]; confirm_live_account_number: string | null }) => {
      const r = await fetch(API_BASE + '/live-alpaca/signals/approve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          signal_ids: payload.signal_ids,
          approved_by: 'dashboard-user',
          confirm_live_account_number: payload.confirm_live_account_number,
        }),
      });
      if (!r.ok) throw new Error(await r.text());
      return r.json();
    },
    onSuccess: () => {
      setSelectedSignals(new Set());
      queryClient.invalidateQueries({ queryKey: ['alpaca'] });
    },
  });
  const rejectMut = useMutation({
    mutationFn: async (ids: number[]) => {
      const r = await fetch(API_BASE + '/live-alpaca/signals/reject', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ signal_ids: ids, reason: 'dashboard reject' }),
      });
      if (!r.ok) throw new Error(await r.text());
      return r.json();
    },
    onSuccess: () => {
      setSelectedSignals(new Set());
      queryClient.invalidateQueries({ queryKey: ['alpaca'] });
    },
  });

  const s = status.data;
  const isLive = s?.connected && s.mode === 'live';
  const isPaper = s?.connected && s.mode === 'paper';

  const totalNav = equity.data && equity.data.length > 0 ? equity.data[equity.data.length - 1].nav : null;
  const totalCash = equity.data && equity.data.length > 0 ? equity.data[equity.data.length - 1].cash : null;
  const totalLongMv = equity.data && equity.data.length > 0 ? equity.data[equity.data.length - 1].long_mv : null;
  const totalBP = equity.data && equity.data.length > 0 ? equity.data[equity.data.length - 1].buying_power : null;

  const handleSelect = (id: number) => {
    const next = new Set(selectedSignals);
    if (next.has(id)) next.delete(id); else next.add(id);
    setSelectedSignals(next);
  };

  const handleApprove = () => {
    if (selectedSignals.size === 0) return;
    if (isLive && confirmAccount !== s?.account_number) {
      alert(`LIVE mode requires you to type the connected account number (${s?.account_number}) to confirm.`);
      return;
    }
    approveMut.mutate({
      signal_ids: Array.from(selectedSignals),
      confirm_live_account_number: isLive ? s!.account_number : null,
    });
  };

  return (
    <div className="space-y-4">
      {/* ============== STATUS HEADER ============== */}
      <section
        className={
          'rounded-lg border p-4 ' +
          (isLive
            ? 'border-rose-500 bg-rose-950/40'
            : isPaper
              ? 'border-emerald-700 bg-emerald-950/30'
              : 'border-gray-800 bg-gray-900/60')
        }
      >
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-gray-100">Live Alpaca</h1>
            <p className="text-xs text-gray-400">
              Mode is driven by <code className="font-mono">ALPACA_MODE</code> env var on the server. Restart the sync loop after switching.
            </p>
          </div>
          <ModeBadge status={s} />
        </div>
        {s && (
          <div className="mt-3 grid grid-cols-2 gap-3 text-xs sm:grid-cols-5">
            <Field label="Mode" value={s.mode ?? '—'} highlight={isLive ? 'rose' : isPaper ? 'emerald' : 'gray'} />
            <Field label="Account #" value={s.account_number ?? '—'} mono />
            <Field label="Status" value={s.status ?? '—'} />
            <Field label="Currency" value={s.currency ?? '—'} mono />
            <Field label="Last sync" value={s.last_seen_at ? new Date(s.last_seen_at).toLocaleTimeString() : '—'} mono />
          </div>
        )}
        {s?.reason && (
          <div className="mt-2 rounded bg-gray-900/80 p-2 text-xs text-amber-400">{s.reason}</div>
        )}
        {isLive && (
          <div className="mt-3 rounded border border-rose-500 bg-rose-950 p-3 text-xs text-rose-200">
            🚨 <strong>LIVE MODE.</strong> Orders submitted here will use real capital from account{' '}
            <code className="font-mono">{s.account_number}</code>. To approve any signal you must type the account number below.
          </div>
        )}
      </section>

      {/* ============== KUBERA ENGINE CONTROL ============== */}
      <section className="rounded-lg border border-gray-800 bg-gray-900/60 p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h2 className="text-[11px] uppercase tracking-wider text-gray-500">Kubera engine</h2>
            <EngineBadge engine={engine.data} />
            {engine.data?.last_run_date && (
              <span className="text-xs text-gray-500">
                last run <span className="font-mono text-gray-300">{engine.data.last_run_date}</span>
                {engine.data.last_run_status && (
                  <span
                    className={
                      'ml-1 ' +
                      (engine.data.last_run_status === 'ok'
                        ? 'text-emerald-400'
                        : engine.data.last_run_status === 'error'
                          ? 'text-rose-400'
                          : 'text-amber-400')
                    }
                  >
                    ({engine.data.last_run_status})
                  </span>
                )}
              </span>
            )}
            {engine.data?.heartbeat_at && engine.data.status === 'running' && (
              <span className="text-xs text-gray-500">
                heartbeat <span className="font-mono text-gray-400">{new Date(engine.data.heartbeat_at).toLocaleTimeString()}</span>
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {engine.data?.status === 'running' ? (
              <button
                onClick={() => engineStopMut.mutate()}
                disabled={engineStopMut.isPending}
                className="rounded bg-rose-600 px-4 py-1.5 text-xs font-semibold text-white hover:bg-rose-500 disabled:opacity-40"
              >
                {engineStopMut.isPending ? 'Stopping…' : 'Stop Kubera'}
              </button>
            ) : (
              <button
                onClick={() => engineStartMut.mutate()}
                disabled={engineStartMut.isPending || !status.data?.connected}
                className="rounded bg-emerald-600 px-4 py-1.5 text-xs font-semibold text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-40"
                title={!status.data?.connected ? 'Connect to Alpaca first' : undefined}
              >
                {engineStartMut.isPending ? 'Starting…' : 'Start Kubera'}
              </button>
            )}
          </div>
        </div>
        {engine.data?.last_error && (
          <div className="mt-2 rounded bg-rose-950/40 p-2 text-xs text-rose-300">
            last error: <span className="font-mono">{engine.data.last_error}</span>
          </div>
        )}
        {engine.data?.status === 'running' && (
          <p className="mt-2 text-[11px] text-gray-500">
            Engine + sync are running as detached processes (engine pid <span className="font-mono">{engine.data.pid ?? '—'}</span>, sync pid <span className="font-mono">{engine.data.sync_pid ?? '—'}</span>). They survive API restarts and this terminal closing. Daily rotation fires at market open.
          </p>
        )}
      </section>

      {/* ============== NAV / CASH / MV / BP ============== */}
      <section className="grid grid-cols-4 gap-3">
        <Tile label="NAV" value={fmtUsd(totalNav)} />
        <Tile label="Cash" value={fmtUsd(totalCash)} />
        <Tile label="Long market value" value={fmtUsd(totalLongMv)} />
        <Tile label="Buying power" value={fmtUsd(totalBP)} />
      </section>

      {/* ============== PENDING-APPROVAL ============== */}
      <section className="rounded-lg border border-gray-800 bg-gray-900/60 p-3">
        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-[11px] uppercase tracking-wider text-gray-500">Pending approval</h2>
          <div className="flex items-center gap-2">
            {isLive && (
              <input
                type="text"
                placeholder="type account # to confirm"
                value={confirmAccount}
                onChange={e => setConfirmAccount(e.target.value)}
                className="rounded border border-rose-600 bg-rose-950 px-2 py-1 text-xs font-mono text-rose-100 placeholder:text-rose-300/50"
              />
            )}
            <button
              onClick={handleApprove}
              disabled={selectedSignals.size === 0 || approveMut.isPending}
              className="rounded bg-emerald-600 px-3 py-1 text-xs font-semibold text-white disabled:cursor-not-allowed disabled:opacity-40"
            >
              {approveMut.isPending ? 'Submitting…' : `Approve ${selectedSignals.size}`}
            </button>
            <button
              onClick={() => rejectMut.mutate(Array.from(selectedSignals))}
              disabled={selectedSignals.size === 0 || rejectMut.isPending}
              className="rounded border border-gray-700 px-3 py-1 text-xs text-gray-300 disabled:opacity-40"
            >
              Reject {selectedSignals.size}
            </button>
          </div>
        </div>
        {pending.data && pending.data.length > 0 ? (
          <table className="w-full text-sm">
            <thead className="text-[10px] uppercase text-gray-500">
              <tr>
                <th className="w-8" />
                <th className="text-left">Date</th>
                <th className="text-left">Action</th>
                <th className="text-left">Symbol</th>
                <th className="text-right">Qty</th>
                <th className="text-right">Target $</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {pending.data.map(p => (
                <tr key={p.id} className="hover:bg-gray-900/80">
                  <td>
                    <input
                      type="checkbox"
                      checked={selectedSignals.has(p.id)}
                      onChange={() => handleSelect(p.id)}
                    />
                  </td>
                  <td className="py-1 font-mono text-xs text-gray-400">{p.signal_date}</td>
                  <td className="text-xs">{p.intended_action}</td>
                  <td className="py-1 font-mono text-sm text-gray-200">{p.symbol}</td>
                  <td className="py-1 text-right font-mono text-xs">{fmt(p.qty, 0)}</td>
                  <td className="py-1 text-right font-mono text-xs text-gray-400">{fmtUsd(p.target_price)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="text-xs text-gray-500">No signals waiting for approval.</p>
        )}
      </section>

      {/* ============== POSITIONS ============== */}
      <section className="rounded-lg border border-gray-800 bg-gray-900/60 p-3">
        <h2 className="mb-2 text-[11px] uppercase tracking-wider text-gray-500">Current positions</h2>
        {positions.data && positions.data.length > 0 ? (
          <table className="w-full text-sm">
            <thead className="text-[10px] uppercase text-gray-500">
              <tr>
                <th className="text-left">Symbol</th>
                <th className="text-left">Side</th>
                <th className="text-right">Qty</th>
                <th className="text-right">Avg entry</th>
                <th className="text-right">Mkt price</th>
                <th className="text-right">Mkt value</th>
                <th className="text-right">Unrealized PnL</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {positions.data.map(p => (
                <tr key={p.symbol}>
                  <td className="py-1 font-mono text-sm text-gray-200">{p.symbol}</td>
                  <td className="py-1 text-xs">{p.side}</td>
                  <td className="py-1 text-right font-mono text-xs">{fmt(p.qty, 0)}</td>
                  <td className="py-1 text-right font-mono text-xs text-gray-400">{fmtUsd(p.avg_entry_price)}</td>
                  <td className="py-1 text-right font-mono text-xs">{fmtUsd(p.mkt_price)}</td>
                  <td className="py-1 text-right font-mono text-xs">{fmtUsd(p.mkt_value)}</td>
                  <td
                    className={
                      'py-1 text-right font-mono text-xs ' +
                      ((p.unreal_pnl ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400')
                    }
                  >
                    {fmtUsd(p.unreal_pnl)}
                    {p.unreal_pnl_pct != null && (
                      <span className="ml-1 text-[10px] text-gray-500">
                        ({fmt(p.unreal_pnl_pct * 100, 2)}%)
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="text-xs text-gray-500">No positions.</p>
        )}
      </section>

      {/* ============== ORDERS TODAY ============== */}
      <section className="rounded-lg border border-gray-800 bg-gray-900/60 p-3">
        <h2 className="mb-2 text-[11px] uppercase tracking-wider text-gray-500">Orders today</h2>
        {orders.data && orders.data.length > 0 ? (
          <table className="w-full text-sm">
            <thead className="text-[10px] uppercase text-gray-500">
              <tr>
                <th className="text-left">Time</th>
                <th className="text-left">Symbol</th>
                <th className="text-left">Side</th>
                <th className="text-right">Qty</th>
                <th className="text-left">Type</th>
                <th className="text-left">Status</th>
                <th className="text-right">Filled @</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {orders.data.map(o => (
                <tr key={o.order_id}>
                  <td className="py-1 font-mono text-xs text-gray-400">{new Date(o.submitted_at).toLocaleTimeString()}</td>
                  <td className="py-1 font-mono text-sm text-gray-200">{o.symbol}</td>
                  <td className="py-1 text-xs">{o.side}</td>
                  <td className="py-1 text-right font-mono text-xs">{fmt(o.qty, 0)}</td>
                  <td className="py-1 text-xs">{o.order_type}</td>
                  <td className="py-1 text-xs">{o.status}</td>
                  <td className="py-1 text-right font-mono text-xs">{fmtUsd(o.filled_avg_price)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="text-xs text-gray-500">No orders submitted today.</p>
        )}
      </section>

      {/* ============== RECONCILIATION ============== */}
      <section className="rounded-lg border border-gray-800 bg-gray-900/60 p-3">
        <h2 className="mb-2 text-[11px] uppercase tracking-wider text-gray-500">
          Reconciliation — Kubera paper vs Alpaca actual
        </h2>
        {recon.data && (
          <div className="mb-3 grid grid-cols-4 gap-2 text-xs">
            <Stat label="Fills" value={String(recon.data.n)} />
            <Stat
              label="Mean slip"
              value={recon.data.mean_slip_bps != null ? `${fmt(recon.data.mean_slip_bps, 1)} bps` : '—'}
              tone={recon.data.mean_slip_bps != null && recon.data.mean_slip_bps > 25 ? 'rose' : 'gray'}
            />
            <Stat
              label="Median slip"
              value={recon.data.median_slip_bps != null ? `${fmt(recon.data.median_slip_bps, 1)} bps` : '—'}
            />
            <Stat label="Total commission" value={fmtUsd(recon.data.total_commission_usd)} />
          </div>
        )}
        {recon.data && recon.data.rows.length > 0 ? (
          <table className="w-full text-sm">
            <thead className="text-[10px] uppercase text-gray-500">
              <tr>
                <th className="text-left">Date</th>
                <th className="text-left">Symbol</th>
                <th className="text-left">Side</th>
                <th className="text-right">Sim $</th>
                <th className="text-right">Actual $</th>
                <th className="text-right">Slippage</th>
                <th className="text-right">Commission</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {recon.data.rows.slice(0, 100).map((r, i) => (
                <tr key={i}>
                  <td className="py-1 font-mono text-xs text-gray-400">{r.trade_date}</td>
                  <td className="py-1 font-mono text-sm text-gray-200">{r.symbol}</td>
                  <td className="py-1 text-xs">{r.side}</td>
                  <td className="py-1 text-right font-mono text-xs">{fmtUsd(r.sim_price)}</td>
                  <td className="py-1 text-right font-mono text-xs">{fmtUsd(r.actual_price)}</td>
                  <td
                    className={
                      'py-1 text-right font-mono text-xs ' +
                      (r.slippage_bps > 25 ? 'text-rose-400' : r.slippage_bps > 0 ? 'text-amber-400' : 'text-emerald-400')
                    }
                  >
                    {fmt(r.slippage_bps, 1)} bps
                  </td>
                  <td className="py-1 text-right font-mono text-xs text-gray-400">{fmtUsd(r.commission_usd)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="text-xs text-gray-500">No fills yet to reconcile.</p>
        )}
      </section>
    </div>
  );
}

function EngineBadge({ engine }: { engine: EngineStatus | undefined }) {
  if (!engine || engine.status === 'stopped') {
    return (
      <span className="rounded-full bg-gray-800 px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-gray-400">
        ● STOPPED
      </span>
    );
  }
  if (engine.status === 'error') {
    return (
      <span className="rounded-full bg-rose-700 px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-white">
        ● ERROR
      </span>
    );
  }
  return (
    <span className="rounded-full bg-emerald-700 px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-white">
      ● RUNNING
    </span>
  );
}

function ModeBadge({ status }: { status: Status | undefined }) {
  if (!status || !status.connected) {
    return (
      <span className="rounded-full bg-gray-800 px-3 py-1 text-xs font-bold uppercase tracking-wider text-gray-400">
        ● DISCONNECTED
      </span>
    );
  }
  if (status.mode === 'live') {
    return (
      <span className="rounded-full bg-rose-600 px-3 py-1 text-xs font-bold uppercase tracking-wider text-white animate-pulse">
        🚨 LIVE
      </span>
    );
  }
  return (
    <span className="rounded-full bg-emerald-700 px-3 py-1 text-xs font-bold uppercase tracking-wider text-white">
      ● PAPER
    </span>
  );
}

function Field({ label, value, mono, highlight }: { label: string; value: string; mono?: boolean; highlight?: 'rose' | 'emerald' | 'gray' }) {
  const valueClass = mono ? 'font-mono' : '';
  const toneClass = highlight === 'rose' ? 'text-rose-300' : highlight === 'emerald' ? 'text-emerald-300' : 'text-gray-100';
  return (
    <div>
      <div className="text-[9px] uppercase tracking-wider text-gray-500">{label}</div>
      <div className={`text-xs ${valueClass} ${toneClass}`}>{value}</div>
    </div>
  );
}

function Tile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-3">
      <div className="text-[10px] uppercase tracking-wider text-gray-500">{label}</div>
      <div className="mt-1 font-mono text-lg text-gray-100">{value}</div>
    </div>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: 'rose' | 'gray' }) {
  return (
    <div className="rounded border border-gray-800 bg-gray-900/40 px-2 py-1">
      <div className="text-[9px] uppercase tracking-wider text-gray-500">{label}</div>
      <div className={`font-mono text-xs ${tone === 'rose' ? 'text-rose-400' : 'text-gray-200'}`}>{value}</div>
    </div>
  );
}
