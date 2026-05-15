import { Activity, LayoutDashboard, LineChart, Wallet, Settings, RefreshCw } from 'lucide-react';
import type { ReactNode } from 'react';
import { NavLink } from 'react-router-dom';
import { useSystemStatus } from '@/hooks/useSystemStatus';

interface LayoutProps {
  children: ReactNode;
}

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  [
    'flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
    isActive
      ? 'bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30'
      : 'text-gray-400 hover:bg-gray-800/60 hover:text-gray-100',
  ].join(' ');

// Format an ISO-8601 UTC timestamp as a compact local-time string.
// Pipeline runs are scheduled in America/Chicago; we render them in the
// user's local tz with the tz suffix so it's unambiguous.
function formatLastRefresh(iso: string | null | undefined): string {
  if (!iso) return 'never';
  // Backend emits naive ISO (utcnow().isoformat()); add 'Z' so JS parses as UTC.
  const isoUtc = /[zZ]|[+-]\d{2}:\d{2}$/.test(iso) ? iso : `${iso}Z`;
  const d = new Date(isoUtc);
  if (Number.isNaN(d.getTime())) return iso;
  const date = d.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
  });
  const time = d.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
  // Short timezone abbreviation (e.g. "CDT") if available.
  const tzMatch = d.toLocaleTimeString(undefined, { timeZoneName: 'short' }).match(/[A-Z]{2,5}$/);
  const tz = tzMatch ? ` ${tzMatch[0]}` : '';
  return `${date} ${time}${tz}`;
}

function formatBarDate(s: string | null | undefined): string {
  if (!s) return '—';
  // Parse YYYY-MM-DD as a local-tz date (avoid UTC shifting it back a day).
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(s);
  if (!m) return s;
  const d = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

// Visual staleness: minutes since last refresh determines the dot color.
// The pipeline fires twice a day, so >12h without a refresh is yellow,
// >36h is red.
function staleColor(iso: string | null | undefined): string {
  if (!iso) return 'bg-gray-600';
  const isoUtc = /[zZ]|[+-]\d{2}:\d{2}$/.test(iso) ? iso : `${iso}Z`;
  const d = new Date(isoUtc);
  if (Number.isNaN(d.getTime())) return 'bg-gray-600';
  const ageHours = (Date.now() - d.getTime()) / 3_600_000;
  if (ageHours > 36) return 'bg-rose-500';
  if (ageHours > 12) return 'bg-amber-400';
  return 'bg-emerald-400';
}

function RefreshIndicator() {
  const { data, isLoading, isError } = useSystemStatus();
  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-xs text-gray-500">
        <RefreshCw className="h-3.5 w-3.5 animate-spin" />
        <span>checking…</span>
      </div>
    );
  }
  if (isError || !data) {
    return (
      <div className="flex items-center gap-2 text-xs text-gray-500">
        <span className="h-2 w-2 rounded-full bg-gray-600" />
        <span>status unavailable</span>
      </div>
    );
  }
  const last = formatLastRefresh(data.last_refresh_utc);
  const bar = formatBarDate(data.latest_bar_date);
  return (
    <div
      className="flex items-center gap-2 text-xs text-gray-400"
      title={
        data.last_refresh_utc
          ? `Pipeline last completed: ${new Date(
              /[zZ]|[+-]\d{2}:\d{2}$/.test(data.last_refresh_utc)
                ? data.last_refresh_utc
                : `${data.last_refresh_utc}Z`,
            ).toString()}\nLatest OHLCV bar: ${data.latest_bar_date ?? '—'}`
          : 'Pipeline has not yet run'
      }
    >
      <span className={`h-2 w-2 rounded-full ${staleColor(data.last_refresh_utc)}`} />
      <span>
        Refreshed <span className="text-gray-200">{last}</span>
      </span>
      <span className="text-gray-600">·</span>
      <span>
        Data through <span className="text-gray-200">{bar}</span>
      </span>
    </div>
  );
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen flex flex-col bg-gray-950 text-gray-100">
      <header className="border-b border-gray-800 bg-gray-950/95 backdrop-blur sticky top-0 z-10">
        <div className="mx-auto flex max-w-[1600px] items-center justify-between px-6 py-3">
          <div className="flex items-center gap-3">
            <Activity className="h-5 w-5 text-emerald-400" />
            <NavLink to="/" className="text-base font-semibold text-gray-100">
              ta-agent
            </NavLink>
            <span className="hidden text-xs text-gray-500 sm:inline">technical-analysis ML, daily picks</span>
          </div>
          <div className="flex items-center gap-4">
            <RefreshIndicator />
            <nav className="flex items-center gap-1">
              <NavLink to="/" className={navLinkClass} end>
                <LayoutDashboard className="h-4 w-4" />
                Dashboard
              </NavLink>
              <NavLink to="/performance" className={navLinkClass}>
                <LineChart className="h-4 w-4" />
                Performance
              </NavLink>
              <NavLink to="/paper" className={navLinkClass}>
                <Wallet className="h-4 w-4" />
                Paper Trade
              </NavLink>
              <NavLink to="/settings" className={navLinkClass}>
                <Settings className="h-4 w-4" />
                Settings
              </NavLink>
            </nav>
          </div>
        </div>
      </header>
      <main className="mx-auto w-full max-w-[1600px] flex-1 px-6 py-6">{children}</main>
      <footer className="border-t border-gray-800 bg-gray-950 py-3 text-center text-xs text-gray-500">
        Predictions are research output, not investment advice.
      </footer>
    </div>
  );
}
