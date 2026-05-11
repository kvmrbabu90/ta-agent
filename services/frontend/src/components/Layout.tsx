import { Activity, LayoutDashboard, LineChart, Wallet, Settings } from 'lucide-react';
import type { ReactNode } from 'react';
import { NavLink } from 'react-router-dom';

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
      </header>
      <main className="mx-auto w-full max-w-[1600px] flex-1 px-6 py-6">{children}</main>
      <footer className="border-t border-gray-800 bg-gray-950 py-3 text-center text-xs text-gray-500">
        Predictions are research output, not investment advice.
      </footer>
    </div>
  );
}
