import { LineChart, Activity, LayoutDashboard } from 'lucide-react';
import type { ReactNode } from 'react';
import { NavLink } from 'react-router-dom';

interface LayoutProps {
  children: ReactNode;
}

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  [
    'flex items-center gap-2 rounded px-3 py-1.5 text-sm transition-colors',
    isActive
      ? 'bg-blue-600 text-white'
      : 'text-gray-700 hover:bg-gray-100 hover:text-gray-900',
  ].join(' ');

export function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-gray-200 bg-white">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3">
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-blue-600" />
            <NavLink to="/" className="text-base font-semibold text-gray-900">
              ta-agent
            </NavLink>
            <span className="text-xs text-gray-400">technical-analysis ML, daily picks</span>
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
          </nav>
        </div>
      </header>
      <main className="mx-auto w-full max-w-7xl flex-1 px-6 py-6">{children}</main>
      <footer className="border-t border-gray-200 bg-white py-3 text-center text-xs text-gray-400">
        Predictions are research output, not investment advice.
      </footer>
    </div>
  );
}
