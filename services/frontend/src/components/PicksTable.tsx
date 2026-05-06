import { useNavigate } from 'react-router-dom';
import type { Direction, TopPick } from '@/api/types';
import { signColor } from '@/utils/colors';
import { formatPercent, formatProba } from '@/utils/format';

interface PicksTableProps {
  universe: string;
  direction: Direction;
  picks: TopPick[];
}

function ProbaBar({ value }: { value: number | null }) {
  if (value === null) return <span className="text-gray-400">—</span>;
  const pct = Math.max(0, Math.min(100, value * 100));
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-24 overflow-hidden rounded bg-gray-200">
        <div className="h-full bg-blue-600" style={{ width: `${pct}%` }} />
      </div>
      <span className="font-mono text-xs text-gray-600">{formatProba(value)}</span>
    </div>
  );
}

export function PicksTable({ universe, direction, picks }: PicksTableProps) {
  const navigate = useNavigate();
  const probaField = direction === 'long' ? 'top_quintile_proba' : 'bottom_quintile_proba';
  const probaLabel = direction === 'long' ? 'Top-quintile proba' : 'Bottom-quintile proba';

  return (
    <div className="overflow-hidden rounded border border-gray-200 bg-white">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr className="text-left text-xs uppercase tracking-wide text-gray-500">
            <th className="w-12 px-3 py-2">#</th>
            <th className="px-3 py-2">Symbol</th>
            <th className="px-3 py-2">Company</th>
            <th className="px-3 py-2 text-right">Predicted 5d</th>
            <th className="px-3 py-2 text-center">Quintile</th>
            <th className="px-3 py-2">{probaLabel}</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {picks.map((p) => (
            <tr
              key={p.symbol}
              onClick={() => navigate(`/stocks/${universe}/${p.symbol}`)}
              className="cursor-pointer text-sm hover:bg-gray-50"
            >
              <td className="px-3 py-2 font-mono text-xs text-gray-500">{p.rank}</td>
              <td className="px-3 py-2 font-mono font-medium text-blue-600">{p.symbol}</td>
              <td className="px-3 py-2 text-gray-700 truncate max-w-[20ch]">
                {p.company_name ?? '—'}
              </td>
              <td className={`px-3 py-2 text-right font-mono ${signColor(p.predicted_return_5d)}`}>
                {formatPercent(p.predicted_return_5d)}
              </td>
              <td className="px-3 py-2 text-center font-mono">
                {p.predicted_quintile ?? '—'}
              </td>
              <td className="px-3 py-2">
                <ProbaBar value={p[probaField]} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
