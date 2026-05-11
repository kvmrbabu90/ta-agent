import type { UniverseInfo } from '@/api/types';

interface UniverseSelectorProps {
  value: string;
  onChange: (universe: string) => void;
  universes: UniverseInfo[];
  loading?: boolean;
}

export function UniverseSelector({
  value,
  onChange,
  universes,
  loading = false,
}: UniverseSelectorProps) {
  return (
    <label className="flex items-center gap-2 text-sm text-gray-300">
      <span className="text-xs uppercase tracking-wider text-gray-500">Universe</span>
      <select
        className="rounded-md border border-gray-700 bg-gray-800/60 px-2 py-1 text-sm text-gray-100 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500/40"
        value={value}
        disabled={loading || universes.length === 0}
        onChange={(e) => onChange(e.target.value)}
      >
        {universes.length === 0 ? <option value="">—</option> : null}
        {universes.map((u) => (
          <option key={u.name} value={u.name}>
            {u.name} ({u.n_members})
          </option>
        ))}
      </select>
    </label>
  );
}
