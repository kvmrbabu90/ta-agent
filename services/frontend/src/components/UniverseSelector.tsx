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
    <label className="flex items-center gap-2 text-sm text-gray-700">
      <span className="font-medium">Universe</span>
      <select
        className="rounded border border-gray-300 bg-white px-2 py-1 text-sm focus:border-blue-500 focus:outline-none"
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
