// Sign-aware color helpers for predicted/realized return styling.
// Tuned for the app's dark theme — emerald/rose match the dashboard.

export function signColor(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return 'text-gray-500';
  if (value > 0) return 'text-emerald-400';
  if (value < 0) return 'text-rose-400';
  return 'text-gray-300';
}

export function signBgColor(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return 'bg-gray-700';
  if (value > 0) return 'bg-emerald-500';
  if (value < 0) return 'bg-rose-500';
  return 'bg-gray-500';
}

// Recharts hex equivalents for use in chart fills/strokes. Tailwind
// emerald-400 / rose-400 / sky-400 / gray-400 so they read on dark bg.
export const CHART_GREEN = '#34d399'; // emerald-400
export const CHART_RED = '#fb7185'; // rose-400
export const CHART_BLUE = '#38bdf8'; // sky-400 — brighter than blue-500 on dark
export const CHART_GRAY = '#9ca3af'; // gray-400
