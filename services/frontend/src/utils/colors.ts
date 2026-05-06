// Sign-aware color helpers for predicted/realized return styling.

export function signColor(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return 'text-gray-500';
  if (value > 0) return 'text-green-600';
  if (value < 0) return 'text-red-600';
  return 'text-gray-700';
}

export function signBgColor(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return 'bg-gray-200';
  if (value > 0) return 'bg-green-500';
  if (value < 0) return 'bg-red-500';
  return 'bg-gray-400';
}

// Recharts hex equivalents (Tailwind 500-level approximations).
export const CHART_GREEN = '#16a34a';
export const CHART_RED = '#dc2626';
export const CHART_BLUE = '#2563eb';
export const CHART_GRAY = '#9ca3af';
