// Centralized formatters so we can change locale / precision in one place.

export function formatPercent(value: number | null | undefined, fractionDigits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return `${(value * 100).toFixed(fractionDigits)}%`;
}

export function formatNumber(value: number | null | undefined, fractionDigits = 4): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return value.toFixed(fractionDigits);
}

export function formatProba(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return `${(value * 100).toFixed(0)}%`;
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return '—';
  // The API hands us ISO YYYY-MM-DD. We don't need date-fns for this.
  return value;
}

export function formatInteger(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return value.toLocaleString();
}

export function formatModelVersion(version: string | null | undefined): string {
  if (!version) return '—';
  // Drop the universe + target prefix for compactness; keep the timestamp.
  const parts = version.split('_');
  return parts.slice(2).join('_') || version;
}
