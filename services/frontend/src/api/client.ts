// Tiny fetch wrapper. Throws ApiError on non-2xx so React Query treats
// failures correctly. No axios — fetch is sufficient.

const DEFAULT_BASE = 'http://localhost:8000';

export const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/+$/, '') ||
  DEFAULT_BASE;

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public detail?: unknown,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

function buildUrl(path: string, params?: Record<string, string | number | undefined | null>): string {
  const url = new URL(path.startsWith('http') ? path : `${API_BASE_URL}${path}`);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v === undefined || v === null || v === '') continue;
      url.searchParams.set(k, String(v));
    }
  }
  return url.toString();
}

export async function apiGet<T>(
  path: string,
  params?: Record<string, string | number | undefined | null>,
): Promise<T> {
  const res = await fetch(buildUrl(path, params), {
    method: 'GET',
    headers: { Accept: 'application/json' },
  });
  if (!res.ok) {
    let detail: unknown = undefined;
    try {
      detail = await res.json();
    } catch {
      // body wasn't JSON; ignore.
    }
    const reason =
      typeof detail === 'object' && detail !== null && 'detail' in detail
        ? String((detail as { detail: unknown }).detail)
        : res.statusText;
    throw new ApiError(`GET ${path} → ${res.status} ${reason}`, res.status, detail);
  }
  return (await res.json()) as T;
}
