import { apiGet, API_BASE_URL, ApiError } from './client';

export interface KiteLoginUrlResponse {
  url: string;
}

export interface KiteExchangeResponse {
  ok: boolean;
  user_id: string | null;
  user_name: string | null;
  exchanged_at: string;
}

export interface KiteStatusResponse {
  configured_api_key: boolean;
  has_token_env: boolean;
  has_token_file: boolean;
  session_path: string;
  user_id: string | null;
  user_name: string | null;
  exchanged_at: string | null;
  file_error: string | null;
}

export function fetchKiteLoginUrl(): Promise<KiteLoginUrlResponse> {
  return apiGet<KiteLoginUrlResponse>('/admin/kite/login-url');
}

export function fetchKiteStatus(): Promise<KiteStatusResponse> {
  return apiGet<KiteStatusResponse>('/admin/kite/status');
}

export async function exchangeKiteRequestToken(
  requestToken: string,
): Promise<KiteExchangeResponse> {
  const res = await fetch(`${API_BASE_URL}/admin/kite/exchange`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ request_token: requestToken }),
  });
  if (!res.ok) {
    let detail: unknown = undefined;
    try {
      detail = await res.json();
    } catch {
      // body wasn't JSON.
    }
    const reason =
      typeof detail === 'object' && detail !== null && 'detail' in detail
        ? String((detail as { detail: unknown }).detail)
        : res.statusText;
    throw new ApiError(`POST /admin/kite/exchange → ${res.status} ${reason}`, res.status, detail);
  }
  return (await res.json()) as KiteExchangeResponse;
}
