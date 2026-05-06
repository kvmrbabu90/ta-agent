import { apiGet } from './client';
import type { OHLCVResponse } from './types';

export function fetchOhlcv(
  symbol: string,
  start?: string,
  end?: string,
): Promise<OHLCVResponse> {
  return apiGet<OHLCVResponse>(`/stocks/${encodeURIComponent(symbol)}/ohlcv`, {
    start,
    end,
  });
}
