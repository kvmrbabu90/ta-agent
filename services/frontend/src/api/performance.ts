import { apiGet } from './client';
import type { PerformanceResponse } from './types';

export function fetchPerformance(
  universe: string,
  lookbackDays = 90,
): Promise<PerformanceResponse> {
  return apiGet<PerformanceResponse>(`/performance/${encodeURIComponent(universe)}`, {
    lookback_days: lookbackDays,
  });
}
