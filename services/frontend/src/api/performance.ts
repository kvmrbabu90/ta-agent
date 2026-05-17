import { apiGet } from './client';
import type {
  ModelInfoResponse,
  PerformanceResponse,
  WalkforwardResponse,
} from './types';

export function fetchPerformance(
  universe: string,
  lookbackDays = 90,
): Promise<PerformanceResponse> {
  return apiGet<PerformanceResponse>(`/performance/${encodeURIComponent(universe)}`, {
    lookback_days: lookbackDays,
  });
}

export function fetchModelInfo(universe: string): Promise<ModelInfoResponse> {
  return apiGet<ModelInfoResponse>(`/performance/model/${encodeURIComponent(universe)}`);
}

export function fetchWalkforward(universe: string): Promise<WalkforwardResponse> {
  return apiGet<WalkforwardResponse>(`/performance/walkforward/${encodeURIComponent(universe)}`);
}
