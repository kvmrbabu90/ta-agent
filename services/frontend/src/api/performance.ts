import { apiGet } from './client';
import type {
  ModelInfoResponse,
  PerformanceResponse,
  StrictWfMonthDetail,
  StrictWfResponse,
  WalkforwardResponse,
} from './types';

export function fetchStrictWf(universe: string): Promise<StrictWfResponse> {
  return apiGet<StrictWfResponse>(`/performance/strict-wf/${encodeURIComponent(universe)}`);
}

export function fetchStrictWfMonth(
  universe: string,
  year: number,
  month: number,
): Promise<StrictWfMonthDetail> {
  return apiGet<StrictWfMonthDetail>(
    `/performance/strict-wf/${encodeURIComponent(universe)}/month/${year}/${month}`,
  );
}

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
