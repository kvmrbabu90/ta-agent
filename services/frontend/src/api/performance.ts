import { apiGet } from './client';
import type {
  StrictWfAnalysisResponse,
  StrictWfMonthDetail,
  StrictWfResponse,
} from './types';

// Strict walk-forward fetchers used by the Live WF tab. The legacy
// /performance/{universe}, /performance/model/, /performance/walkforward/
// fetchers were removed alongside the Performance tab on 2026-06-02 —
// the Live WF tab covers all backtest-performance views now.

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

export function fetchStrictWfAnalysis(
  universe: string,
): Promise<StrictWfAnalysisResponse> {
  return apiGet<StrictWfAnalysisResponse>(
    `/performance/strict-wf/${encodeURIComponent(universe)}/analysis`,
  );
}
