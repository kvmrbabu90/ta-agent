import { apiGet } from './client';
import type { PaperSnapshotResponse, PaperTradesResponse } from './types';

export function fetchPaperSnapshot(
  runId = 'default',
  lookbackDays = 60,
): Promise<PaperSnapshotResponse> {
  return apiGet<PaperSnapshotResponse>('/paper/snapshot', {
    run_id: runId,
    lookback_days: lookbackDays,
  });
}

export function fetchPaperTrades(
  runId = 'default',
  limit = 50,
  closesOnly = false,
): Promise<PaperTradesResponse> {
  return apiGet<PaperTradesResponse>('/paper/trades', {
    run_id: runId,
    limit,
    closes_only: closesOnly,
  });
}
