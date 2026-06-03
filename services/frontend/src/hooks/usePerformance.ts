import { useQuery } from '@tanstack/react-query';
import {
  fetchStrictWf,
  fetchStrictWfAnalysis,
  fetchStrictWfMonth,
} from '@/api/performance';

// Strict walk-forward hooks used by the Live WF tab. The legacy
// usePerformance / useModelInfo / useWalkforward hooks were removed
// alongside the Performance tab on 2026-06-02 — the Live WF tab
// covers all backtest-performance views now.

export function useStrictWf(universe: string, refetchMs = 60_000) {
  return useQuery({
    queryKey: ['strict-wf', universe],
    queryFn: () => fetchStrictWf(universe),
    enabled: Boolean(universe),
    refetchInterval: refetchMs,
    staleTime: 30_000,
  });
}

export function useStrictWfMonth(
  universe: string,
  year: number | null,
  month: number | null,
) {
  return useQuery({
    queryKey: ['strict-wf-month', universe, year, month],
    queryFn: () => fetchStrictWfMonth(universe, year!, month!),
    // Only fire when both year + month are provided.
    enabled: Boolean(universe && year && month),
    staleTime: 60_000,
  });
}

export function useStrictWfAnalysis(universe: string, refetchMs = 60_000) {
  return useQuery({
    queryKey: ['strict-wf-analysis', universe],
    queryFn: () => fetchStrictWfAnalysis(universe),
    enabled: Boolean(universe),
    refetchInterval: refetchMs,
    staleTime: 30_000,
  });
}
