import { useQuery } from '@tanstack/react-query';
import {
  fetchModelInfo,
  fetchPerformance,
  fetchStrictWf,
  fetchStrictWfAnalysis,
  fetchStrictWfMonth,
  fetchWalkforward,
} from '@/api/performance';

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

export function usePerformance(universe: string, lookbackDays = 90) {
  return useQuery({
    queryKey: ['performance', universe, lookbackDays],
    queryFn: () => fetchPerformance(universe, lookbackDays),
    enabled: Boolean(universe),
    staleTime: 60_000,
  });
}

export function useModelInfo(universe: string) {
  return useQuery({
    queryKey: ['model-info', universe],
    queryFn: () => fetchModelInfo(universe),
    enabled: Boolean(universe),
    staleTime: 5 * 60_000,
  });
}

export function useWalkforward(universe: string) {
  return useQuery({
    queryKey: ['walkforward', universe],
    queryFn: () => fetchWalkforward(universe),
    enabled: Boolean(universe),
    staleTime: 5 * 60_000,
  });
}
