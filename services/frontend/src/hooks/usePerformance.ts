import { useQuery } from '@tanstack/react-query';
import {
  fetchModelInfo,
  fetchPerformance,
  fetchStrictWf,
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
