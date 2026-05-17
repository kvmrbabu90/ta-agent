import { useQuery } from '@tanstack/react-query';
import {
  fetchModelInfo,
  fetchPerformance,
  fetchWalkforward,
} from '@/api/performance';

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
