import { useQuery } from '@tanstack/react-query';
import { fetchPerformance } from '@/api/performance';

export function usePerformance(universe: string, lookbackDays = 90) {
  return useQuery({
    queryKey: ['performance', universe, lookbackDays],
    queryFn: () => fetchPerformance(universe, lookbackDays),
    enabled: Boolean(universe),
    staleTime: 60_000,
  });
}
