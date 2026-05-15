import { useQuery } from '@tanstack/react-query';
import { fetchNewsVerdicts } from '@/api/news';

export function useNewsVerdicts(universe: string, asOf?: string) {
  return useQuery({
    queryKey: ['news-verdicts', universe, asOf],
    queryFn: () => fetchNewsVerdicts(universe, asOf),
    enabled: Boolean(universe),
    staleTime: 60_000,
  });
}
