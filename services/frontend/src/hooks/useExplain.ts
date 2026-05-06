import { useQuery } from '@tanstack/react-query';
import { fetchExplain } from '@/api/explain';

export function useExplain(
  universe: string,
  symbol: string,
  asOf?: string,
  topK = 10,
) {
  return useQuery({
    queryKey: ['explain', universe, symbol, asOf, topK],
    queryFn: () => fetchExplain(universe, symbol, asOf, topK),
    enabled: Boolean(universe && symbol),
    // Explain is server-side expensive; cache aggressively in the client.
    staleTime: 5 * 60_000,
    retry: 1,
  });
}
