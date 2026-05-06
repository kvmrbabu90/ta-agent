import { useQuery } from '@tanstack/react-query';
import { fetchStockHistory } from '@/api/predictions';

export function useStockHistory(universe: string, symbol: string, lookbackDays = 180) {
  return useQuery({
    queryKey: ['stock-history', universe, symbol, lookbackDays],
    queryFn: () => fetchStockHistory(universe, symbol, lookbackDays),
    enabled: Boolean(universe && symbol),
    staleTime: 60_000,
  });
}
