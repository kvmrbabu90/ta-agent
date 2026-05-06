import { useQuery } from '@tanstack/react-query';
import { fetchOhlcv } from '@/api/stocks';

export function useStockOhlcv(symbol: string, start?: string, end?: string) {
  return useQuery({
    queryKey: ['ohlcv', symbol, start, end],
    queryFn: () => fetchOhlcv(symbol, start, end),
    enabled: Boolean(symbol),
    staleTime: 5 * 60_000,
  });
}
