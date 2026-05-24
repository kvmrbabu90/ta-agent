import { useQuery } from '@tanstack/react-query';
import { fetchPaperSnapshot, fetchPaperTrades } from '@/api/paper';

export function usePaperSnapshot(runId = 'default', lookbackDays = 60) {
  return useQuery({
    queryKey: ['paper-snapshot', runId, lookbackDays],
    queryFn: () => fetchPaperSnapshot(runId, lookbackDays),
    staleTime: 30_000,
  });
}

export function usePaperTrades(runId = 'default', limit = 50, closesOnly = false) {
  return useQuery({
    queryKey: ['paper-trades', runId, limit, closesOnly],
    queryFn: () => fetchPaperTrades(runId, limit, closesOnly),
    staleTime: 30_000,
  });
}
