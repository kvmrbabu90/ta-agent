import { useQuery } from '@tanstack/react-query';
import { fetchNextDayPicks, fetchPaperSnapshot, fetchPaperTrades } from '@/api/paper';

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

export function useNextDayPicks(runId = 'default') {
  return useQuery({
    queryKey: ['paper-next-day-picks', runId],
    queryFn: () => fetchNextDayPicks(runId),
    staleTime: 30_000,
    // Refetch on window focus — the picks update after the 17:00 CT
    // daily_predict step, and the user typically refocuses to check.
    refetchOnWindowFocus: true,
  });
}
