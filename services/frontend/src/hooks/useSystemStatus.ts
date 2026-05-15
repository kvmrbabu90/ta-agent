import { useQuery } from '@tanstack/react-query';
import { fetchSystemStatus } from '@/api/system';

// Refetch every 60s so the header indicator stays current as the
// pipeline runs in the background.
export function useSystemStatus() {
  return useQuery({
    queryKey: ['system-status'],
    queryFn: fetchSystemStatus,
    staleTime: 30_000,
    refetchInterval: 60_000,
    refetchOnWindowFocus: true,
  });
}
