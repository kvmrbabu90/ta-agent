import { useQuery } from '@tanstack/react-query';
import { fetchUniverses } from '@/api/universe';

export function useUniverses() {
  return useQuery({
    queryKey: ['universes'],
    queryFn: fetchUniverses,
    staleTime: 5 * 60_000,
  });
}
