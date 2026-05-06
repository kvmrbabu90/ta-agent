import { useQuery } from '@tanstack/react-query';
import { fetchTopPicks, type TopPicksArgs } from '@/api/predictions';

export function useTopPicks(args: TopPicksArgs, enabled = true) {
  return useQuery({
    queryKey: ['top-picks', args.universe, args.direction, args.limit, args.asOf],
    queryFn: () => fetchTopPicks(args),
    enabled,
    staleTime: 60_000,
  });
}
