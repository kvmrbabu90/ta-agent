import { apiGet } from './client';
import type { Direction, StockHistoryResponse, TopPicksResponse } from './types';

export interface TopPicksArgs {
  universe: string;
  direction: Direction;
  limit?: number;
  asOf?: string;
}

export function fetchTopPicks(args: TopPicksArgs): Promise<TopPicksResponse> {
  return apiGet<TopPicksResponse>('/predictions/top', {
    universe: args.universe,
    direction: args.direction,
    limit: args.limit ?? 20,
    as_of: args.asOf,
  });
}

export function fetchStockHistory(
  universe: string,
  symbol: string,
  lookbackDays = 180,
): Promise<StockHistoryResponse> {
  return apiGet<StockHistoryResponse>(
    `/predictions/${encodeURIComponent(universe)}/${encodeURIComponent(symbol)}`,
    { lookback_days: lookbackDays },
  );
}
