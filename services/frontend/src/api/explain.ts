import { apiGet } from './client';
import type { ExplainResponse } from './types';

export function fetchExplain(
  universe: string,
  symbol: string,
  asOf?: string,
  topK = 10,
): Promise<ExplainResponse> {
  return apiGet<ExplainResponse>(
    `/explain/${encodeURIComponent(universe)}/${encodeURIComponent(symbol)}`,
    { as_of: asOf, top_k: topK },
  );
}
