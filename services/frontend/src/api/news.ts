import { apiGet } from './client';
import type { NewsVerdictsResponse } from './types';

export function fetchNewsVerdicts(
  universe: string,
  asOf?: string,
): Promise<NewsVerdictsResponse> {
  return apiGet<NewsVerdictsResponse>('/news/verdicts', {
    universe,
    as_of: asOf,
  });
}
