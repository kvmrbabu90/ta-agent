import { apiGet } from './client';
import type { MemberInfo, UniverseInfo } from './types';

export function fetchUniverses(): Promise<UniverseInfo[]> {
  return apiGet<UniverseInfo[]>('/universes');
}

export function fetchMembers(universe: string, asOf?: string): Promise<MemberInfo[]> {
  return apiGet<MemberInfo[]>(`/universes/${encodeURIComponent(universe)}/members`, {
    as_of: asOf,
  });
}
