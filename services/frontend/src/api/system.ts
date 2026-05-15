import { apiGet } from './client';
import type { SystemStatusResponse } from './types';

export function fetchSystemStatus(): Promise<SystemStatusResponse> {
  return apiGet<SystemStatusResponse>('/system/status');
}
