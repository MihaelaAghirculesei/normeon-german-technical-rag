import { httpResource } from '@angular/common/http';

import type { GetResponse } from './api-types';

const STATS = '/api/v1/admin/stats' as const;

export type AdminStats = GetResponse<typeof STATS>;

/** The signed-in admin's tenant statistics over the last `hours`. Must be
 * called in an injection context. */
export function adminStatsResource(hours: () => number) {
  return httpResource<AdminStats>(() => ({ url: STATS, params: { hours: hours() } }));
}
