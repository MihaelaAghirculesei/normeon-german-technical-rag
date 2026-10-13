import { httpResource } from '@angular/common/http';

import type { GetQuery, GetResponse } from './api-types';

const DOCUMENTS = '/api/v1/documents' as const;

/** The tenant's documents, refetched whenever `tenantId` changes. Must be
 * called in an injection context (e.g. a component field initializer). */
export function documentsResource(tenantId: () => string) {
  return httpResource<GetResponse<typeof DOCUMENTS>>(
    () => ({
      url: DOCUMENTS,
      params: { tenant_id: tenantId() } satisfies GetQuery<typeof DOCUMENTS>,
    }),
    { defaultValue: [] },
  );
}
