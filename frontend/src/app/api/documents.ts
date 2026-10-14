import { httpResource } from '@angular/common/http';

import type { GetResponse } from './api-types';

const DOCUMENTS = '/api/v1/documents' as const;

/** The signed-in tenant's documents. Must be called in an injection context
 * (e.g. a component field initializer). */
export function documentsResource() {
  return httpResource<GetResponse<typeof DOCUMENTS>>(() => ({ url: DOCUMENTS }), {
    defaultValue: [],
  });
}
