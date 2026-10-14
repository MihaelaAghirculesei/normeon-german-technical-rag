import { httpResource } from '@angular/common/http';

import type { GetQuery, GetResponse } from './api-types';

const CONTEXT = '/api/v1/chunks/{chunk_id}/context' as const;

export type ChunkContext = GetResponse<typeof CONTEXT>;
export type ContextChunk = ChunkContext['chunk'];

/** A chunk between its neighbours, refetched whenever `chunkId` changes;
 * no request while it is undefined. Must be called in an injection
 * context. */
export function chunkContextResource(chunkId: () => string | undefined, tenantId: () => string) {
  return httpResource<ChunkContext>(() => {
    const id = chunkId();
    if (id === undefined) return undefined;
    return {
      url: CONTEXT.replace('{chunk_id}', encodeURIComponent(id)),
      params: { tenant_id: tenantId(), window: 1 } satisfies GetQuery<typeof CONTEXT>,
    };
  });
}
