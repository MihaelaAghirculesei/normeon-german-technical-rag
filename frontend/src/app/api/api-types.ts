// Helpers that read request/response types straight out of the generated
// `schema.d.ts` (npm run gen:api). Everything the app sends or receives is
// typed through these, so a backend contract change breaks the build here
// instead of drifting silently at runtime.
import type { components, paths } from './schema';

type Get<P extends keyof paths> = paths[P]['get'];

export type GetResponse<P extends keyof paths> =
  Get<P> extends { responses: { 200: { content: { 'application/json': infer R } } } } ? R : never;

export type GetQuery<P extends keyof paths> =
  Get<P> extends { parameters: { query?: infer Q } } ? NonNullable<Q> : never;

export type DocumentSummary = components['schemas']['DocumentSummary'];
export type DocumentStatus = 'pending' | 'parsing' | 'embedding' | 'ready' | 'failed';
