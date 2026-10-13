import { InjectionToken } from '@angular/core';

/** The demo tenant seeded by backend/src/app/db/seed.py. Until auth lands
 * (Giorno 25) every request is scoped to it; afterwards this token is
 * provided from the signed-in user instead. */
export const TENANT_ID = new InjectionToken<string>('TENANT_ID', {
  providedIn: 'root',
  factory: () => '00000000-0000-0000-0000-000000000001',
});
