import type { CurrentUser } from './auth';

export const ANNA: CurrentUser = {
  user_id: '77777777-0000-0000-0000-000000000001',
  email: 'anna@demo.normeon.dev',
  role: 'admin',
  tenant_id: '00000000-0000-0000-0000-000000000001',
  tenant_name: 'Demo Tenant',
};

/** Put a session in storage *before* AuthService is created (it restores
 * from there), as if the user had signed in earlier. */
export function signedIn(user: CurrentUser = ANNA, expiresInMs = 60 * 60 * 1000): void {
  sessionStorage.setItem(
    'normeon.session',
    JSON.stringify({
      token: 'test-token',
      expiresAt: new Date(Date.now() + expiresInMs).toISOString(),
      user,
    }),
  );
}

export function signedOut(): void {
  sessionStorage.clear();
}
