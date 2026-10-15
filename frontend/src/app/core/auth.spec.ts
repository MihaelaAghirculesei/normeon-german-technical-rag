import { HttpClient, provideHttpClient, withInterceptors } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { Router, provideRouter, type UrlTree } from '@angular/router';

import { safeTarget } from '../login/login-page';
import { AuthService } from './auth';
import { adminGuard, authGuard, authInterceptor } from './auth-http';
import { ANNA, signedIn, signedOut } from './testing';

describe('auth', () => {
  let http: HttpTestingController;

  function setup(): void {
    TestBed.configureTestingModule({
      providers: [
        provideRouter([]),
        provideHttpClient(withInterceptors([authInterceptor])),
        provideHttpClientTesting(),
      ],
    });
    http = TestBed.inject(HttpTestingController);
  }

  afterEach(() => {
    http?.verify();
    signedOut();
  });

  describe('AuthService', () => {
    it('signs in, keeps the session across a reload, and signs out', async () => {
      signedOut();
      setup();
      const auth = TestBed.inject(AuthService);

      const done = auth.login('anna@demo.normeon.dev', 'pw');
      http.expectOne('/api/v1/auth/login').flush({
        access_token: 'abc',
        token_type: 'bearer',
        expires_at: new Date(Date.now() + 60_000).toISOString(),
        user: ANNA,
      });
      await done;

      expect(auth.token()).toBe('abc');
      expect(auth.user()?.tenant_name).toBe('Demo Tenant');
      expect(auth.isAdmin()).toBe(true);
      expect(sessionStorage.getItem('normeon.session')).toContain('abc');

      auth.logout();
      expect(auth.token()).toBeNull();
      expect(sessionStorage.getItem('normeon.session')).toBeNull();
    });

    it('treats an expired session as signed out', () => {
      signedIn(ANNA, -1000);
      setup();
      const auth = TestBed.inject(AuthService);

      expect(auth.token()).toBeNull();
      expect(auth.isAuthenticated()).toBe(false);
    });
  });

  describe('authInterceptor', () => {
    it('signs API calls, but not the login itself', () => {
      signedIn();
      setup();
      const client = TestBed.inject(HttpClient);

      client.get('/api/v1/documents').subscribe();
      client.post('/api/v1/auth/login', {}).subscribe();

      expect(http.expectOne('/api/v1/documents').request.headers.get('Authorization')).toBe(
        'Bearer test-token',
      );
      expect(http.expectOne('/api/v1/auth/login').request.headers.has('Authorization')).toBe(false);
    });

    it('ends the session on a 401 and goes to the login page', () => {
      signedIn();
      setup();
      const navigate = vi.spyOn(TestBed.inject(Router), 'navigate').mockResolvedValue(true);

      TestBed.inject(HttpClient)
        .get('/api/v1/documents')
        .subscribe({ error: () => undefined });
      http.expectOne('/api/v1/documents').flush(null, { status: 401, statusText: 'Unauthorized' });

      expect(TestBed.inject(AuthService).isAuthenticated()).toBe(false);
      expect(navigate).toHaveBeenCalledWith(['/anmelden'], expect.anything());
    });
  });

  describe('authGuard', () => {
    const run = (url: string) =>
      TestBed.runInInjectionContext(() => authGuard({} as never, { url } as never));

    it('lets a signed-in user through', () => {
      signedIn();
      setup();

      expect(run('/dokumente')).toBe(true);
    });

    it('sends everyone else to the login page, remembering the target', () => {
      signedOut();
      setup();

      const tree = run('/dokumente') as UrlTree;

      expect(TestBed.inject(Router).serializeUrl(tree)).toBe('/anmelden?weiter=%2Fdokumente');
    });
  });

  describe('adminGuard', () => {
    const run = (url: string) =>
      TestBed.runInInjectionContext(() => adminGuard({} as never, { url } as never));
    const url = (result: unknown) => TestBed.inject(Router).serializeUrl(result as UrlTree);

    it('lets an admin through', () => {
      signedIn();
      setup();

      expect(run('/statistik')).toBe(true);
    });

    it('sends a reader to the chat', () => {
      signedIn({ ...ANNA, role: 'reader' });
      setup();

      expect(url(run('/statistik'))).toBe('/fragen');
    });

    it('sends a visitor to the login page first', () => {
      signedOut();
      setup();

      expect(url(run('/statistik'))).toBe('/anmelden?weiter=%2Fstatistik');
    });
  });

  describe('safeTarget', () => {
    it.each([
      [undefined, '/fragen'],
      ['/dokumente', '/dokumente'],
      ['https://evil.example', '/fragen'],
      ['//evil.example', '/fragen'],
      ['/\\evil.example', '/fragen'],
      ['/anmelden', '/fragen'],
    ])('%s -> %s', (target, expected) => {
      expect(safeTarget(target)).toBe(expected);
    });
  });
});
