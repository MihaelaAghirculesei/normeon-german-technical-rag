import { HttpErrorResponse, type HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { Router, type CanActivateFn } from '@angular/router';
import { catchError, throwError } from 'rxjs';

import { AuthService } from './auth';

const LOGIN_URL = '/api/v1/auth/login';

/** Attach the bearer token to API calls; on a 401 the session is over,
 * so sign out and send the user to the login page. */
export const authInterceptor: HttpInterceptorFn = (request, next) => {
  const auth = inject(AuthService);
  const router = inject(Router);
  const token = auth.token();
  const isApi = request.url.startsWith('/api/');
  const authorized =
    token && isApi && request.url !== LOGIN_URL
      ? request.clone({ setHeaders: { Authorization: `Bearer ${token}` } })
      : request;

  return next(authorized).pipe(
    catchError((error: unknown) => {
      if (error instanceof HttpErrorResponse && error.status === 401 && request.url !== LOGIN_URL) {
        signOut(auth, router);
      }
      return throwError(() => error);
    }),
  );
};

/** Pages behind a login; remembers where the user was going. */
export const authGuard: CanActivateFn = (_route, state) => {
  const auth = inject(AuthService);
  return auth.isAuthenticated()
    ? true
    : inject(Router).createUrlTree(['/anmelden'], { queryParams: { weiter: state.url } });
};

/** Pages for admins only; anyone else signed in lands on the chat. */
export const adminGuard: CanActivateFn = (route, state) => {
  const auth = inject(AuthService);
  if (!auth.isAuthenticated()) return authGuard(route, state);
  return auth.isAdmin() ? true : inject(Router).createUrlTree(['/fragen']);
};

export function signOut(auth: AuthService, router: Router): void {
  auth.logout();
  void router.navigate(['/anmelden'], { queryParams: { weiter: router.url } });
}
