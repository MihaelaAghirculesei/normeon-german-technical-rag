import { HttpErrorResponse } from '@angular/common/http';
import { Component, inject, input, signal } from '@angular/core';
import { Router } from '@angular/router';

import { AuthService } from '../core/auth';
import { Notice } from '../shared/notice';

/** The public read-only account of the demo tenant (seeded by the backend).
 * It can ask questions and open citations, but not upload documents. */
export const GUEST = { email: 'gast@demo.normeon.dev', password: 'normeon-demo' } as const;

@Component({
  selector: 'app-login-page',
  imports: [Notice],
  templateUrl: './login-page.html',
  styleUrl: './login-page.css',
})
export class LoginPage {
  private readonly auth = inject(AuthService);
  private readonly router = inject(Router);

  /** Where to go after signing in (query parameter, bound by the router). */
  readonly weiter = input<string>();

  protected readonly email = signal('');
  protected readonly password = signal('');
  protected readonly busy = signal(false);
  protected readonly error = signal<string | null>(null);

  protected submit(event: Event): Promise<void> {
    event.preventDefault();
    return this.signIn(this.email().trim(), this.password());
  }

  protected signInAsGuest(): Promise<void> {
    return this.signIn(GUEST.email, GUEST.password);
  }

  private async signIn(email: string, password: string): Promise<void> {
    if (this.busy()) return;
    this.busy.set(true);
    this.error.set(null);
    try {
      await this.auth.login(email, password);
      await this.router.navigateByUrl(safeTarget(this.weiter()));
    } catch (error) {
      this.error.set(
        error instanceof HttpErrorResponse && error.status === 401
          ? 'E-Mail oder Passwort ist falsch.'
          : 'Anmeldung nicht möglich. Läuft das Backend?',
      );
    } finally {
      this.busy.set(false);
    }
  }
}

/** Only ever continue to a path inside this app -- never to an absolute
 * or protocol-relative URL smuggled into the query string. */
export function safeTarget(target: string | undefined): string {
  if (
    !target ||
    !target.startsWith('/') ||
    target.startsWith('//') ||
    target.startsWith('/\\') ||
    target.startsWith('/anmelden')
  ) {
    return '/fragen';
  }
  return target;
}
