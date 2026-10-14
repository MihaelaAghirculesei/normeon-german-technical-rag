import { HttpClient } from '@angular/common/http';
import { Injectable, computed, inject, signal } from '@angular/core';
import { firstValueFrom } from 'rxjs';

import type { components } from '../api/schema';

export type LoginResponse = components['schemas']['LoginResponse'];
export type CurrentUser = components['schemas']['UserOut'];

interface StoredSession {
  token: string;
  expiresAt: string;
  user: CurrentUser;
}

const STORAGE_KEY = 'normeon.session';

/**
 * The signed-in user and their access token. The tenant is whatever the
 * token says -- the client never chooses one.
 *
 * Kept in sessionStorage: it survives a reload but not closing the tab.
 * Like any storage readable from script, it is exposed to XSS; the
 * mitigation is short-lived tokens and Angular's escaping, and the
 * honest upgrade path is an httpOnly cookie set by the backend.
 */
@Injectable({ providedIn: 'root' })
export class AuthService {
  private readonly http = inject(HttpClient);
  private readonly session = signal<StoredSession | null>(restore());

  readonly user = computed(() => this.session()?.user ?? null);
  readonly isAdmin = computed(() => this.user()?.role === 'admin');

  /** The bearer token, or null when signed out or expired. */
  token(): string | null {
    const session = this.session();
    if (!session) return null;
    if (Date.parse(session.expiresAt) <= Date.now()) {
      this.logout();
      return null;
    }
    return session.token;
  }

  isAuthenticated(): boolean {
    return this.token() !== null;
  }

  async login(email: string, password: string): Promise<void> {
    const response = await firstValueFrom(
      this.http.post<LoginResponse>('/api/v1/auth/login', { email, password }),
    );
    const session: StoredSession = {
      token: response.access_token,
      expiresAt: response.expires_at,
      user: response.user,
    };
    this.session.set(session);
    store(session);
  }

  logout(): void {
    this.session.set(null);
    store(null);
  }
}

function restore(): StoredSession | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as StoredSession) : null;
  } catch {
    return null;
  }
}

function store(session: StoredSession | null): void {
  try {
    if (session) sessionStorage.setItem(STORAGE_KEY, JSON.stringify(session));
    else sessionStorage.removeItem(STORAGE_KEY);
  } catch {
    // Storage unavailable (private mode, blocked): the session simply
    // lasts until the page is reloaded.
  }
}
