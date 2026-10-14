import { Injectable, computed, inject, signal } from '@angular/core';
import { Router } from '@angular/router';

import {
  ChatHttpError,
  STREAM_CHAT,
  type Citation,
  type Latency,
  type Source,
  type Stage,
} from '../api/chat';
import { signOut } from '../core/auth-http';
import { AuthService } from '../core/auth';
import { TENANT_ID } from '../core/tenant';

export type ChatStatus = 'idle' | 'streaming' | 'done' | 'cancelled' | 'error';

/** What the backend answers when the sources don't support an answer. */
export const ABSTENTION = 'NICHT_GEFUNDEN';

const ERROR_MESSAGES: Record<string, string> = {
  llm_timeout: 'Das Sprachmodell hat nicht rechtzeitig geantwortet.',
  llm_unavailable: 'Das Sprachmodell ist gerade nicht erreichbar.',
  llm_misconfigured: 'Das Sprachmodell ist nicht konfiguriert.',
  embedding_misconfigured: 'Die Suche ist nicht konfiguriert.',
};

/**
 * The state of one question/answer exchange, as signals, fed by the SSE
 * stream. Provided per chat page, not app-wide: leaving the page drops it.
 *
 * Only the most recent request may write state: asking again or
 * cancelling aborts the previous request, and every write checks that its
 * own request is still the current one.
 */
@Injectable()
export class ChatSession {
  private readonly streamChat = inject(STREAM_CHAT);
  private readonly tenantId = inject(TENANT_ID);
  private readonly auth = inject(AuthService);
  private readonly router = inject(Router);
  private controller: AbortController | null = null;

  readonly status = signal<ChatStatus>('idle');
  readonly question = signal('');
  /** The last stage the backend reported. Kept once the request ends, so
   * the UI can show where a cancelled or failed request stopped. */
  readonly stage = signal<Stage | null>(null);
  readonly sources = signal<Source[]>([]);
  readonly answer = signal('');
  readonly citations = signal<Citation[]>([]);
  readonly latency = signal<Latency | null>(null);
  readonly costUsd = signal<number | null>(null);
  readonly error = signal<string | null>(null);

  readonly streaming = computed(() => this.status() === 'streaming');
  readonly abstained = computed(
    () => this.status() === 'done' && this.answer().trim() === ABSTENTION,
  );
  readonly citedMarkers = computed(() => new Set(this.citations().map((c) => c.marker)));

  async ask(question: string): Promise<void> {
    this.controller?.abort();
    const controller = new AbortController();
    this.controller = controller;
    const current = () => this.controller === controller && !controller.signal.aborted;

    this.question.set(question);
    this.stage.set(null);
    this.sources.set([]);
    this.answer.set('');
    this.citations.set([]);
    this.latency.set(null);
    this.costUsd.set(null);
    this.error.set(null);
    this.status.set('streaming');

    try {
      const events = this.streamChat({ question, tenant_id: this.tenantId }, controller.signal);
      for await (const event of events) {
        if (!current()) return;
        switch (event.event) {
          case 'stage':
            this.stage.set(event.data.stage);
            break;
          case 'sources':
            this.sources.set(event.data.sources);
            break;
          case 'token':
            this.answer.update((text) => text + event.data.text);
            break;
          case 'done':
            this.citations.set(event.data.citations);
            this.latency.set(event.data.latency_ms);
            this.costUsd.set(event.data.cost_usd);
            this.finish('done');
            return;
          case 'error':
            this.fail(
              ERROR_MESSAGES[event.data.code] ?? 'Bei der Antwort ist ein Fehler aufgetreten.',
            );
            return;
        }
      }
      if (current()) {
        this.fail('Die Verbindung wurde unterbrochen, bevor die Antwort vollständig war.');
      }
    } catch (error) {
      if (!current()) return;
      if (error instanceof ChatHttpError && error.status === 401) {
        // The token expired or was revoked: same as any other 401.
        this.fail('Die Sitzung ist abgelaufen. Bitte erneut anmelden.');
        signOut(this.auth, this.router);
        return;
      }
      this.fail(
        error instanceof ChatHttpError
          ? error.status === 422
            ? 'Die Frage wurde vom Server abgelehnt.'
            : `Der Server hat die Anfrage abgelehnt (HTTP ${error.status}).`
          : 'Keine Verbindung zum Server.',
      );
    } finally {
      if (this.controller === controller) this.controller = null;
    }
  }

  /** Abort the running request. The backend notices the disconnect and
   * stops the model call; what was already streamed stays visible. */
  cancel(): void {
    if (!this.controller) return;
    this.controller.abort();
    this.controller = null;
    this.finish('cancelled');
  }

  private finish(status: 'done' | 'cancelled'): void {
    this.status.set(status);
  }

  private fail(message: string): void {
    this.error.set(message);
    this.status.set('error');
  }
}
