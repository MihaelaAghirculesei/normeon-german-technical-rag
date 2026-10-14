import { HttpClient, HttpErrorResponse, HttpEventType } from '@angular/common/http';
import { DestroyRef, Injectable, InjectionToken, inject, signal } from '@angular/core';
import { Subject, defer, repeat, takeWhile, type Subscription } from 'rxjs';

import type { components } from '../api/schema';
import { TENANT_ID } from '../core/tenant';

type UploadResponse = components['schemas']['DocumentUploadResponse'];
type StatusResponse = components['schemas']['DocumentStatusResponse'];

export const SUPPORTED_EXTENSIONS = ['.pdf', '.docx'] as const;

/** Milliseconds between two status reads while a document is processed. */
export const POLL_INTERVAL_MS = new InjectionToken<number>('POLL_INTERVAL_MS', {
  providedIn: 'root',
  factory: () => 1500,
});

export const DUPLICATE_MESSAGE =
  'Dokument bereits vorhanden (identischer Hash) — keine Neuverarbeitung.';

/** Where one file is: being sent, being processed (with the backend's own
 * status), or finished one way or another. */
export type UploadState =
  'uploading' | 'processing' | 'ready' | 'duplicate' | 'failed' | 'rejected';

export interface UploadItem {
  id: number;
  name: string;
  size: number;
  state: UploadState;
  /** Share of bytes sent, 0..1, while uploading. */
  progress: number;
  /** The document's ingestion status from the backend, once known. */
  status: string | null;
  message: string | null;
}

export interface UploadOptions {
  docType: string;
  versionLabel: string;
}

const TERMINAL = new Set(['ready', 'failed']);

/**
 * Uploads files one request each, reports byte progress, then polls the
 * document's status until it is ready or failed. Provided per documents
 * page; leaving the page stops every request and poll it started.
 */
@Injectable()
export class UploadQueue {
  private readonly http = inject(HttpClient);
  private readonly tenantId = inject(TENANT_ID);
  private readonly pollMs = inject(POLL_INTERVAL_MS);
  private readonly subscriptions = new Set<Subscription>();
  private nextId = 1;

  readonly items = signal<UploadItem[]>([]);
  /** Fires when the document list may have changed: a document was
   * created, or one finished processing. */
  readonly changed = new Subject<void>();

  constructor() {
    inject(DestroyRef).onDestroy(() => this.subscriptions.forEach((s) => s.unsubscribe()));
  }

  add(files: Iterable<File>, options: UploadOptions): void {
    for (const file of files) {
      const item: UploadItem = {
        id: this.nextId++,
        name: file.name,
        size: file.size,
        state: 'uploading',
        progress: 0,
        status: null,
        message: null,
      };
      this.items.update((items) => [item, ...items]);

      if (!SUPPORTED_EXTENSIONS.some((ext) => file.name.toLowerCase().endsWith(ext))) {
        this.patch(item.id, { state: 'rejected', message: 'Nur PDF- und DOCX-Dateien.' });
        continue;
      }
      this.upload(item.id, file, options);
    }
  }

  /** Drop finished entries from the list; running ones stay. */
  clearFinished(): void {
    this.items.update((items) =>
      items.filter((i) => i.state === 'uploading' || i.state === 'processing'),
    );
  }

  private upload(id: number, file: File, options: UploadOptions): void {
    const form = new FormData();
    form.append('tenant_id', this.tenantId);
    form.append('doc_type', options.docType);
    if (options.versionLabel.trim()) form.append('version_label', options.versionLabel.trim());
    form.append('file', file, file.name);

    this.track(
      this.http
        .post<UploadResponse>('/api/v1/documents', form, {
          reportProgress: true,
          observe: 'events',
        })
        .subscribe({
          next: (event) => {
            if (event.type === HttpEventType.UploadProgress && event.total) {
              this.patch(id, { progress: event.loaded / event.total });
            } else if (event.type === HttpEventType.Response && event.body) {
              this.received(id, event.body);
            }
          },
          error: (error: unknown) => this.patch(id, { state: 'failed', message: describe(error) }),
        }),
    );
  }

  private received(id: number, response: UploadResponse): void {
    this.patch(id, { progress: 1, status: response.status });
    switch (response.outcome) {
      case 'duplicate':
        this.patch(id, { state: 'duplicate', message: DUPLICATE_MESSAGE });
        return;
      case 'in_progress':
        this.patch(id, {
          message: 'Identische Datei wird bereits verarbeitet — keine zweite Verarbeitung.',
        });
        break;
      case 'retrying':
        this.patch(id, { message: 'Frühere Verarbeitung war fehlgeschlagen — neuer Versuch.' });
        break;
      case 'created':
        this.changed.next();
        break;
    }
    this.patch(id, { state: 'processing' });
    this.poll(id, response.document_id);
  }

  private poll(id: number, documentId: string): void {
    this.track(
      defer(() => this.http.get<StatusResponse>(`/api/v1/documents/${documentId}`))
        .pipe(
          repeat({ delay: this.pollMs }),
          takeWhile((s) => !TERMINAL.has(s.status), true),
        )
        .subscribe({
          next: (s) => {
            this.patch(id, { status: s.status });
            if (s.status === 'ready') this.patch(id, { state: 'ready', message: null });
            if (s.status === 'failed') {
              this.patch(id, { state: 'failed', message: 'Verarbeitung fehlgeschlagen.' });
            }
          },
          complete: () => this.changed.next(),
          error: (error: unknown) => this.patch(id, { state: 'failed', message: describe(error) }),
        }),
    );
  }

  private track(subscription: Subscription): void {
    this.subscriptions.add(subscription);
    subscription.add(() => this.subscriptions.delete(subscription));
  }

  private patch(id: number, change: Partial<UploadItem>): void {
    this.items.update((items) => items.map((i) => (i.id === id ? { ...i, ...change } : i)));
  }
}

function describe(error: unknown): string {
  if (error instanceof HttpErrorResponse) {
    if (error.status === 415) return 'Dateityp wird nicht unterstützt (nur PDF und DOCX).';
    if (error.status === 0) return 'Keine Verbindung zum Server.';
    return `Hochladen fehlgeschlagen (HTTP ${error.status}).`;
  }
  return 'Hochladen fehlgeschlagen.';
}
