import { HttpClient, HttpErrorResponse, HttpEventType } from '@angular/common/http';
import { DestroyRef, Injectable, InjectionToken, inject, signal } from '@angular/core';
import { Subject, defer, repeat, takeWhile, type Subscription } from 'rxjs';

import type { components } from '../api/schema';

type UploadResponse = components['schemas']['DocumentUploadResponse'];
type StatusResponse = components['schemas']['DocumentStatusResponse'];

export const SUPPORTED_EXTENSIONS = ['.pdf', '.docx'] as const;

/** Mirrors the backend's default `max_upload_mb`, so a file that would be
 * refused is not sent first. The server stays the authority. */
export const MAX_UPLOAD_MB = 50;

/** Why the backend could not ingest a document (`error_code`). */
const INGESTION_ERRORS: Record<string, string> = {
  parse_timeout: 'Das Lesen des Dokuments hat zu lange gedauert.',
  parse_failed: 'Das Dokument konnte nicht gelesen werden.',
  embedding_failed: 'Die Indexierung ist fehlgeschlagen.',
};

export function ingestionErrorMessage(code: string | null | undefined): string {
  return (code && INGESTION_ERRORS[code]) || 'Verarbeitung fehlgeschlagen.';
}

/** Why the backend refused an upload (`code` of the error body). */
const UPLOAD_ERRORS: Record<string, string> = {
  upload_too_large: `Die Datei ist zu groß (höchstens ${MAX_UPLOAD_MB} MB).`,
  too_many_pages: 'Das Dokument hat zu viele Seiten.',
  encrypted_document: 'Die Datei ist passwortgeschützt. Bitte eine ungeschützte Kopie hochladen.',
  malformed_document: 'Die Datei ist beschädigt oder kein gültiges PDF- bzw. Word-Dokument.',
  suspicious_archive: 'Die Word-Datei würde beim Entpacken ungewöhnlich groß und wurde abgelehnt.',
  unsupported_file_type: 'Dateityp wird nicht unterstützt (nur PDF und DOCX).',
  forbidden: 'Nur Administratoren können Dokumente hochladen.',
};

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
      if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
        this.patch(item.id, { state: 'rejected', message: UPLOAD_ERRORS['upload_too_large'] });
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
              this.patch(id, { state: 'failed', message: ingestionErrorMessage(s.error_code) });
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
    const code: unknown = (error.error as { code?: unknown } | null)?.code;
    if (typeof code === 'string' && UPLOAD_ERRORS[code]) return UPLOAD_ERRORS[code];
    if (error.status === 0) return 'Keine Verbindung zum Server.';
    return `Hochladen fehlgeschlagen (HTTP ${error.status}).`;
  }
  return 'Hochladen fehlgeschlagen.';
}
