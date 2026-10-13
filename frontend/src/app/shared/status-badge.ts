import { Component, computed, input } from '@angular/core';

import type { DocumentStatus } from '../api/api-types';

const LABELS: Record<DocumentStatus, string> = {
  pending: 'Wartend',
  parsing: 'Wird gelesen',
  embedding: 'Wird indexiert',
  ready: 'Bereit',
  failed: 'Fehler',
};

const TONES: Record<DocumentStatus, 'ok' | 'warn' | 'danger' | 'neutral'> = {
  pending: 'neutral',
  parsing: 'warn',
  embedding: 'warn',
  ready: 'ok',
  failed: 'danger',
};

/** A document's ingestion status as a small labelled pill. Unknown values
 * (a status added on the backend first) render as-is, in neutral. */
@Component({
  selector: 'app-status-badge',
  template: `<span class="badge" [class]="tone()">{{ label() }}</span>`,
  styles: `
    .badge {
      display: inline-block;
      padding: 0 var(--space-2);
      border-radius: var(--radius);
      font-size: 0.8rem;
      line-height: 1.6;
      white-space: nowrap;
    }
    .ok {
      color: var(--ok);
      background: var(--ok-bg);
    }
    .warn {
      color: var(--warn);
      background: var(--warn-bg);
    }
    .danger {
      color: var(--danger);
      background: var(--danger-bg);
    }
    .neutral {
      color: var(--text-muted);
      background: var(--neutral-bg);
    }
  `,
})
export class StatusBadge {
  readonly status = input.required<string>();

  protected readonly label = computed(
    () => LABELS[this.status() as DocumentStatus] ?? this.status(),
  );
  protected readonly tone = computed(() => TONES[this.status() as DocumentStatus] ?? 'neutral');
}
