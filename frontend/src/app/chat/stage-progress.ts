import { Component, computed, input } from '@angular/core';

import type { Stage } from '../api/chat';
import type { ChatStatus } from './chat-session';

const STEPS: { stage: Stage; label: string }[] = [
  { stage: 'retrieving', label: 'Dokumente durchsuchen' },
  { stage: 'reranking', label: 'Quellen bewerten' },
  { stage: 'generating', label: 'Antwort erstellen' },
];

type StepState = 'pending' | 'active' | 'complete' | 'stopped';

/** The pipeline phase, spelled out: which step is running, which are
 * done, and where a cancelled or failed request stopped. Never a generic
 * spinner -- the user sees what the system is actually doing. */
@Component({
  selector: 'app-stage-progress',
  template: `
    <ol aria-label="Fortschritt">
      @for (step of steps(); track step.stage) {
        <li [class]="step.state" [attr.aria-current]="step.state === 'active' ? 'step' : null">
          <span class="mark" aria-hidden="true">{{ marks[step.state] }}</span>
          {{ step.label }}{{ step.state === 'active' ? '…' : '' }}
          @if (step.state === 'stopped') {
            <span class="sr-only">(abgebrochen)</span>
          }
        </li>
      }
    </ol>
  `,
  styles: `
    ol {
      display: flex;
      flex-wrap: wrap;
      gap: var(--space-2) var(--space-6);
      margin: 0;
      padding: 0;
      list-style: none;
      font-size: 0.9rem;
    }
    li {
      display: flex;
      align-items: baseline;
      gap: var(--space-2);
      color: var(--text-muted);
    }
    .mark {
      display: inline-block;
      width: 1.2em;
      text-align: center;
      font-family: var(--font-mono);
    }
    .active {
      color: var(--text);
      font-weight: 600;
    }
    .active .mark {
      color: var(--accent);
    }
    .complete .mark {
      color: var(--ok);
    }
    .stopped {
      color: var(--danger);
    }
  `,
})
export class StageProgress {
  readonly stage = input<Stage | null>(null);
  readonly status = input.required<ChatStatus>();

  protected readonly marks: Record<StepState, string> = {
    pending: '○',
    active: '▸',
    complete: '✓',
    stopped: '×',
  };

  protected readonly steps = computed(() => {
    const status = this.status();
    const reached = STEPS.findIndex((s) => s.stage === this.stage());
    return STEPS.map((step, i): { stage: Stage; label: string; state: StepState } => {
      let state: StepState;
      if (status === 'done') state = 'complete';
      else if (i < reached) state = 'complete';
      else if (i > reached) state = 'pending';
      else state = status === 'streaming' ? 'active' : 'stopped';
      return { ...step, state };
    });
  });
}
