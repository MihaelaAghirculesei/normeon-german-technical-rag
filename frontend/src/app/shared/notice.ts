import { Component, input } from '@angular/core';

/** A plain bordered message box for empty, loading and error states. */
@Component({
  selector: 'app-notice',
  template: `<div
    class="notice"
    [class.error]="kind() === 'error'"
    [attr.role]="kind() === 'error' ? 'alert' : 'status'"
  >
    <ng-content />
  </div>`,
  styles: `
    .notice {
      padding: var(--space-4);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: var(--surface);
      color: var(--text-muted);
    }
    .error {
      color: var(--danger);
      background: var(--danger-bg);
      border-color: transparent;
    }
  `,
})
export class Notice {
  readonly kind = input<'info' | 'error'>('info');
}
