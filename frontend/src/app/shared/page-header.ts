import { Component, input } from '@angular/core';

/** Page title with an optional one-line description and an actions slot. */
@Component({
  selector: 'app-page-header',
  template: `
    <div>
      <h1>{{ heading() }}</h1>
      @if (description()) {
        <p class="muted">{{ description() }}</p>
      }
    </div>
    <ng-content />
  `,
  styles: `
    :host {
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: var(--space-4);
      margin-bottom: var(--space-6);
    }
    p {
      margin: var(--space-1) 0 0;
    }
  `,
})
export class PageHeader {
  readonly heading = input.required<string>();
  readonly description = input<string>();
}
