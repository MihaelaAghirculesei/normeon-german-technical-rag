import { Component, inject, input, output } from '@angular/core';

import type { Source } from '../api/chat';
import { chunkContextResource } from '../api/chunks';
import { TENANT_ID } from '../core/tenant';
import { Notice } from '../shared/notice';

/** The source behind a citation: document, section and pages, and the
 * cited chunk highlighted between the text before and after it. */
@Component({
  selector: 'app-citation-panel',
  imports: [Notice],
  host: { '(document:keydown.escape)': 'closed.emit()' },
  templateUrl: './citation-panel.html',
  styleUrl: './citation-panel.css',
})
export class CitationPanel {
  private readonly tenantId = inject(TENANT_ID);

  readonly source = input.required<Source>();
  readonly closed = output<void>();

  protected readonly context = chunkContextResource(
    () => this.source().chunk_id,
    () => this.tenantId,
  );

  protected pages(from: number, to: number): string {
    return from === to ? `S. ${from}` : `S. ${from}–${to}`;
  }
}
