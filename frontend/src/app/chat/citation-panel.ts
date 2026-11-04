import { Component, input, linkedSignal, output } from '@angular/core';

import type { Source } from '../api/chat';
import { chunkContextResource } from '../api/chunks';
import { Notice } from '../shared/notice';
import { PdfViewer } from './pdf-viewer';

/** The source behind a citation: document, section and pages, and the
 * cited chunk highlighted between the text before and after it, or on its
 * page of the original PDF when positions and the file are available. */
@Component({
  selector: 'app-citation-panel',
  imports: [Notice, PdfViewer],
  host: { '(document:keydown.escape)': 'closed.emit()' },
  templateUrl: './citation-panel.html',
  styleUrl: './citation-panel.css',
})
export class CitationPanel {
  readonly source = input.required<Source>();
  readonly closed = output<void>();

  protected readonly context = chunkContextResource(() => this.source().chunk_id);
  /** Back to the text view with every new source. */
  protected readonly showPdf = linkedSignal({ source: this.source, computation: () => false });
  protected readonly pdfUnavailable = linkedSignal({
    source: this.source,
    computation: () => false,
  });

  protected onUnavailable(): void {
    this.showPdf.set(false);
    this.pdfUnavailable.set(true);
  }

  protected pages(from: number, to: number): string {
    return from === to ? `S. ${from}` : `S. ${from}–${to}`;
  }
}
