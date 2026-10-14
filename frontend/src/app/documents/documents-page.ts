import { DatePipe } from '@angular/common';
import { Component, inject } from '@angular/core';

import type { DocumentSummary } from '../api/api-types';
import { documentsResource } from '../api/documents';
import { AuthService } from '../core/auth';
import { Notice } from '../shared/notice';
import { PageHeader } from '../shared/page-header';
import { StatusBadge } from '../shared/status-badge';
import { UploadPanel } from './upload-panel';

@Component({
  selector: 'app-documents-page',
  imports: [DatePipe, Notice, PageHeader, StatusBadge, UploadPanel],
  templateUrl: './documents-page.html',
  styleUrl: './documents-page.css',
})
export class DocumentsPage {
  protected readonly auth = inject(AuthService);
  protected readonly documents = documentsResource();

  /** Chunk counts per strategy, in a stable order for the table. */
  protected chunkCounts(doc: DocumentSummary): { strategy: string; count: number }[] {
    return Object.entries(doc.chunk_counts)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([strategy, count]) => ({ strategy, count }));
  }
}
