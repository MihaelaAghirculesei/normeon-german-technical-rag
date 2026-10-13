import { DatePipe } from '@angular/common';
import { Component, inject } from '@angular/core';

import { documentsResource } from '../api/documents';
import { TENANT_ID } from '../core/tenant';
import { Notice } from '../shared/notice';
import { PageHeader } from '../shared/page-header';
import { StatusBadge } from '../shared/status-badge';

@Component({
  selector: 'app-documents-page',
  imports: [DatePipe, Notice, PageHeader, StatusBadge],
  templateUrl: './documents-page.html',
  styleUrl: './documents-page.css',
})
export class DocumentsPage {
  private readonly tenantId = inject(TENANT_ID);

  protected readonly documents = documentsResource(() => this.tenantId);
}
