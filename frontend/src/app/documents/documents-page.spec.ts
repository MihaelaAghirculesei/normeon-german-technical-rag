import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import type { DocumentSummary } from '../api/api-types';
import { DocumentsPage } from './documents-page';

const LASTENHEFT: DocumentSummary = {
  document_id: '11111111-1111-1111-1111-111111111111',
  filename: 'Lastenheft-EPS-v2.0.docx',
  doc_type: 'lastenheft',
  version_label: 'v2.0',
  valid_from: '2026-01-01',
  valid_until: null,
  page_count: null,
  status: 'ready',
  created_at: '2026-09-15T10:00:00',
  chunk_counts: { structural: 6, fixed_500: 5 },
};

describe('DocumentsPage', () => {
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [DocumentsPage],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  async function render(respond: (url: string) => void) {
    const fixture = TestBed.createComponent(DocumentsPage);
    // Not whenStable() yet: the app only becomes stable once the request
    // in flight is answered, which is what respond() does.
    TestBed.tick();
    respond('/api/v1/documents');
    await fixture.whenStable();
    return fixture.nativeElement as HTMLElement;
  }

  it('lists the tenant documents from the backend', async () => {
    const el = await render((url) => http.expectOne(url).flush([LASTENHEFT]));

    const cells = [...el.querySelectorAll('tbody td')].map((td) => td.textContent?.trim());
    expect(cells[0]).toBe('Lastenheft-EPS-v2.0.docx');
    expect(cells[2]).toBe('v2.0');
    expect(cells[3]).toContain('01.01.2026');
    expect(cells[3]).toContain('heute');
    const chunkLines = [...el.querySelectorAll('td.chunks > span')].map((s) =>
      s.textContent?.replace(/\s+/g, ' ').trim(),
    );
    expect(chunkLines).toEqual(['fixed_500 5', 'structural 6']);
    expect(cells[6]).toBe('Bereit');
  });

  it('says so when there are no documents', async () => {
    const el = await render((url) => http.expectOne(url).flush([]));

    expect(el.textContent).toContain('Noch keine Dokumente vorhanden.');
    expect(el.querySelector('table')).toBeNull();
  });

  it('shows an error when the backend is unreachable', async () => {
    const el = await render((url) =>
      http.expectOne(url).flush(null, { status: 502, statusText: 'Bad Gateway' }),
    );

    expect(el.querySelector('[role=alert]')?.textContent).toContain('nicht geladen');
  });
});
