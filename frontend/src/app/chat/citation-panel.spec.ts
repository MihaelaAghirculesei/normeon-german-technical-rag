import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import type { ChunkContext } from '../api/chunks';
import { CitationPanel } from './citation-panel';
import { PDF_LOADER } from './pdf-viewer';
import { SOURCE_S1 } from './testing';

const DOCUMENT_ID = 'dddddddd-0000-0000-0000-000000000001';

const context = (regions: ChunkContext['chunk']['regions']): ChunkContext => ({
  document: {
    document_id: DOCUMENT_ID,
    filename: 'StVZO.pdf',
    doc_type: 'norm',
    version_label: null,
  },
  chunk: {
    chunk_id: SOURCE_S1.chunk_id,
    ordinal: 1,
    page_from: 40,
    page_to: 40,
    section_path: null,
    heading: null,
    content: 'Zitierter Text',
    ocr: false,
    regions,
  },
  before: [],
  after: [],
});

describe('CitationPanel', () => {
  let fixture: ComponentFixture<CitationPanel>;
  let http: HttpTestingController;

  const settle = async () => {
    await fixture.whenStable();
    fixture.detectChanges();
  };

  async function open(regions: ChunkContext['chunk']['regions']): Promise<void> {
    fixture = TestBed.createComponent(CitationPanel);
    fixture.componentRef.setInput('source', SOURCE_S1);
    fixture.detectChanges();
    http.expectOne((r) => r.url.includes('/context')).flush(context(regions));
    await settle();
  }

  const toggle = () => fixture.nativeElement.querySelector('.toggle') as HTMLButtonElement | null;
  const excerpt = () => fixture.nativeElement.querySelector('.excerpt');

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: PDF_LOADER, useValue: async () => ({}) },
      ],
    });
    http = TestBed.inject(HttpTestingController);
  });

  it('offers no PDF view for a chunk without positions', async () => {
    await open([]);

    expect(toggle()).toBeNull();
    expect(excerpt()?.textContent).toContain('Zitierter Text');
  });

  it('opens the PDF from the button, in place of the text', async () => {
    await open([{ page: 40, bbox: [1, 2, 3, 4] }]);
    expect(toggle()?.textContent?.trim()).toBe('Im PDF anzeigen');

    toggle()?.click();
    await settle();

    expect(fixture.nativeElement.querySelector('app-pdf-viewer')).not.toBeNull();
    expect(excerpt()).toBeNull();
    http.expectOne(`/api/v1/documents/${DOCUMENT_ID}/file`);
  });

  it('falls back to the text when the file is not available', async () => {
    await open([{ page: 40, bbox: [1, 2, 3, 4] }]);
    toggle()?.click();
    await settle();

    http
      .expectOne(`/api/v1/documents/${DOCUMENT_ID}/file`)
      .flush(new ArrayBuffer(0), { status: 404, statusText: 'Not Found' });
    await new Promise((resolve) => setTimeout(resolve, 0));
    await settle();

    expect(fixture.nativeElement.querySelector('app-pdf-viewer')).toBeNull();
    expect(excerpt()?.textContent).toContain('Zitierter Text');
    expect(fixture.nativeElement.textContent).toContain('Original-PDF ist nicht verfügbar');
    expect(toggle()).toBeNull();
  });
});
