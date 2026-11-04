import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { type Mock, vi } from 'vitest';

import { PDF_LOADER, PdfDocument, PdfLoader, PdfPage, PdfViewer, Region } from './pdf-viewer';

const DOCUMENT_ID = 'dddddddd-0000-0000-0000-000000000001';
const URL = `/api/v1/documents/${DOCUMENT_ID}/file`;
const PAGE_WIDTH = 300;
const HOST_WIDTH = 600; // the fallback in the test DOM, which has no layout

const REGIONS: Region[] = [
  { page: 4, bbox: [10, 20, 110, 40] },
  { page: 5, bbox: [30, 40, 130, 80] },
];

const settle = async (fixture: ComponentFixture<PdfViewer>) => {
  await new Promise((resolve) => setTimeout(resolve, 0));
  await fixture.whenStable();
  fixture.detectChanges();
};

describe('PdfViewer', () => {
  let fixture: ComponentFixture<PdfViewer>;
  let http: HttpTestingController;
  let drawn: number[];
  let destroy: Mock<PdfDocument['destroy']>;
  let load: Mock<PdfLoader>;
  let unavailable: number;

  const page = (number: number): PdfPage => ({
    getViewport: ({ scale }) => ({ width: PAGE_WIDTH * scale, height: 400 * scale }),
    render: () => {
      drawn.push(number);
      return { promise: Promise.resolve(), cancel: () => {} };
    },
  });

  function open(regions: Region[] = REGIONS): void {
    fixture = TestBed.createComponent(PdfViewer);
    fixture.componentRef.setInput('documentId', DOCUMENT_ID);
    fixture.componentRef.setInput('regions', regions);
    fixture.componentInstance.unavailable.subscribe(() => unavailable++);
    fixture.detectChanges();
  }

  const marks = () => [...fixture.nativeElement.querySelectorAll('mark')] as HTMLElement[];
  const button = (text: string) =>
    [...fixture.nativeElement.querySelectorAll('button')].find(
      (b: HTMLButtonElement) => b.textContent?.trim() === text,
    ) as HTMLButtonElement;

  beforeEach(() => {
    drawn = [];
    unavailable = 0;
    destroy = vi.fn<PdfDocument['destroy']>(() => Promise.resolve());
    const document: PdfDocument = { getPage: async (n) => page(n), destroy };
    load = vi.fn<PdfLoader>(async () => document);
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: PDF_LOADER, useValue: load },
      ],
    });
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('fetches the file as bytes and draws the first cited page', async () => {
    open();

    const request = http.expectOne(URL);
    expect(request.request.responseType).toBe('arraybuffer');
    request.flush(new ArrayBuffer(4));
    await settle(fixture);

    expect(load).toHaveBeenCalledOnce();
    expect(drawn).toEqual([4]);
  });

  it('places the highlight at the region, scaled to the shown width', async () => {
    open();
    http.expectOne(URL).flush(new ArrayBuffer(4));
    await settle(fixture);

    const scale = HOST_WIDTH / PAGE_WIDTH;
    const [mark] = marks();
    expect(marks()).toHaveLength(1);
    expect(mark.style.left).toBe(`${10 * scale}px`);
    expect(mark.style.top).toBe(`${20 * scale}px`);
    expect(mark.style.width).toBe(`${100 * scale}px`);
    expect(mark.style.height).toBe(`${20 * scale}px`);
  });

  it('moves between the cited pages', async () => {
    open();
    http.expectOne(URL).flush(new ArrayBuffer(4));
    await settle(fixture);
    expect(button('Zurück').disabled).toBe(true);

    button('Weiter').click();
    await settle(fixture);

    expect(drawn).toEqual([4, 5]);
    expect(button('Weiter').disabled).toBe(true);
    expect(marks()[0].style.left).toBe(`${30 * (HOST_WIDTH / PAGE_WIDTH)}px`);
  });

  it('shows no page navigation for a citation on one page', async () => {
    open([REGIONS[0]]);
    http.expectOne(URL).flush(new ArrayBuffer(4));
    await settle(fixture);

    expect(fixture.nativeElement.querySelector('nav')).toBeNull();
  });

  it('reports the file as unavailable when it is not stored', async () => {
    open();

    http.expectOne(URL).flush(new ArrayBuffer(0), { status: 404, statusText: 'Not Found' });
    await settle(fixture);

    expect(unavailable).toBe(1);
    expect(load).not.toHaveBeenCalled();
  });

  it('reports the file as unavailable when it cannot be read', async () => {
    load.mockRejectedValueOnce(new Error('broken'));
    open();

    http.expectOne(URL).flush(new ArrayBuffer(4));
    await settle(fixture);

    expect(unavailable).toBe(1);
  });

  it('frees the document when it closes', async () => {
    open();
    http.expectOne(URL).flush(new ArrayBuffer(4));
    await settle(fixture);

    fixture.destroy();

    expect(destroy).toHaveBeenCalledOnce();
  });
});
