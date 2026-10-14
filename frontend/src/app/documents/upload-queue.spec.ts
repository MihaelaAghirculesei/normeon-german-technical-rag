import { HttpEventType, provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { DUPLICATE_MESSAGE, POLL_INTERVAL_MS, UploadQueue } from './upload-queue';

const DOC_ID = '11111111-1111-1111-1111-111111111111';
const STATUS_URL = `/api/v1/documents/${DOC_ID}`;
const OPTIONS = { docType: 'lastenheft', versionLabel: ' v2.0 ' };

function file(name: string, bytes = 2048): File {
  return new File([new Uint8Array(bytes)], name);
}

/** Let a `repeat({ delay: 0 })` timer fire. */
const nextTick = () => new Promise((resolve) => setTimeout(resolve, 0));

describe('UploadQueue', () => {
  let queue: UploadQueue;
  let http: HttpTestingController;
  let changed: number;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        UploadQueue,
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: POLL_INTERVAL_MS, useValue: 0 },
      ],
    });
    queue = TestBed.inject(UploadQueue);
    http = TestBed.inject(HttpTestingController);
    changed = 0;
    queue.changed.subscribe(() => changed++);
  });

  afterEach(() => http.verify());

  const item = () => queue.items()[0];
  const respond = (outcome: string, status = 'pending', code = 201) =>
    http
      .expectOne({ method: 'POST', url: '/api/v1/documents' })
      .flush(
        { document_id: DOC_ID, filename: 'Lastenheft.docx', status, outcome },
        { status: code, statusText: 'OK' },
      );

  it('sends the file with tenant, type and trimmed version', () => {
    queue.add([file('Lastenheft.docx')], OPTIONS);

    const request = http.expectOne('/api/v1/documents');
    const form = request.request.body as FormData;
    expect(form.get('tenant_id')).toBe('00000000-0000-0000-0000-000000000001');
    expect(form.get('doc_type')).toBe('lastenheft');
    expect(form.get('version_label')).toBe('v2.0');
    expect((form.get('file') as File).name).toBe('Lastenheft.docx');
    request.flush({ document_id: DOC_ID, filename: 'x', status: 'ready', outcome: 'duplicate' });
  });

  it('leaves the version out when it is blank', () => {
    queue.add([file('FZV.pdf')], { docType: 'norm', versionLabel: '  ' });

    const request = http.expectOne('/api/v1/documents');
    expect((request.request.body as FormData).has('version_label')).toBe(false);
    request.flush({ document_id: DOC_ID, filename: 'x', status: 'ready', outcome: 'duplicate' });
  });

  it('reports byte progress, then follows ingestion until ready', async () => {
    queue.add([file('Lastenheft.docx')], OPTIONS);
    const upload = http.expectOne('/api/v1/documents');
    upload.event({ type: HttpEventType.UploadProgress, loaded: 512, total: 2048 });
    expect(item().state).toBe('uploading');
    expect(item().progress).toBe(0.25);

    upload.flush({ document_id: DOC_ID, filename: 'x', status: 'pending', outcome: 'created' });
    expect(item().state).toBe('processing');
    expect(changed).toBe(1);

    http.expectOne(STATUS_URL).flush({ document_id: DOC_ID, filename: 'x', status: 'parsing' });
    expect(item().status).toBe('parsing');
    await nextTick();
    http.expectOne(STATUS_URL).flush({ document_id: DOC_ID, filename: 'x', status: 'embedding' });
    expect(item().status).toBe('embedding');
    await nextTick();
    http.expectOne(STATUS_URL).flush({ document_id: DOC_ID, filename: 'x', status: 'ready' });

    expect(item().state).toBe('ready');
    expect(changed).toBe(2);
    await nextTick();
    http.expectNone(STATUS_URL); // polling stops at a terminal status
  });

  it('says plainly that an identical file is already there, and does no work', async () => {
    queue.add([file('FZV.pdf')], OPTIONS);
    respond('duplicate', 'ready', 200);

    expect(item().state).toBe('duplicate');
    expect(item().message).toBe(DUPLICATE_MESSAGE);
    expect(changed).toBe(0);
    await nextTick();
    http.expectNone(STATUS_URL);
  });

  it('follows a file the backend is already processing without processing it twice', () => {
    queue.add([file('FZV.pdf')], OPTIONS);
    respond('in_progress', 'embedding', 200);

    expect(item().state).toBe('processing');
    expect(item().message).toContain('bereits verarbeitet');
    http.expectOne(STATUS_URL).flush({ document_id: DOC_ID, filename: 'x', status: 'failed' });
    expect(item().state).toBe('failed');
    expect(item().message).toBe('Verarbeitung fehlgeschlagen.');
  });

  it('rejects an unsupported file before sending anything', () => {
    queue.add([file('notizen.txt')], OPTIONS);

    expect(item().state).toBe('rejected');
    http.expectNone('/api/v1/documents');
  });

  it('explains a refused upload', () => {
    queue.add([file('FZV.pdf')], OPTIONS);
    http
      .expectOne('/api/v1/documents')
      .flush({ code: 'unsupported_file_type' }, { status: 415, statusText: 'Unsupported' });

    expect(item().state).toBe('failed');
    expect(item().message).toContain('nur PDF und DOCX');
  });

  it('lists the newest file first and clears only finished ones', () => {
    queue.add([file('a.txt'), file('b.pdf')], OPTIONS);
    expect(queue.items().map((i) => i.name)).toEqual(['b.pdf', 'a.txt']);

    queue.clearFinished();

    expect(queue.items().map((i) => i.name)).toEqual(['b.pdf']);
    respond('duplicate', 'ready', 200);
  });
});
