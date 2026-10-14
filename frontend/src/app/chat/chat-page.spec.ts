import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed, type ComponentFixture } from '@angular/core/testing';

import { STREAM_CHAT } from '../api/chat';
import { ChatPage } from './chat-page';
import { FakeStream, SOURCE_S1, SOURCE_S2, settle } from './testing';

describe('ChatPage', () => {
  let fake: FakeStream;
  let fixture: ComponentFixture<ChatPage>;
  let el: HTMLElement;

  beforeEach(async () => {
    fake = new FakeStream();
    TestBed.configureTestingModule({
      imports: [ChatPage],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: STREAM_CHAT, useValue: fake.fn },
      ],
    });
    fixture = TestBed.createComponent(ChatPage);
    el = fixture.nativeElement as HTMLElement;
    await fixture.whenStable();
  });

  async function render(): Promise<void> {
    await settle();
    fixture.detectChanges();
    await fixture.whenStable();
  }

  async function ask(question: string): Promise<void> {
    const textarea = el.querySelector('textarea')!;
    textarea.value = question;
    textarea.dispatchEvent(new Event('input'));
    await render();
    el.querySelector<HTMLButtonElement>('button[type=submit]')!.click();
    await render();
  }

  const activeStep = () => el.querySelector('[aria-current=step]')?.textContent?.trim();
  const button = (label: string) =>
    [...el.querySelectorAll('button')].find((b) => b.textContent?.trim() === label);

  it('names the current phase explicitly, in order', async () => {
    await ask('Welche Lenkkraft?');

    fake.push({ event: 'stage', data: { stage: 'retrieving' } });
    await render();
    expect(activeStep()).toContain('Dokumente durchsuchen…');

    fake.push({ event: 'stage', data: { stage: 'reranking' } });
    await render();
    expect(activeStep()).toContain('Quellen bewerten…');

    fake.push({ event: 'stage', data: { stage: 'generating' } });
    await render();
    expect(activeStep()).toContain('Antwort erstellen…');
  });

  it('shows the sources before any answer text arrives', async () => {
    await ask('Welche Lenkkraft?');

    fake.push(
      { event: 'stage', data: { stage: 'reranking' } },
      { event: 'sources', data: { sources: [SOURCE_S1] } },
    );
    await render();

    expect(el.querySelector('.sources')?.textContent).toContain('Lastenheft-EPS-v2.0.docx');
    expect(el.querySelector('.text')).toBeNull();

    fake.push({ event: 'token', data: { text: 'Höchstens 250 N [S1].' } });
    await render();
    const answer = el.querySelector('.text')!;
    expect([...answer.querySelectorAll('.segment')].map((s) => s.textContent)).toEqual([
      'Höchstens 250 N ',
      '.',
    ]);
    expect(answer.querySelector('button.cite')?.textContent?.trim()).toBe('S1');
  });

  it('Abbrechen really aborts the request', async () => {
    await ask('Welche Lenkkraft?');
    fake.push({ event: 'stage', data: { stage: 'generating' } });
    await render();

    button('Abbrechen')!.click();
    await render();

    expect(fake.signal?.aborted).toBe(true);
    expect(el.textContent).toContain('Abgebrochen.');
    expect(button('Abbrechen')).toBeUndefined();
    expect(button('Fragen')).toBeDefined();
  });

  describe('citations', () => {
    const contextUrl = (chunkId: string) => `/api/v1/chunks/${chunkId}/context?window=1`;

    async function answered(): Promise<void> {
      await ask('Welche Lenkkraft?');
      fake.push(
        { event: 'sources', data: { sources: [SOURCE_S1, SOURCE_S2] } },
        { event: 'token', data: { text: 'Höchstens 250 N [S1], vgl. [S2] und [S9].' } },
      );
      await render();
    }

    /** Click, then answer the context request the panel makes. */
    async function open(target: HTMLElement, chunkId: string, flush = true): Promise<void> {
      target.click();
      TestBed.tick();
      const request = TestBed.inject(HttpTestingController).expectOne(contextUrl(chunkId));
      if (flush) {
        request.flush({
          document: {
            document_id: SOURCE_S1.document_id,
            filename: SOURCE_S1.filename,
            doc_type: 'lastenheft',
            version_label: 'v2.0',
          },
          chunk: { ...chunk(3), content: 'Die Lenkkraft darf 250 N nicht überschreiten.' },
          before: [{ ...chunk(2), content: 'Davor.' }],
          after: [{ ...chunk(4), content: 'Danach.' }],
        });
      }
      await render();
    }

    const chunk = (ordinal: number) => ({
      chunk_id: `c-${ordinal}`,
      ordinal,
      page_from: 1,
      page_to: 1,
      section_path: SOURCE_S1.section_path,
      heading: null,
    });
    const chips = () => [...el.querySelectorAll<HTMLButtonElement>('button.cite')];

    it('turns known markers into chips and leaves unknown ones as text', async () => {
      await answered();

      expect(chips().map((c) => c.textContent?.trim())).toEqual(['S1', 'S2']);
      expect(el.querySelector('.text')?.textContent).toContain('[S9]');
    });

    it('opens the cited chunk, highlighted between its neighbours', async () => {
      await answered();

      await open(chips()[0], SOURCE_S1.chunk_id);

      const panel = el.querySelector('app-citation-panel')!;
      expect(panel.textContent).toContain('Lastenheft-EPS-v2.0.docx');
      expect(panel.textContent).toContain('3 Lenkung > 3.2 Lenkkraft');
      expect(panel.textContent).toContain('v2.0');
      expect(panel.querySelector('mark')?.textContent).toBe(
        'Die Lenkkraft darf 250 N nicht überschreiten.',
      );
      expect([...panel.querySelectorAll('.around')].map((p) => p.textContent)).toEqual([
        'Davor.',
        'Danach.',
      ]);
      expect(chips()[0].getAttribute('aria-pressed')).toBe('true');
      expect(el.querySelector('.sources')).toBeNull();
    });

    it('opens from the source list too, and Escape closes the panel', async () => {
      await answered();

      await open(el.querySelectorAll<HTMLButtonElement>('button.source')[1], SOURCE_S2.chunk_id);
      expect(el.querySelector('app-citation-panel h2')?.textContent).toContain('StVZO.pdf');

      document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
      await render();
      expect(el.querySelector('app-citation-panel')).toBeNull();
      expect(el.querySelector('.sources')).not.toBeNull();
    });

    it('says so when the source text cannot be loaded', async () => {
      await answered();
      chips()[0].click();
      TestBed.tick();
      TestBed.inject(HttpTestingController)
        .expectOne(contextUrl(SOURCE_S1.chunk_id))
        .flush(null, { status: 404, statusText: 'Not Found' });
      await render();

      expect(el.querySelector('app-citation-panel [role=alert]')?.textContent).toContain(
        'nicht geladen',
      );
    });
  });

  it('does not send an empty question', async () => {
    await ask('   ');

    expect(fake.requests).toEqual([]);
  });
});
