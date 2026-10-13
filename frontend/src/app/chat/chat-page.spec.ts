import { TestBed, type ComponentFixture } from '@angular/core/testing';

import { STREAM_CHAT } from '../api/chat';
import { ChatPage } from './chat-page';
import { FakeStream, SOURCE_S1, settle } from './testing';

describe('ChatPage', () => {
  let fake: FakeStream;
  let fixture: ComponentFixture<ChatPage>;
  let el: HTMLElement;

  beforeEach(async () => {
    fake = new FakeStream();
    TestBed.configureTestingModule({
      imports: [ChatPage],
      providers: [{ provide: STREAM_CHAT, useValue: fake.fn }],
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
    expect(el.querySelector('.text')?.textContent).toBe('Höchstens 250 N [S1].');
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

  it('does not send an empty question', async () => {
    await ask('   ');

    expect(fake.requests).toEqual([]);
  });
});
