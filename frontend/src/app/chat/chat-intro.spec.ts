import { TestBed, type ComponentFixture } from '@angular/core/testing';

import type { DocumentSummary } from '../api/api-types';
import { ChatIntro } from './chat-intro';
import { examplesFor } from './examples';
import { doc } from './testing';

describe('examplesFor', () => {
  it('offers only questions about documents the tenant has', () => {
    expect(examplesFor(['FZV.pdf'])).toEqual([
      'Welcher Buchstabe kennzeichnet ein Elektrofahrzeug auf dem Kennzeichen?',
    ]);
    expect(examplesFor(['Lastenheft-EPS-v1.2.docx', 'StVZO.pdf'])).toEqual([
      'Welche Mindestprofiltiefe müssen die Reifen eines Pkw haben?',
      'Was gilt für Nebelschlussleuchten?',
    ]);
    expect(examplesFor(['Betriebsanleitung.pdf'])).toEqual([]);
  });
});

describe('ChatIntro', () => {
  let fixture: ComponentFixture<ChatIntro>;
  let el: HTMLElement;

  async function show(documents: DocumentSummary[], loaded = true): Promise<void> {
    fixture = TestBed.createComponent(ChatIntro);
    fixture.componentRef.setInput('documents', documents);
    fixture.componentRef.setInput('loaded', loaded);
    el = fixture.nativeElement as HTMLElement;
    await fixture.whenStable();
  }

  it('says what it answers, lists the ready documents and offers matching examples', async () => {
    await show([
      doc('StVZO.pdf'),
      doc('Lastenheft-EPS-v2.0.docx', { version_label: 'v2.0' }),
      doc('Entwurf.pdf', { status: 'parsing' }),
    ]);

    expect(el.textContent).toContain('Allgemeine Fragen beantwortet es nicht.');
    expect([...el.querySelectorAll('.documents li')].map((li) => li.textContent?.trim())).toEqual([
      'StVZO.pdf',
      'Lastenheft-EPS-v2.0.docx v2.0',
    ]);
    expect([...el.querySelectorAll('button.example')].map((b) => b.textContent?.trim())).toEqual([
      'Welche Mindestprofiltiefe müssen die Reifen eines Pkw haben?',
      'Was gilt für Nebelschlussleuchten?',
    ]);
  });

  it('emits the example that was clicked', async () => {
    await show([doc('FZV.pdf')]);
    const asked: string[] = [];
    fixture.componentInstance.ask.subscribe((q) => asked.push(q));

    el.querySelector<HTMLButtonElement>('button.example')!.click();

    expect(asked).toEqual([
      'Welcher Buchstabe kennzeichnet ein Elektrofahrzeug auf dem Kennzeichen?',
    ]);
  });

  it('tells a tenant without documents why nothing can be answered yet', async () => {
    await show([]);

    expect(el.querySelector('app-notice')?.textContent).toContain(
      'Für Ihren Mandanten sind noch keine Dokumente hinterlegt.',
    );
    expect(el.querySelector('button.example')).toBeNull();
  });

  it('shows only the explanation until the documents have loaded', async () => {
    await show([], false);

    expect(el.textContent).toContain('Normeon antwortet nur aus den Dokumenten');
    expect(el.querySelector('app-notice')).toBeNull();
    expect(el.querySelector('.documents')).toBeNull();
  });
});
