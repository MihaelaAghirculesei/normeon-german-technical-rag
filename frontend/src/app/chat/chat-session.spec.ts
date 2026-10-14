import { TestBed } from '@angular/core/testing';
import { Router } from '@angular/router';

import { AuthService } from '../core/auth';

import { ChatHttpError, STREAM_CHAT, type ChatStreamEvent } from '../api/chat';
import { ChatSession } from './chat-session';
import { FakeStream, SOURCE_S1, SOURCE_S2, settle } from './testing';

const DONE: Extract<ChatStreamEvent, { event: 'done' }> = {
  event: 'done',
  data: {
    latency_ms: { retrieval_ms: 900, generation_ms: 1200, total_ms: 2100 },
    cost_usd: null,
    citations: [
      {
        marker: 'S1',
        chunk_id: '22222222-2222-2222-2222-222222222222',
        document_id: SOURCE_S1.document_id,
        filename: SOURCE_S1.filename,
        page_from: 1,
        page_to: 1,
        section_path: SOURCE_S1.section_path,
        snippet: 'Lenkkraft max. 250 N',
      },
    ],
  },
};

describe('ChatSession', () => {
  let fake: FakeStream;
  let chat: ChatSession;

  beforeEach(() => {
    fake = new FakeStream();
    TestBed.configureTestingModule({
      providers: [ChatSession, { provide: STREAM_CHAT, useValue: fake.fn }],
    });
    chat = TestBed.inject(ChatSession);
  });

  it('follows the stream: stage, sources before tokens, then done', async () => {
    const asked = chat.ask('Welche Lenkkraft?');
    expect(chat.status()).toBe('streaming');
    // No tenant in the request: the backend takes it from the token.
    expect(fake.requests).toEqual([{ question: 'Welche Lenkkraft?' }]);

    fake.push({ event: 'stage', data: { stage: 'retrieving' } });
    await settle();
    expect(chat.stage()).toBe('retrieving');

    fake.push(
      { event: 'stage', data: { stage: 'reranking' } },
      { event: 'sources', data: { sources: [SOURCE_S1, SOURCE_S2] } },
    );
    await settle();
    expect(chat.sources().map((s) => s.marker)).toEqual(['S1', 'S2']);
    expect(chat.answer()).toBe('');

    fake.push(
      { event: 'stage', data: { stage: 'generating' } },
      { event: 'token', data: { text: 'Höchstens 250 N ' } },
      { event: 'token', data: { text: '[S1].' } },
    );
    await settle();
    expect(chat.answer()).toBe('Höchstens 250 N [S1].');
    expect(chat.status()).toBe('streaming');

    fake.push(DONE);
    await asked;
    expect(chat.status()).toBe('done');
    expect(chat.latency()?.total_ms).toBe(2100);
    expect([...chat.citedMarkers()]).toEqual(['S1']);
    expect(chat.abstained()).toBe(false);
  });

  it('cancel aborts the request and keeps what was already streamed', async () => {
    const asked = chat.ask('Welche Lenkkraft?');
    fake.push(
      { event: 'stage', data: { stage: 'generating' } },
      { event: 'token', data: { text: 'Höchstens' } },
    );
    await settle();

    chat.cancel();
    await asked;

    expect(fake.signal?.aborted).toBe(true);
    expect(chat.status()).toBe('cancelled');
    expect(chat.stage()).toBe('generating');
    expect(chat.answer()).toBe('Höchstens');
    expect(chat.error()).toBeNull();
  });

  it('a new question aborts the previous one, which then writes nothing', async () => {
    const first = new FakeStream();
    const second = new FakeStream();
    let calls = 0;
    TestBed.resetTestingModule();
    TestBed.configureTestingModule({
      providers: [
        ChatSession,
        {
          provide: STREAM_CHAT,
          useValue: (...args: Parameters<FakeStream['fn']>) =>
            (calls++ === 0 ? first : second).fn(...args),
        },
      ],
    });
    chat = TestBed.inject(ChatSession);

    const one = chat.ask('erste');
    await settle();
    const two = chat.ask('zweite');
    first.push({ event: 'token', data: { text: 'veraltet' } });
    second.push({ event: 'token', data: { text: 'aktuell' } });
    await settle();

    expect(first.signal?.aborted).toBe(true);
    expect(chat.question()).toBe('zweite');
    expect(chat.answer()).toBe('aktuell');

    second.push(DONE);
    await Promise.all([one, two]);
    expect(chat.status()).toBe('done');
  });

  it('recognises an abstention', async () => {
    const asked = chat.ask('Ölpreis?');
    fake.push(
      { event: 'sources', data: { sources: [] } },
      { event: 'token', data: { text: 'NICHT_GEFUNDEN' } },
      { ...DONE, data: { ...DONE.data, citations: [] } },
    );
    await asked;

    expect(chat.abstained()).toBe(true);
  });

  it('turns an error event into a German message', async () => {
    const asked = chat.ask('Welche Lenkkraft?');
    fake.push({ event: 'error', data: { code: 'llm_timeout', message: 'timeout' } });
    await asked;

    expect(chat.status()).toBe('error');
    expect(chat.error()).toBe('Das Sprachmodell hat nicht rechtzeitig geantwortet.');
  });

  it('ends the session when the stream is refused as unauthenticated', async () => {
    const auth = TestBed.inject(AuthService);
    const logout = vi.spyOn(auth, 'logout');
    vi.spyOn(TestBed.inject(Router), 'navigate').mockResolvedValue(true);

    const asked = chat.ask('x');
    fake.fail(new ChatHttpError(401));
    await asked;

    expect(chat.error()).toContain('Sitzung ist abgelaufen');
    expect(logout).toHaveBeenCalled();
  });

  it('reports a refused request and a dropped connection', async () => {
    let asked = chat.ask('x');
    fake.fail(new ChatHttpError(502));
    await asked;
    expect(chat.error()).toContain('HTTP 502');

    fake = new FakeStream();
    TestBed.resetTestingModule();
    TestBed.configureTestingModule({
      providers: [ChatSession, { provide: STREAM_CHAT, useValue: fake.fn }],
    });
    chat = TestBed.inject(ChatSession);
    asked = chat.ask('x');
    fake.push({ event: 'stage', data: { stage: 'generating' } });
    fake.end();
    await asked;
    expect(chat.status()).toBe('error');
    expect(chat.error()).toContain('Verbindung wurde unterbrochen');
  });
});
