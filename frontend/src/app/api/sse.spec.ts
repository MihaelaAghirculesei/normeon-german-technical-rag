import { readSse, type SseFrame } from './sse';

function body(...chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

async function frames(stream: ReadableStream<Uint8Array>): Promise<SseFrame[]> {
  const out: SseFrame[] = [];
  for await (const frame of readSse(stream)) out.push(frame);
  return out;
}

describe('readSse', () => {
  it('parses the frames the backend sends', async () => {
    const result = await frames(
      body(
        'event: stage\ndata: {"stage":"retrieving"}\n\n',
        'event: token\ndata: {"text":"Die"}\n\n',
      ),
    );

    expect(result).toEqual([
      { event: 'stage', data: '{"stage":"retrieving"}' },
      { event: 'token', data: '{"text":"Die"}' },
    ]);
  });

  it('reassembles a frame split at arbitrary byte boundaries', async () => {
    const wire = 'event: token\ndata: {"text":"Prüfung"}\n\n';
    const bytes = new TextEncoder().encode(wire);
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        // One byte at a time, splitting the two-byte "ü" as well.
        for (const byte of bytes) controller.enqueue(new Uint8Array([byte]));
        controller.close();
      },
    });

    expect(await frames(stream)).toEqual([{ event: 'token', data: '{"text":"Prüfung"}' }]);
  });

  it('skips heartbeat comments', async () => {
    const result = await frames(body(': heartbeat\n\n', 'event: done\ndata: {}\n\n'));

    expect(result).toEqual([{ event: 'done', data: '{}' }]);
  });

  it('accepts CRLF line endings, even split between chunks', async () => {
    const result = await frames(body('event: done\r', '\ndata: {}\r\n\r', '\n'));

    expect(result).toEqual([{ event: 'done', data: '{}' }]);
  });

  it('joins multiple data lines and defaults the event name', async () => {
    const result = await frames(body('data: a\ndata: b\n\n'));

    expect(result).toEqual([{ event: 'message', data: 'a\nb' }]);
  });

  it('drops an unterminated frame at end of stream', async () => {
    const result = await frames(body('event: done\ndata: {}\n'));

    expect(result).toEqual([]);
  });

  it('cancels the body when the consumer stops early', async () => {
    let cancelled = false;
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode('data: 1\n\ndata: 2\n\n'));
      },
      cancel() {
        cancelled = true;
      },
    });

    for await (const _ of readSse(stream)) break;

    expect(cancelled).toBe(true);
  });
});
