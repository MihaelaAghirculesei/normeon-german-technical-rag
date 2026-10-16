import type { ChatRequest, ChatStreamEvent, Source, StreamChat } from '../api/chat';

export const SOURCE_S1: Source = {
  marker: 'S1',
  chunk_id: 'aaaaaaaa-0000-0000-0000-000000000001',
  ocr: false,
  document_id: '11111111-1111-1111-1111-111111111111',
  filename: 'Lastenheft-EPS-v2.0.docx',
  page_from: 1,
  page_to: 1,
  section_path: '3 Lenkung > 3.2 Lenkkraft',
  heading: '3.2 Lenkkraft',
};

export const SOURCE_S2: Source = {
  ...SOURCE_S1,
  marker: 'S2',
  chunk_id: 'aaaaaaaa-0000-0000-0000-000000000002',
  filename: 'StVZO.pdf',
  page_from: 40,
  page_to: 41,
  section_path: '§ 38 Lenkeinrichtung',
};

/**
 * A hand-driven stand-in for `streamChat`: the test pushes events one at a
 * time and sees exactly what the UI does in between, as it would while a
 * real stream is still open. Aborting the signal ends it the way fetch
 * does, with an AbortError.
 */
export class FakeStream {
  readonly requests: ChatRequest[] = [];
  signal: AbortSignal | null = null;
  private readonly queue: ChatStreamEvent[] = [];
  private wake: (() => void) | null = null;
  private ended = false;
  private failure: unknown = null;

  readonly fn: StreamChat = (request, signal) => this.run(request, signal);

  push(...events: ChatStreamEvent[]): void {
    this.queue.push(...events);
    this.wake?.();
  }

  end(): void {
    this.ended = true;
    this.wake?.();
  }

  fail(error: unknown): void {
    this.failure = error;
    this.wake?.();
  }

  private async *run(request: ChatRequest, signal: AbortSignal): AsyncGenerator<ChatStreamEvent> {
    this.requests.push(request);
    this.signal = signal;
    while (true) {
      if (signal.aborted) throw new DOMException('aborted', 'AbortError');
      const next = this.queue.shift();
      if (next) {
        yield next;
        continue;
      }
      if (this.failure !== null) throw this.failure;
      if (this.ended) return;
      await new Promise<void>((resolve) => {
        this.wake = resolve;
        signal.addEventListener('abort', () => resolve(), { once: true });
      });
    }
  }
}

/** Let the session's `for await` loop drain what has been pushed. */
export async function settle(): Promise<void> {
  for (let i = 0; i < 20; i++) await Promise.resolve();
}
