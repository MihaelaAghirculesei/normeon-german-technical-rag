import { ChatHttpError, streamChat } from './chat';

async function refusalFor(response: Response): Promise<ChatHttpError> {
  vi.spyOn(globalThis, 'fetch').mockResolvedValue(response);
  try {
    for await (const _ of streamChat({ question: 'x' }, new AbortController().signal, null)) {
      // never reached: the request is refused
    }
  } catch (error) {
    return error as ChatHttpError;
  } finally {
    vi.restoreAllMocks();
  }
  throw new Error('expected the request to be refused');
}

describe('streamChat', () => {
  it('carries the code and Retry-After of a refusal', async () => {
    const error = await refusalFor(
      new Response(JSON.stringify({ code: 'rate_limited', detail: 'too many' }), {
        status: 429,
        headers: { 'Content-Type': 'application/json', 'Retry-After': '42' },
      }),
    );

    expect(error).toBeInstanceOf(ChatHttpError);
    expect(error.status).toBe(429);
    expect(error.code).toBe('rate_limited');
    expect(error.retryAfterS).toBe(42);
  });

  it('keeps the status when the body is not JSON', async () => {
    const error = await refusalFor(new Response('<html>Bad Gateway</html>', { status: 502 }));

    expect(error.status).toBe(502);
    expect(error.code).toBeNull();
    expect(error.retryAfterS).toBeNull();
  });
});
