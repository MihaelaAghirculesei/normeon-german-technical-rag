import { InjectionToken, inject } from '@angular/core';

import { AuthService } from '../core/auth';

import type { components } from './schema';
import { readSse } from './sse';

export type ChatRequest = components['schemas']['ChatRequest'];
export type ChatStreamEvent = components['schemas']['ChatStreamEvent'];
export type Source = components['schemas']['SourceOut'];
export type Citation = components['schemas']['CitationOut'];
export type Stage = components['schemas']['StageData']['stage'];
export type Latency = components['schemas']['StreamLatencyOut'];

const CHAT_STREAM_URL = '/api/v1/chat/stream';

/** The request was refused before any event was streamed (validation, a
 * rate limit, a proxy error, the backend down). Once streaming has started,
 * failures arrive as an `error` event instead. `code` is the backend's
 * error code when the body carried one; `retryAfterS` its `Retry-After`. */
export class ChatHttpError extends Error {
  constructor(
    readonly status: number,
    readonly code: string | null = null,
    readonly retryAfterS: number | null = null,
  ) {
    super(`chat stream request failed with HTTP ${status}`);
  }
}

async function refusal(response: Response): Promise<ChatHttpError> {
  let code: string | null = null;
  try {
    const body: unknown = await response.json();
    if (body && typeof body === 'object' && 'code' in body && typeof body.code === 'string') {
      code = body.code;
    }
  } catch {
    // Not JSON (a proxy's error page): the status alone has to do.
  }
  const retryAfter = Number(response.headers.get('Retry-After'));
  return new ChatHttpError(
    response.status,
    code,
    Number.isFinite(retryAfter) && retryAfter > 0 ? retryAfter : null,
  );
}

/** POST a question and yield the typed SSE events as they arrive. Aborting
 * `signal` cancels the request; the backend sees the disconnect and stops
 * the model call. Plain fetch, so the HttpClient interceptor does not run:
 * the bearer token is passed in explicitly. */
export async function* streamChat(
  request: ChatRequest,
  signal: AbortSignal,
  token: string | null,
): AsyncGenerator<ChatStreamEvent> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'text/event-stream',
  };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  const response = await fetch(CHAT_STREAM_URL, {
    method: 'POST',
    headers,
    body: JSON.stringify(request),
    signal,
  });
  if (!response.ok || response.body === null) {
    throw await refusal(response);
  }
  for await (const frame of readSse(response.body)) {
    yield { event: frame.event, data: JSON.parse(frame.data) } as ChatStreamEvent;
  }
}

export type StreamChat = (
  request: ChatRequest,
  signal: AbortSignal,
) => AsyncGenerator<ChatStreamEvent>;

/** Seam for tests: the chat state takes its transport from here. The real
 * one signs every request with the current session's token. */
export const STREAM_CHAT = new InjectionToken<StreamChat>('STREAM_CHAT', {
  providedIn: 'root',
  factory: () => {
    const auth = inject(AuthService);
    return (request, signal) => streamChat(request, signal, auth.token());
  },
});
