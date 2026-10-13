import { InjectionToken } from '@angular/core';

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
 * proxy error, the backend down). Once streaming has started, failures
 * arrive as an `error` event instead. */
export class ChatHttpError extends Error {
  constructor(readonly status: number) {
    super(`chat stream request failed with HTTP ${status}`);
  }
}

/** POST a question and yield the typed SSE events as they arrive. Aborting
 * `signal` cancels the request; the backend sees the disconnect and stops
 * the model call. */
export async function* streamChat(
  request: ChatRequest,
  signal: AbortSignal,
): AsyncGenerator<ChatStreamEvent> {
  const response = await fetch(CHAT_STREAM_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(request),
    signal,
  });
  if (!response.ok || response.body === null) {
    throw new ChatHttpError(response.status);
  }
  for await (const frame of readSse(response.body)) {
    yield { event: frame.event, data: JSON.parse(frame.data) } as ChatStreamEvent;
  }
}

export type StreamChat = typeof streamChat;

/** Seam for tests: the chat state takes its transport from here. */
export const STREAM_CHAT = new InjectionToken<StreamChat>('STREAM_CHAT', {
  providedIn: 'root',
  factory: () => streamChat,
});
