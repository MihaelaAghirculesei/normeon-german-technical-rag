/** One frame of a `text/event-stream`: the `event:` name (defaults to
 * "message") and the `data:` lines joined with "\n". */
export interface SseFrame {
  event: string;
  data: string;
}

/**
 * Parse a `text/event-stream` body into frames as bytes arrive. EventSource
 * would do this for free, but it can't send a POST or custom headers, so
 * the chat stream is read with fetch and parsed here instead.
 *
 * Follows the WHATWG rules the backend relies on: frames end at a blank
 * line, `:` lines are comments (the heartbeat), CRLF/CR/LF all end a line,
 * one optional space after the colon is dropped, a frame with no `data:`
 * line is not dispatched, and an unterminated frame at end of stream is
 * discarded. Stopping iteration early cancels the body.
 */
export async function* readSse(body: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = body.getReader();
  // stream: true keeps a multi-byte character split across chunks intact.
  const decoder = new TextDecoder();
  let finished = false;
  let buffer = '';
  let event = '';
  let data: string[] = [];

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        finished = true;
        return;
      }
      buffer += decoder.decode(value, { stream: true });

      // A trailing "\r" may be the first half of a "\r\n" split across
      // chunks: hold it back until the next chunk shows which it is.
      const held = buffer.endsWith('\r') ? '\r' : '';
      const lines = (held ? buffer.slice(0, -1) : buffer).split(/\r\n|\r|\n/);
      buffer = (lines.pop() ?? '') + held;

      for (const line of lines) {
        if (line === '') {
          if (data.length > 0) {
            yield { event: event || 'message', data: data.join('\n') };
          }
          event = '';
          data = [];
          continue;
        }
        if (line.startsWith(':')) continue;

        const colon = line.indexOf(':');
        const field = colon === -1 ? line : line.slice(0, colon);
        let fieldValue = colon === -1 ? '' : line.slice(colon + 1);
        if (fieldValue.startsWith(' ')) fieldValue = fieldValue.slice(1);

        if (field === 'event') event = fieldValue;
        else if (field === 'data') data.push(fieldValue);
      }
    }
  } finally {
    if (!finished) {
      await reader.cancel().catch(() => undefined);
    }
    reader.releaseLock();
  }
}
