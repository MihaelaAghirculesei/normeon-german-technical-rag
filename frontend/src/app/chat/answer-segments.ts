export type AnswerSegment = { kind: 'text'; text: string } | { kind: 'citation'; marker: string };

// Same shape the backend validates (domain/citations.py: \[S\d+\]).
const MARKER = /\[(S\d+)\]/g;

/**
 * Split answer text into plain text and citation markers, so each `[S1]`
 * can render as a clickable chip. Only markers with a known source become
 * citations; anything else stays literal text -- including a marker still
 * half-streamed ("[S1" without its bracket), which completes on a later
 * token.
 */
export function segmentAnswer(text: string, knownMarkers: ReadonlySet<string>): AnswerSegment[] {
  const segments: AnswerSegment[] = [];
  let plain = '';
  let last = 0;
  for (const match of text.matchAll(MARKER)) {
    plain += text.slice(last, match.index);
    last = match.index + match[0].length;
    if (knownMarkers.has(match[1])) {
      if (plain) segments.push({ kind: 'text', text: plain });
      segments.push({ kind: 'citation', marker: match[1] });
      plain = '';
    } else {
      plain += match[0];
    }
  }
  plain += text.slice(last);
  if (plain) segments.push({ kind: 'text', text: plain });
  return segments;
}
