import { segmentAnswer, type AnswerSegment } from './answer-segments';

export type Inline = AnswerSegment | { kind: 'strong'; text: string };

export type AnswerBlock =
  { kind: 'paragraph'; inlines: Inline[] } | { kind: 'list'; items: Inline[][] };

const BULLET = /^\s*[*-]\s+(.*)$/;
const STRONG = /\*\*([^*\n]+?)\*\*/g;

/**
 * Lay out the model's answer: lines starting with `* ` or `- ` become a
 * bulleted list, `**text**` becomes bold, blank lines separate paragraphs.
 * That is all the markdown it takes; everything else -- headings, links,
 * a `**` still waiting for its closing pair while streaming -- stays
 * literal text. Nothing is rendered as HTML, so the answer cannot inject
 * markup.
 */
export function formatAnswer(text: string, knownMarkers: ReadonlySet<string>): AnswerBlock[] {
  const blocks: AnswerBlock[] = [];
  let paragraph: string[] = [];
  let items: string[] = [];

  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push({ kind: 'paragraph', inlines: inlines(paragraph.join('\n'), knownMarkers) });
      paragraph = [];
    }
  };
  const flushList = () => {
    if (items.length) {
      blocks.push({ kind: 'list', items: items.map((item) => inlines(item, knownMarkers)) });
      items = [];
    }
  };

  for (const line of text.split('\n')) {
    const bullet = BULLET.exec(line);
    if (bullet) {
      flushParagraph();
      items.push(bullet[1]);
    } else if (!line.trim()) {
      flushParagraph();
      flushList();
    } else {
      flushList();
      paragraph.push(line);
    }
  }
  flushParagraph();
  flushList();
  return blocks;
}

function inlines(text: string, knownMarkers: ReadonlySet<string>): Inline[] {
  return segmentAnswer(text, knownMarkers).flatMap((segment) =>
    segment.kind === 'text' ? splitStrong(segment.text) : [segment],
  );
}

function splitStrong(text: string): Inline[] {
  const out: Inline[] = [];
  let last = 0;
  for (const match of text.matchAll(STRONG)) {
    if (match.index > last) out.push({ kind: 'text', text: text.slice(last, match.index) });
    out.push({ kind: 'strong', text: match[1] });
    last = match.index + match[0].length;
  }
  if (last < text.length) out.push({ kind: 'text', text: text.slice(last) });
  return out;
}
