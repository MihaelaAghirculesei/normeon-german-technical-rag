import { formatAnswer } from './answer-format';

const KNOWN = new Set(['S1', 'S2']);

describe('formatAnswer', () => {
  it('keeps a plain answer as one paragraph with its citations', () => {
    expect(formatAnswer('Höchstens 250 N [S1].', KNOWN)).toEqual([
      {
        kind: 'paragraph',
        inlines: [
          { kind: 'text', text: 'Höchstens 250 N ' },
          { kind: 'citation', marker: 'S1' },
          { kind: 'text', text: '.' },
        ],
      },
    ]);
  });

  it('turns bullet lines into a list and bold markup into bold text', () => {
    const answer = [
      'Die Versionen weichen ab:',
      '',
      '* In **v1.2** höchstens **300 N** [S1].',
      '- In **v2.0** höchstens **250 N** [S2].',
    ].join('\n');

    expect(formatAnswer(answer, KNOWN)).toEqual([
      { kind: 'paragraph', inlines: [{ kind: 'text', text: 'Die Versionen weichen ab:' }] },
      {
        kind: 'list',
        items: [
          [
            { kind: 'text', text: 'In ' },
            { kind: 'strong', text: 'v1.2' },
            { kind: 'text', text: ' höchstens ' },
            { kind: 'strong', text: '300 N' },
            { kind: 'text', text: ' ' },
            { kind: 'citation', marker: 'S1' },
            { kind: 'text', text: '.' },
          ],
          [
            { kind: 'text', text: 'In ' },
            { kind: 'strong', text: 'v2.0' },
            { kind: 'text', text: ' höchstens ' },
            { kind: 'strong', text: '250 N' },
            { kind: 'text', text: ' ' },
            { kind: 'citation', marker: 'S2' },
            { kind: 'text', text: '.' },
          ],
        ],
      },
    ]);
  });

  it('separates paragraphs on blank lines and keeps single line breaks', () => {
    expect(formatAnswer('Erste Zeile\nzweite Zeile\n\nNeuer Absatz', KNOWN)).toEqual([
      { kind: 'paragraph', inlines: [{ kind: 'text', text: 'Erste Zeile\nzweite Zeile' }] },
      { kind: 'paragraph', inlines: [{ kind: 'text', text: 'Neuer Absatz' }] },
    ]);
  });

  it('ends a list at the next ordinary line', () => {
    const blocks = formatAnswer('* eins\n* zwei\nDanach Text', KNOWN);
    expect(blocks.map((b) => b.kind)).toEqual(['list', 'paragraph']);
  });

  it('leaves bold markup still waiting for its closing pair as literal text', () => {
    expect(formatAnswer('Höchstens **250', KNOWN)).toEqual([
      { kind: 'paragraph', inlines: [{ kind: 'text', text: 'Höchstens **250' }] },
    ]);
  });

  it('does not treat a multiplication or a lone asterisk as markup', () => {
    expect(formatAnswer('2 * 3 = 6 und *kursiv*', KNOWN)).toEqual([
      { kind: 'paragraph', inlines: [{ kind: 'text', text: '2 * 3 = 6 und *kursiv*' }] },
    ]);
  });

  it('keeps markup that looks like HTML as text', () => {
    expect(formatAnswer('**<b>x</b>**', KNOWN)).toEqual([
      { kind: 'paragraph', inlines: [{ kind: 'strong', text: '<b>x</b>' }] },
    ]);
  });

  it('returns nothing for an empty answer', () => {
    expect(formatAnswer('', KNOWN)).toEqual([]);
  });
});
