import { segmentAnswer } from './answer-segments';

const KNOWN = new Set(['S1', 'S2']);

describe('segmentAnswer', () => {
  it('turns known markers into citations between text', () => {
    expect(segmentAnswer('Höchstens 250 N [S1]. Siehe auch [S2]', KNOWN)).toEqual([
      { kind: 'text', text: 'Höchstens 250 N ' },
      { kind: 'citation', marker: 'S1' },
      { kind: 'text', text: '. Siehe auch ' },
      { kind: 'citation', marker: 'S2' },
    ]);
  });

  it('keeps adjacent markers as separate citations', () => {
    expect(segmentAnswer('[S1][S2]', KNOWN)).toEqual([
      { kind: 'citation', marker: 'S1' },
      { kind: 'citation', marker: 'S2' },
    ]);
  });

  it('leaves an unknown marker as literal text', () => {
    expect(segmentAnswer('Wert [S9] fehlt', KNOWN)).toEqual([
      { kind: 'text', text: 'Wert [S9] fehlt' },
    ]);
  });

  it('leaves a half-streamed marker as text until it completes', () => {
    expect(segmentAnswer('Höchstens 250 N [S', KNOWN)).toEqual([
      { kind: 'text', text: 'Höchstens 250 N [S' },
    ]);
  });

  it('returns nothing for an empty answer', () => {
    expect(segmentAnswer('', KNOWN)).toEqual([]);
  });
});
