/** A question known to get a cited answer from one of the demo documents. */
export interface ExampleQuestion {
  question: string;
  /** Shown only when the tenant has a ready document whose name starts with this. */
  document: string;
}

export const EXAMPLE_QUESTIONS: readonly ExampleQuestion[] = [
  { question: 'Welche Mindestprofiltiefe müssen die Reifen eines Pkw haben?', document: 'StVZO' },
  { question: 'Welche Lenkkraft fordert das Lastenheft?', document: 'Lastenheft' },
  {
    question: 'Welcher Buchstabe kennzeichnet ein Elektrofahrzeug auf dem Kennzeichen?',
    document: 'FZV',
  },
];

/** The examples the tenant's own documents can answer -- never a question
 * about a document it does not have. */
export function examplesFor(filenames: readonly string[]): string[] {
  return EXAMPLE_QUESTIONS.filter((example) =>
    filenames.some((name) => name.startsWith(example.document)),
  ).map((example) => example.question);
}
