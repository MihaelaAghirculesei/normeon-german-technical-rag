/** A question known to get a cited answer from one of the demo documents.
 * Each was asked five times on the deployed configuration and answered every time;
 * questions about the Lastenheft were left out on purpose: with two
 * versions of it in the corpus the model declines some of them part of
 * the time (#38), which would make the example look broken. */
export interface ExampleQuestion {
  question: string;
  /** Shown only when the tenant has a ready document whose name starts with this. */
  document: string;
}

export const EXAMPLE_QUESTIONS: readonly ExampleQuestion[] = [
  { question: 'Welche Mindestprofiltiefe müssen die Reifen eines Pkw haben?', document: 'StVZO' },
  { question: 'Was gilt für Nebelschlussleuchten?', document: 'StVZO' },
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
