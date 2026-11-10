import { Component, computed, input, output } from '@angular/core';

import type { DocumentSummary } from '../api/api-types';
import { Notice } from '../shared/notice';
import { examplesFor } from './examples';

/**
 * What Normeon answers, shown before the first question: only the tenant's
 * own documents, with sources -- and which documents those are, with a few
 * questions they answer. Without it, a general question gets a correct
 * "not in the documents" that looks like a broken app.
 */
@Component({
  selector: 'app-chat-intro',
  imports: [Notice],
  templateUrl: './chat-intro.html',
  styleUrl: './chat-intro.css',
})
export class ChatIntro {
  /** All of the tenant's documents; only the ready ones are listed. */
  readonly documents = input.required<readonly DocumentSummary[]>();
  /** False until the document list has loaded, so nothing flickers. */
  readonly loaded = input(true);
  readonly ask = output<string>();

  protected readonly ready = computed(() =>
    this.documents().filter((doc) => doc.status === 'ready'),
  );
  protected readonly examples = computed(() =>
    examplesFor(this.ready().map((doc) => doc.filename)),
  );
}
