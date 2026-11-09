import { DecimalPipe, NgTemplateOutlet } from '@angular/common';
import { Component, computed, inject, signal } from '@angular/core';

import { Notice } from '../shared/notice';
import { PageHeader } from '../shared/page-header';
import { formatAnswer, type Inline } from './answer-format';
import { CitationPanel } from './citation-panel';
import { ChatSession } from './chat-session';
import { StageProgress } from './stage-progress';

@Component({
  selector: 'app-chat-page',
  imports: [CitationPanel, DecimalPipe, NgTemplateOutlet, Notice, PageHeader, StageProgress],
  providers: [ChatSession],
  templateUrl: './chat-page.html',
  styleUrl: './chat-page.css',
})
export class ChatPage {
  protected readonly chat = inject(ChatSession);
  protected readonly draft = signal('');
  protected readonly selectedMarker = signal<string | null>(null);

  private readonly sourceMarkers = computed(
    () => new Set(this.chat.sources().map((s) => s.marker)),
  );
  protected readonly blocks = computed(() =>
    formatAnswer(this.chat.answer(), this.sourceMarkers()),
  );
  protected readonly selectedSource = computed(
    () => this.chat.sources().find((s) => s.marker === this.selectedMarker()) ?? null,
  );

  /** Types the context of the inline template, which Angular leaves untyped. */
  protected asInlines(inlines: unknown): Inline[] {
    return inlines as Inline[];
  }

  protected submit(event?: Event): void {
    event?.preventDefault();
    const question = this.draft().trim();
    if (!question || this.chat.streaming()) return;
    this.selectedMarker.set(null);
    void this.chat.ask(question);
  }

  /** Enter sends, Shift+Enter starts a new line. */
  protected onKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      this.submit(event);
    }
  }

  protected select(marker: string): void {
    this.selectedMarker.set(marker);
  }
}
