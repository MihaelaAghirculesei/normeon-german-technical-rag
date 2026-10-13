import { DecimalPipe } from '@angular/common';
import { Component, inject, signal } from '@angular/core';

import { Notice } from '../shared/notice';
import { PageHeader } from '../shared/page-header';
import { ChatSession } from './chat-session';
import { StageProgress } from './stage-progress';

@Component({
  selector: 'app-chat-page',
  imports: [DecimalPipe, Notice, PageHeader, StageProgress],
  providers: [ChatSession],
  templateUrl: './chat-page.html',
  styleUrl: './chat-page.css',
})
export class ChatPage {
  protected readonly chat = inject(ChatSession);
  protected readonly draft = signal('');

  protected submit(event?: Event): void {
    event?.preventDefault();
    const question = this.draft().trim();
    if (!question || this.chat.streaming()) return;
    void this.chat.ask(question);
  }

  /** Enter sends, Shift+Enter starts a new line. */
  protected onKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      this.submit(event);
    }
  }
}
