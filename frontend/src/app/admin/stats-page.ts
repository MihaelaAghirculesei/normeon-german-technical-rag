import { DatePipe, DecimalPipe, PercentPipe } from '@angular/common';
import { Component, DestroyRef, InjectionToken, inject, signal } from '@angular/core';

import { adminStatsResource } from '../api/admin';
import { Notice } from '../shared/notice';
import { PageHeader } from '../shared/page-header';

/** How often the page refreshes itself, in milliseconds. */
export const STATS_REFRESH_MS = new InjectionToken<number>('STATS_REFRESH_MS', {
  providedIn: 'root',
  factory: () => 5000,
});

const PHASES = [
  { key: 'retrieval_ms', label: 'Suche und Bewertung' },
  { key: 'generation_ms', label: 'Antwort erstellen' },
  { key: 'total_ms', label: 'Gesamt' },
] as const;

/** Live numbers for the admin: questions, abstentions, cost, latency per
 * phase, documents, the latest questions, and the server's own counters. */
@Component({
  selector: 'app-stats-page',
  imports: [DatePipe, DecimalPipe, PercentPipe, Notice, PageHeader],
  templateUrl: './stats-page.html',
  styleUrl: './stats-page.css',
})
export class StatsPage {
  protected readonly hours = signal(24);
  protected readonly stats = adminStatsResource(() => this.hours());
  protected readonly phases = PHASES;

  constructor() {
    const timer = setInterval(() => this.stats.reload(), inject(STATS_REFRESH_MS));
    inject(DestroyRef).onDestroy(() => clearInterval(timer));
  }

  protected setHours(event: Event): void {
    this.hours.set(Number((event.target as HTMLSelectElement).value));
  }

  /** Invented citations as a share of all citation markers, if any. */
  protected hallucinationRate(server: {
    citations: number;
    hallucinated_citations: number;
  }): number | null {
    const all = server.citations + server.hallucinated_citations;
    return all === 0 ? null : server.hallucinated_citations / all;
  }
}
