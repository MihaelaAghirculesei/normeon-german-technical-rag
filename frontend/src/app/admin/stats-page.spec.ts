import { registerLocaleData } from '@angular/common';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import localeDe from '@angular/common/locales/de';
import { LOCALE_ID } from '@angular/core';
import { TestBed, type ComponentFixture } from '@angular/core/testing';

import type { AdminStats } from '../api/admin';
import { STATS_REFRESH_MS, StatsPage } from './stats-page';

registerLocaleData(localeDe);

const URL = '/api/v1/admin/stats?hours=24';

const BUSY: AdminStats = {
  window_hours: 24,
  queries: 12,
  abstention_rate: 0.25,
  cost_usd_total: 0.0123,
  cost_usd_per_query: 0.001025,
  latency_ms: {
    retrieval_ms: { p50: 640, p95: 1200 },
    generation_ms: { p50: 900, p95: 2100 },
    total_ms: { p50: 1540, p95: 3300 },
  },
  documents: { ready: 4, processing: 1, failed: 1, chunks: 1219 },
  recent: [
    {
      created_at: '2026-10-14T09:30:00',
      question: 'Welche Lenkkraft ist zulässig?',
      abstained: false,
      total_ms: 1540,
      cost_usd: 0.001,
    },
  ],
  server: { citations: 18, hallucinated_citations: 2, llm_errors: 0 },
};

const QUIET: AdminStats = {
  ...BUSY,
  queries: 0,
  abstention_rate: null,
  cost_usd_total: 0,
  cost_usd_per_query: null,
  latency_ms: null,
  recent: [],
};

describe('StatsPage', () => {
  let http: HttpTestingController;

  function setup(refreshMs = 60_000): void {
    TestBed.configureTestingModule({
      imports: [StatsPage],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: LOCALE_ID, useValue: 'de' },
        { provide: STATS_REFRESH_MS, useValue: refreshMs },
      ],
    });
    http = TestBed.inject(HttpTestingController);
  }

  let fixture: ComponentFixture<StatsPage>;

  async function render(stats: AdminStats): Promise<HTMLElement> {
    fixture = TestBed.createComponent(StatsPage);
    TestBed.tick();
    http.expectOne(URL).flush(stats);
    await fixture.whenStable();
    return fixture.nativeElement as HTMLElement;
  }

  afterEach(() => http.verify());

  it('shows the key figures the German way', async () => {
    setup();
    const el = await render(BUSY);

    // The German locale separates "25 %" with a non-breaking space.
    const tiles = [...el.querySelectorAll('.tile .value')].map((t) =>
      t.textContent?.replace(/\s+/g, ' ').trim(),
    );
    expect(tiles).toEqual(['12', '25 %', '0,0123 $', '3,3 s']);
    const rows = [...el.querySelectorAll('tbody tr')].map((r) => r.textContent ?? '');
    expect(rows[0]).toContain('Suche und Bewertung');
    expect(rows[0]).toContain('640 ms');
    expect(el.textContent).toContain('Welche Lenkkraft ist zulässig?');
    // 2 invented out of 20 markers
    expect(el.textContent?.replace(/\s+/g, ' ')).toContain('(10 %)');
  });

  it('says so when the window has no questions', async () => {
    setup();
    const el = await render(QUIET);

    expect(el.querySelector('.tile .value')?.textContent?.trim()).toBe('0');
    expect(el.textContent).toContain('Noch keine Fragen in diesem Zeitraum.');
  });

  it('refreshes itself', async () => {
    setup(10);
    // Not render(): with a refresh every 10 ms a request is always in
    // flight, so the app never becomes stable to wait for.
    fixture = TestBed.createComponent(StatsPage);
    TestBed.tick();
    http.expectOne(URL).flush(BUSY);

    // Wait for the next poll rather than a fixed time: the suite may be busy.
    let refreshed = 0;
    for (let tries = 0; tries < 200 && refreshed === 0; tries++) {
      await new Promise((resolve) => setTimeout(resolve, 10));
      TestBed.tick();
      const requests = http.match(URL);
      requests.forEach((r) => r.flush(BUSY));
      refreshed = requests.length;
    }

    expect(refreshed).toBeGreaterThan(0);
    // Leaving the page stops the timer; nothing may be left in flight.
    fixture.destroy();
    http.match(URL).forEach((r) => r.flush(BUSY));
  });
});
