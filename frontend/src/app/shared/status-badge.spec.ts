import { TestBed } from '@angular/core/testing';

import { StatusBadge } from './status-badge';

describe('StatusBadge', () => {
  async function render(status: string) {
    const fixture = TestBed.createComponent(StatusBadge);
    fixture.componentRef.setInput('status', status);
    await fixture.whenStable();
    return (fixture.nativeElement as HTMLElement).querySelector('.badge')!;
  }

  it('labels a known status in German with its tone', async () => {
    const badge = await render('failed');

    expect(badge.textContent).toBe('Fehler');
    expect(badge.classList).toContain('danger');
  });

  it('falls back to the raw value for a status it does not know', async () => {
    const badge = await render('archived');

    expect(badge.textContent).toBe('archived');
    expect(badge.classList).toContain('neutral');
  });
});
