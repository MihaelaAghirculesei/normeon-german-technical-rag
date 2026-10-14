import { provideHttpClient } from '@angular/common/http';
import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { App } from './app';
import { signedIn, signedOut } from './core/testing';

describe('App', () => {
  afterEach(() => signedOut());

  async function render(): Promise<HTMLElement> {
    TestBed.configureTestingModule({
      imports: [App],
      providers: [provideRouter([]), provideHttpClient()],
    });
    const fixture = TestBed.createComponent(App);
    await fixture.whenStable();
    return fixture.nativeElement as HTMLElement;
  }

  it('shows the navigation and who is signed in, with their tenant', async () => {
    signedIn();
    const el = await render();

    const links = [...el.querySelectorAll('nav a')].map((a) => a.getAttribute('href'));
    expect(links).toEqual(['/fragen', '/dokumente']);
    expect(el.querySelector('.who')?.textContent).toContain('anna@demo.normeon.dev');
    expect(el.querySelector('.who')?.textContent).toContain('Demo Tenant');
  });

  it('shows no navigation to a signed-out visitor', async () => {
    signedOut();
    const el = await render();

    expect(el.querySelector('.brand')?.textContent).toContain('Normeon');
    expect(el.querySelector('nav')).toBeNull();
  });
});
