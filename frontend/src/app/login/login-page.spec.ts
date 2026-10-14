import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed, type ComponentFixture } from '@angular/core/testing';
import { Router, provideRouter } from '@angular/router';

import { signedOut } from '../core/testing';
import { ANNA } from '../core/testing';
import { LoginPage } from './login-page';

describe('LoginPage', () => {
  let fixture: ComponentFixture<LoginPage>;
  let el: HTMLElement;
  let http: HttpTestingController;

  beforeEach(async () => {
    signedOut();
    TestBed.configureTestingModule({
      imports: [LoginPage],
      providers: [provideRouter([]), provideHttpClient(), provideHttpClientTesting()],
    });
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(LoginPage);
    fixture.componentRef.setInput('weiter', '/dokumente');
    el = fixture.nativeElement as HTMLElement;
    await fixture.whenStable();
  });

  afterEach(() => signedOut());

  async function submit(email: string, password: string): Promise<void> {
    for (const [selector, value] of [
      ['input[type=email]', email],
      ['input[type=password]', password],
    ]) {
      const input = el.querySelector<HTMLInputElement>(selector)!;
      input.value = value;
      input.dispatchEvent(new Event('input'));
    }
    fixture.detectChanges();
    el.querySelector<HTMLButtonElement>('button[type=submit]')!.click();
  }

  it('signs in and continues to where the user was going', async () => {
    const navigate = vi.spyOn(TestBed.inject(Router), 'navigateByUrl').mockResolvedValue(true);

    await submit(' anna@demo.normeon.dev ', 'pw');
    const request = http.expectOne('/api/v1/auth/login');
    expect(request.request.body).toEqual({ email: 'anna@demo.normeon.dev', password: 'pw' });
    request.flush({
      access_token: 'abc',
      token_type: 'bearer',
      expires_at: new Date(Date.now() + 60_000).toISOString(),
      user: ANNA,
    });
    await fixture.whenStable();

    expect(navigate).toHaveBeenCalledWith('/dokumente');
  });

  it('says plainly when the credentials are wrong', async () => {
    await submit('anna@demo.normeon.dev', 'falsch');
    http
      .expectOne('/api/v1/auth/login')
      .flush({ code: 'invalid_credentials' }, { status: 401, statusText: 'Unauthorized' });
    await fixture.whenStable();

    expect(el.querySelector('[role=alert]')?.textContent).toContain(
      'E-Mail oder Passwort ist falsch.',
    );
  });
});
