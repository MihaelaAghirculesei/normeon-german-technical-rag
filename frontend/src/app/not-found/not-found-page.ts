import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';

import { PageHeader } from '../shared/page-header';

@Component({
  selector: 'app-not-found-page',
  imports: [PageHeader, RouterLink],
  template: `
    <app-page-header heading="Seite nicht gefunden" />
    <p>Diese Adresse gibt es nicht. <a routerLink="/dokumente">Zu den Dokumenten</a></p>
  `,
})
export class NotFoundPage {}
