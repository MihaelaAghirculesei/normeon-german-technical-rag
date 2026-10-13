import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'dokumente' },
  {
    path: 'dokumente',
    title: 'Dokumente · Normeon',
    loadComponent: () => import('./documents/documents-page').then((m) => m.DocumentsPage),
  },
  {
    path: '**',
    title: 'Nicht gefunden · Normeon',
    loadComponent: () => import('./not-found/not-found-page').then((m) => m.NotFoundPage),
  },
];
