import { Routes } from '@angular/router';

import { adminGuard, authGuard } from './core/auth-http';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'fragen' },
  {
    path: 'anmelden',
    title: 'Anmelden · Normeon',
    loadComponent: () => import('./login/login-page').then((m) => m.LoginPage),
  },
  {
    path: 'fragen',
    title: 'Fragen · Normeon',
    canActivate: [authGuard],
    loadComponent: () => import('./chat/chat-page').then((m) => m.ChatPage),
  },
  {
    path: 'dokumente',
    title: 'Dokumente · Normeon',
    canActivate: [authGuard],
    loadComponent: () => import('./documents/documents-page').then((m) => m.DocumentsPage),
  },
  {
    path: 'statistik',
    title: 'Statistik · Normeon',
    canActivate: [adminGuard],
    loadComponent: () => import('./admin/stats-page').then((m) => m.StatsPage),
  },
  {
    path: '**',
    title: 'Nicht gefunden · Normeon',
    loadComponent: () => import('./not-found/not-found-page').then((m) => m.NotFoundPage),
  },
];
