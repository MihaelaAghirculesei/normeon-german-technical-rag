import { HttpClient } from '@angular/common/http';
import {
  Component,
  ElementRef,
  InjectionToken,
  afterRenderEffect,
  computed,
  effect,
  inject,
  input,
  linkedSignal,
  output,
  signal,
  viewChild,
} from '@angular/core';
import { firstValueFrom } from 'rxjs';

import type { ContextChunk } from '../api/chunks';

export type Region = ContextChunk['regions'][number];

/** The part of pdf.js this component uses. */
export interface PdfPage {
  getViewport(options: { scale: number }): { width: number; height: number };
  render(options: { canvas: HTMLCanvasElement; viewport: unknown }): {
    promise: Promise<unknown>;
    cancel(): void;
  };
}

export interface PdfDocument {
  getPage(page: number): Promise<PdfPage>;
  destroy(): Promise<unknown>;
}

export type PdfLoader = (data: ArrayBuffer) => Promise<PdfDocument>;

/** pdf.js is large, so it is only fetched when a reader opens a PDF. */
const loadWithPdfJs: PdfLoader = async (data) => {
  const pdfjs = await import('pdfjs-dist');
  pdfjs.GlobalWorkerOptions.workerSrc = '/pdfjs/pdf.worker.min.mjs';
  const task = pdfjs.getDocument({ data });
  const pdf = await task.promise;
  return {
    getPage: (page) => pdf.getPage(page) as Promise<unknown> as Promise<PdfPage>,
    destroy: () => task.destroy(),
  };
};

export const PDF_LOADER = new InjectionToken<PdfLoader>('PDF_LOADER', {
  factory: () => loadWithPdfJs,
});

const FALLBACK_WIDTH_PX = 600;

/** One page of the original PDF with the cited passage highlighted. Emits
 * `unavailable` when the file cannot be shown, so the panel can keep the
 * text view. */
@Component({
  selector: 'app-pdf-viewer',
  template: `
    @if (pages().length > 1) {
      <nav aria-label="Zitierte Seiten">
        <button type="button" [disabled]="index() === 0" (click)="move(-1)">Zurück</button>
        <span>Seite {{ page() }}</span>
        <button type="button" [disabled]="index() === pages().length - 1" (click)="move(1)">
          Weiter
        </button>
      </nav>
    }
    @if (!rendered()) {
      <p class="loading">PDF wird geladen…</p>
    }
    <div class="sheet" [class.hidden]="!rendered()">
      <canvas #canvas [attr.aria-label]="'Seite ' + page() + ' des Originals'"></canvas>
      @for (box of highlights(); track $index) {
        <mark
          aria-label="Zitierte Stelle"
          [style.left.px]="box.left"
          [style.top.px]="box.top"
          [style.width.px]="box.width"
          [style.height.px]="box.height"
        ></mark>
      }
    </div>
  `,
  styleUrl: './pdf-viewer.css',
})
export class PdfViewer {
  readonly documentId = input.required<string>();
  readonly regions = input.required<readonly Region[]>();
  readonly unavailable = output<void>();

  private readonly http = inject(HttpClient);
  private readonly load = inject(PDF_LOADER);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly canvas = viewChild.required<ElementRef<HTMLCanvasElement>>('canvas');

  private readonly pdf = signal<PdfDocument | null>(null);
  private readonly scale = signal(1);
  protected readonly rendered = signal(false);

  protected readonly pages = computed(() =>
    [...new Set(this.regions().map((region) => region.page))].sort((a, b) => a - b),
  );
  protected readonly index = linkedSignal({ source: this.pages, computation: () => 0 });
  protected readonly page = computed(() => this.pages()[this.index()]);
  /** Regions are in PDF points from the top left, as CSS pixels are. */
  protected readonly highlights = computed(() => {
    const scale = this.scale();
    return this.regions()
      .filter((region) => region.page === this.page())
      .map(({ bbox: [x0, y0, x1, y1] }) => ({
        left: x0 * scale,
        top: y0 * scale,
        width: (x1 - x0) * scale,
        height: (y1 - y0) * scale,
      }));
  });

  constructor() {
    effect((onCleanup) => {
      const url = `/api/v1/documents/${encodeURIComponent(this.documentId())}/file`;
      let current: PdfDocument | null = null;
      let stale = false;
      onCleanup(() => {
        stale = true;
        this.pdf.set(null);
        void current?.destroy();
      });
      firstValueFrom(this.http.get(url, { responseType: 'arraybuffer' }))
        .then((data) => this.load(data))
        .then((pdf) => {
          if (stale) {
            void pdf.destroy();
            return;
          }
          current = pdf;
          this.pdf.set(pdf);
        })
        .catch(() => {
          if (!stale) this.unavailable.emit();
        });
    });

    afterRenderEffect((onCleanup) => {
      const pdf = this.pdf();
      const page = this.page();
      if (pdf === null || page === undefined) return;
      const canvas = this.canvas().nativeElement;
      let stale = false;
      let cancel = () => {};
      onCleanup(() => {
        stale = true;
        cancel();
      });
      this.rendered.set(false);
      void this.draw(pdf, page, canvas, (task) => (cancel = task))
        .then(() => {
          if (stale) return;
          this.rendered.set(true);
          this.host.nativeElement.querySelector('mark')?.scrollIntoView?.({ block: 'center' });
        })
        .catch(() => {
          // Moving to another page cancels the drawing of the previous one.
          if (!stale) this.unavailable.emit();
        });
    });
  }

  protected move(by: number): void {
    this.index.update((index) => index + by);
  }

  private async draw(
    pdf: PdfDocument,
    number: number,
    canvas: HTMLCanvasElement,
    onCancel: (cancel: () => void) => void,
  ): Promise<void> {
    const page = await pdf.getPage(number);
    const width = this.host.nativeElement.clientWidth || FALLBACK_WIDTH_PX;
    const scale = width / page.getViewport({ scale: 1 }).width;
    this.scale.set(scale);
    // Drawn at the device's resolution and shown at CSS size, so text stays sharp.
    const viewport = page.getViewport({ scale: scale * (globalThis.devicePixelRatio || 1) });
    canvas.width = viewport.width;
    canvas.height = viewport.height;
    const task = page.render({ canvas, viewport });
    onCancel(() => task.cancel());
    await task.promise;
  }
}
