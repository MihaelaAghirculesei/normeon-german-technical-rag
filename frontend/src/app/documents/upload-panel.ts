import { DecimalPipe } from '@angular/common';
import { Component, computed, inject, output, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import { StatusBadge } from '../shared/status-badge';
import { SUPPORTED_EXTENSIONS, UploadQueue } from './upload-queue';

/** Drag-and-drop (or pick) files, then follow each one from upload
 * through ingestion. */
@Component({
  selector: 'app-upload-panel',
  imports: [DecimalPipe, StatusBadge],
  providers: [UploadQueue],
  templateUrl: './upload-panel.html',
  styleUrl: './upload-panel.css',
})
export class UploadPanel {
  protected readonly queue = inject(UploadQueue);
  protected readonly accept = SUPPORTED_EXTENSIONS.join(',');
  protected readonly docType = signal('norm');
  protected readonly versionLabel = signal('');
  protected readonly dragging = signal(false);
  protected readonly hasFinished = computed(() =>
    this.queue.items().some((i) => i.state !== 'uploading' && i.state !== 'processing'),
  );

  /** The document list should be refreshed. */
  readonly changed = output<void>();

  constructor() {
    this.queue.changed.pipe(takeUntilDestroyed()).subscribe(() => this.changed.emit());
  }

  protected onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.dragging.set(true);
  }

  protected onDrop(event: DragEvent): void {
    event.preventDefault();
    this.dragging.set(false);
    this.enqueue(event.dataTransfer?.files);
  }

  protected onPick(input: HTMLInputElement): void {
    this.enqueue(input.files);
    input.value = ''; // picking the same file again must fire `change` again
  }

  private enqueue(files: FileList | null | undefined): void {
    if (!files || files.length === 0) return;
    this.queue.add(Array.from(files), {
      docType: this.docType(),
      versionLabel: this.versionLabel(),
    });
  }
}
