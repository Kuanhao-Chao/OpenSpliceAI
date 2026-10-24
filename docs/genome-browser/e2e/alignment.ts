import { expect, test, type Page } from '@playwright/test';

/** Observe actual Canvas2D draw calls, independently of the application's geometry helpers. */
export async function captureDrawing(page: Page) {
  await page.addInitScript(() => {
    const original = CanvasRenderingContext2D.prototype.fillRect;
    CanvasRenderingContext2D.prototype.fillRect = function(x, y, width, height) {
      const canvas = this.canvas as HTMLCanvasElement & { drawnRects?: { x: number; y: number; width: number; height: number; color: string; alpha: number }[] };
      if (canvas.id === 'track-canvas' && x === 0 && y === 0 && height > 50) canvas.drawnRects = [];
      const records = canvas.drawnRects ||= [];
      if (records.length < 20000) records.push({ x, y, width, height, color: String(this.fillStyle), alpha: this.globalAlpha });
      original.call(this, x, y, width, height);
    };
  });
  await page.reload();
  await expect(page.locator('#igv-browser')).toHaveAttribute('data-ready', 'true');
}

export async function addAlignmentMarkers(page: Page) {
  // Independent one-base native bedGraph features across two authenticated, fully scored windows.
  const windows: [string, number, number][] = [['chr7', 117503339, 117505387], ['chr13', 32315503, 32317551]];
  const lines = ['track type=bedGraph name=AlignmentMarkers color=255,0,255'];
  for (const [chr, start, end] of windows) for (let p = start; p < end; p++) lines.push(`${chr}\t${p}\t${p + 1}\t1`);
  await page.locator('#track-files').setInputFiles({ name: 'alignment-markers.bedgraph', mimeType: 'text/plain', buffer: Buffer.from(lines.join('\n') + '\n') });
  await expect(page.locator('#track-status')).toContainText('loaded');
}

export async function measuredAlignment(page: Page, panes: number) {
  await expect(page.locator('#track-canvas')).toHaveAttribute('data-ready', 'true');
  const check = async () => page.evaluate(() => {
    type Rect = { x: number; y: number; width: number; height: number; color: string; alpha: number };
    type Canvas = HTMLCanvasElement & { drawnRects?: Rect[]; _data?: { bpStart: number; bpPerPixel: number } };
    const matrix = document.querySelector<Canvas>('#track-canvas')!, bounds = matrix.getBoundingClientRect();
    const scale = matrix.width / bounds.width, draws = matrix.drawnRects || [];
    const root = document.querySelector('#igv-browser')!.shadowRoot!;
    return [...root.querySelectorAll('.igv-column')].map((column, pane) => {
      const native = [...column.querySelectorAll<Canvas>('.igv-viewport canvas')].find(c => c.drawnRects?.some(r => r.color === '#ff00ff' || r.color === 'rgb(255, 0, 255)'));
      if (!native?._data) return { pane, samples: 0, error: 'Native marker track has not painted' };
      const nativeBounds = native.getBoundingClientRect(), viewport = native.closest('.igv-viewport')!.getBoundingClientRect();
      const markers = native.drawnRects!.filter(r => (r.color === '#ff00ff' || r.color === 'rgb(255, 0, 255)') && r.width > 2)
        .map(r => ({ ...r, screenX: nativeBounds.left + r.x }))
        .filter(r => r.screenX > viewport.left + 2 && r.screenX + r.width < viewport.right - 2);
      const samples = [markers[1], markers[Math.floor(markers.length / 2)], markers[markers.length - 2]].filter(Boolean);
      const errors = samples.map(marker => {
        const cell = draws.filter(r => r.y >= 84 && r.y < 156 && r.alpha < 1 && r.height > 2)
          .sort((a, b) => Math.abs(bounds.left + a.x - marker.screenX) - Math.abs(bounds.left + b.x - marker.screenX))[0];
        const difference = draws.filter(r => r.y >= 184 && r.y <= 222 && r.alpha === 1 && r.width < viewport.width / 2 && r.height <= 20)
          .sort((a, b) => Math.abs(bounds.left + a.x - marker.screenX) - Math.abs(bounds.left + b.x - marker.screenX))[0];
        return { position: Math.round(native._data!.bpStart + marker.x * native._data!.bpPerPixel), nativeX: marker.screenX,
          heatmapError: cell ? Math.abs(bounds.left + cell.x - marker.screenX) * scale : 1e9,
          differenceError: difference ? Math.abs(bounds.left + difference.x - marker.screenX) * scale : 1e9,
          widthError: cell ? Math.abs(cell.width - marker.width) * scale : 1e9 };
      });
      return { pane, samples: samples.length, errors, maxError: Math.max(...errors.flatMap(e => [e.heatmapError, e.differenceError, e.widthError])) };
    });
  });
  try {
    await expect.poll(async () => {
      const result = await check();
      return result.length === panes && result.every(p => p.samples === 3 && p.maxError !== undefined && p.maxError <= 1.01);
    }, { timeout: 15000, message: 'Rendered matrix and native one-base marker columns must agree within one physical pixel' }).toBe(true);
  } catch (error) {
    await test.info().attach('alignment-failure.json', { body: JSON.stringify(await check(), null, 2), contentType: 'application/json' });
    throw error;
  }
  return check();
}

export async function nativeBase(page: Page, pane: number, fraction = .5) {
  return page.locator('#igv-browser .igv-column').nth(pane).evaluate((column, fraction) => {
    type Canvas = HTMLCanvasElement & { _data?: { bpStart: number; bpPerPixel: number; referenceFrame: { chr: string; start: number } } };
    const c = [...column.querySelectorAll<Canvas>('canvas')].find(c => c._data)!, d = c._data!;
    const viewport = c.closest('.igv-viewport')!.getBoundingClientRect(), canvas = c.getBoundingClientRect();
    const position = Math.floor(d.bpStart + (viewport.left + viewport.width * fraction - canvas.left) * d.bpPerPixel);
    return { chrom: d.referenceFrame.chr, position, screenX: canvas.left + (position + .5 - d.bpStart) / d.bpPerPixel };
  }, fraction);
}
