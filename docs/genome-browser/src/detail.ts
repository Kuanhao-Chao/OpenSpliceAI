import type { DataSource } from './data';
import { detailHeight, draw, type DetailPane, type Palette } from './paint';
import type { BrowserState, ExactPlot, LoadedView, View } from './types';
import { baseAt, covers, requestedView, type PaneGeometry } from './viewport';

interface RecordView { view: View; filter: string; exact: boolean; plot?: ExactPlot; error?: string }
interface Request { key: string; abort: AbortController }
const filterKey = (state: BrowserState) => JSON.stringify([state.model, state.alt, state.gene, state.threshold]);
const requestKey = (view: View, filter: string) => JSON.stringify([view.chrom, view.start, view.end, filter]);

/** Geometry paints synchronously from cached arrays; decoding never holds up a drag frame. */
export class DetailPanel {
  private records = new Map<string, RecordView>();
  private requests = new Map<string, Request>();
  private running = 0;
  private disposed = false;
  private timer?: ReturnType<typeof setTimeout>;
  private frame = 0;
  private activePane = 0;
  private resizeObserver: ResizeObserver;
  private details: HTMLDetailsElement;
  constructor(private canvas: HTMLCanvasElement, private source: DataSource,
    private geometry: () => PaneGeometry[], private state: () => BrowserState, private palette: () => Palette,
    private select: (pane: PaneGeometry, position: number) => Promise<void>, private report: (error: unknown) => void) {
    this.details = canvas.closest('details')!;
    this.resizeObserver = new ResizeObserver(this.synchronize);
    this.resizeObserver.observe(canvas.parentElement!);
    this.details.addEventListener('toggle', this.synchronize);
    canvas.addEventListener('click', this.click);
    this.synchronize();
  }
  use(data: LoadedView, activePane: number) {
    this.activePane = activePane;
    const pane = this.geometry().find(p => p.index === activePane), filter = filterKey(this.state());
    if (pane && pane.chrom === data.view.chrom && (!data.exact || data.query)) {
      const previous = this.records.get(pane.id);
      if (!previous || previous.filter !== filter || previous.error || !covers(previous.view, data.view)) {
        this.records.set(pane.id, { view: data.view, exact: data.exact, plot: data.query?.plot, filter });
      }
    }
    this.synchronize();
  }
  private wanted() {
    return this.geometry().map(geometry => {
      const contig = this.source.manifest.contigs.find(c => c.name === geometry.chrom)!;
      const view = requestedView(geometry, contig.length), filter = filterKey(this.state());
      return { geometry, view, filter, key: requestKey(view, filter) };
    });
  }
  synchronize = () => {
    if (this.disposed) return;
    const wanted = this.wanted(), ids = new Set(wanted.map(p => p.geometry.id));
    for (const id of this.records.keys()) if (!ids.has(id)) this.records.delete(id);
    for (const [id, request] of this.requests) {
      if (!this.details.open || !wanted.some(p => p.geometry.id === id && p.key === request.key)) request.abort.abort();
    }
    if (!this.frame) this.frame = requestAnimationFrame(() => { this.frame = 0; this.paint(); });
    clearTimeout(this.timer);
    if (this.details.open) this.timer = setTimeout(() => this.pump(), 120);
  };
  private paint() {
    if (this.disposed || !this.details.open || !this.canvas.clientWidth) return;
    const state = this.state(), rect = this.canvas.getBoundingClientRect(), height = detailHeight(state);
    const ratio = devicePixelRatio || 1, width = rect.width;
    const bitmapWidth = Math.max(1, Math.round(width * ratio)), bitmapHeight = Math.round(height * ratio);
    if (this.canvas.width !== bitmapWidth) this.canvas.width = bitmapWidth;
    if (this.canvas.height !== bitmapHeight) this.canvas.height = bitmapHeight;
    this.canvas.style.height = `${height}px`;
    const ctx = this.canvas.getContext('2d')!;
    ctx.setTransform(bitmapWidth / width, 0, 0, bitmapHeight / height, 0, 0);
    const wanted = this.wanted();
    const panes: DetailPane[] = wanted.map(({ geometry, view, filter }) => {
      const cached = this.records.get(geometry.id), record = cached?.filter === filter && cached.view.chrom === geometry.chrom ? cached : undefined;
      return { geometry, ...record, pending: !record || !covers(record.view, view) };
    });
    draw(ctx, width, state, panes, this.activePane, this.palette(), rect.left, bitmapWidth / width);
    const ready = panes.every(p => p.geometry.ready && !p.pending && !p.error);
    this.canvas.dataset.ready = String(ready);
    const status = document.getElementById('matrix-status');
    if (status) status.textContent = panes.length ? `${panes.length} aligned region${panes.length === 1 ? '' : 's'} · ${panes.filter(p => p.exact && !p.pending && !p.error).length} exact plot${panes.length === 1 ? '' : 's'} · inspector: region ${this.activePane + 1}${panes.some(p => p.error) ? ' · data failed; use Retry data' : ready ? '' : ' · loading'}` : 'Choose a chromosome region for exact scores.';
  }
  private pump() {
    if (this.disposed || !this.details.open) return;
    const wanted = this.wanted().sort((a, b) => Number(b.geometry.index === this.activePane) - Number(a.geometry.index === this.activePane));
    for (const target of wanted) {
      if (this.running >= 2) break;
      const { geometry, view, filter, key } = target, record = this.records.get(geometry.id);
      if (!geometry.ready || view.end <= view.start || this.requests.has(geometry.id) || record?.filter === filter && covers(record.view, view)) continue;
      const request = { key, abort: new AbortController() };
      this.requests.set(geometry.id, request); this.running++;
      const state = { ...structuredClone(this.state()), ...view };
      void (async () => {
        let data: LoadedView | undefined;
        try {
          data = await this.source.load(view, request.abort.signal);
          if (request.abort.signal.aborted) return;
          const plot = data.queryToken === undefined ? undefined : await this.source.resources.decoder.query<ExactPlot>(data.queryToken, state, 0, 0, undefined, 'plot');
          if (!this.disposed && !request.abort.signal.aborted) this.records.set(geometry.id, { view, filter, exact: data.exact, plot });
        } catch (error) {
          if (!this.disposed && !request.abort.signal.aborted) {
            this.records.set(geometry.id, { view, filter, exact: false, error: error instanceof Error ? error.message : String(error) });
            this.report(error);
          }
        } finally {
          this.source.resources.decoder.release(data?.queryToken);
          this.requests.delete(geometry.id); this.running--;
          if (!this.disposed) { this.synchronize(); this.pump(); }
        }
      })();
    }
  }
  retry() { for (const [id, record] of this.records) if (record.error) this.records.delete(id); this.synchronize(); }
  private click = (event: MouseEvent) => {
    const pane = this.geometry().find(p => event.clientX >= p.left && event.clientX < p.left + p.width);
    if (!pane) return;
    const position = baseAt(pane, event.clientX);
    const length = this.source.manifest.contigs.find(c => c.name === pane.chrom)!.length;
    if (position !== null && position >= 0 && position < length) void this.select(pane, position).catch(this.report);
  };
  destroy() {
    this.disposed = true; clearTimeout(this.timer); cancelAnimationFrame(this.frame);
    this.resizeObserver.disconnect(); this.details.removeEventListener('toggle', this.synchronize);
    this.canvas.removeEventListener('click', this.click);
    for (const request of this.requests.values()) request.abort.abort();
    this.records.clear();
  }
}
