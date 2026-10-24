import igv from 'igv';
import type { BrowserState, Descriptor, Model, View } from './types';
import type { DataSource } from './data';
import { EVENTS } from './types';
import { formatView } from './state';
import type { PaneGeometry } from './viewport';

export type Config = Record<string, unknown>;
type Track = { id: string; type: string; name: string; order: number; height?: number; reversed?: boolean; frameTranslate?: boolean; getState?(): Config; config: Config;
  trackView: { setTrackHeight(height: number, force: boolean): void } };
type Frame = { id: string; chr: string; start: number; end: number; bpPerPixel: number };
type NativeCanvas = HTMLCanvasElement & { _data?: { referenceFrame: Frame; bpStart: number; bpPerPixel: number } };
interface Engine {
  on(event: string, handler: (...args: never[]) => unknown): void;
  search(locus: string): Promise<void>; loadTrack(config: Config): Promise<Track>;
  findTracks(property: string, value: string): Track[]; removeTrack(track: Track): void;
  currentLoci(): string[]; toSVG(): string; reorderTracks(): void;
  trackViews: { track: Track }[];
  referenceFrameList: Frame[];
  roiManager: { toJSON(): Config[]; showOverlays: boolean; roiSets: {name?:string;getAllFeatures():Promise<Record<string,{chr:string;start:number;end:number;name?:string}[]>>}[] };
  loadROI(config: Config[]): Promise<void>; clearROIs(): void;
  getUserDefinedROIs(): Promise<{ chr: string; start: number; end: number; name?: string }[]>;
}
const colors = ['#0f9c88', '#2979d3', '#b15cce', '#e39828'];

/** IGV owns gestures; one operation queue protects the immutable snapshot view. */
export class GenomeEngine {
  private browser!: Engine;
  private updating = 0;
  private disposed = false;
  private files = new Map<string, Promise<File>>();
  private signatures = new Map<string, string>();
  private presentations = new Map<string, Config>();
  private pending = Promise.resolve();
  private pointerPane = 0;
  private presentationTimer?: ReturnType<typeof setTimeout>;
  private restored = new Map<string, Config>();
  private lastState?: BrowserState;
  private geometryFrame = 0;
  private resizeObserver?: ResizeObserver;
  private mutationObserver?: MutationObserver;
  views: View[] = [];
  constructor(private source: DataSource, private element: HTMLElement,
    private onLoci: (views: View[]) => void, private onPosition: (chrom: string, pos: number, pane: number) => void,
    private onPresentation: () => void = () => {}, private onGeometry: () => void = () => {}) {}

  private enqueue(action: () => Promise<void>) {
    const run = this.pending.catch(() => {}).then(async () => {
      if (this.disposed) return;
      this.updating++;
      try { await action(); } finally { this.updating--; this.geometryChanged(); }
    });
    this.pending = run; return run;
  }
  private async resource(d: Descriptor): Promise<string | File> {
    if (this.source.manifest.scope !== 'demonstration') return this.source.resources.url(d.path);
    let pending = this.files.get(d.path);
    if (!pending) {
      pending = this.source.resources.bytes(d.path, d).then(bytes => new File([bytes], d.path.split('/').at(-1)!));
      this.files.set(d.path, pending);
      pending.catch(() => this.files.delete(d.path));
    }
    return pending;
  }
  private logical(value: unknown): unknown {
    if (value instanceof File) {
      const d = this.source.manifest.files.find(d => d.path.split('/').at(-1) === value.name);
      return d ? { snapshotPath: d.path } : { localFile: value.name };
    }
    if (Array.isArray(value)) return value.map(v => this.logical(v));
    if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).filter(([,v]) => typeof v !== 'function' && !(v instanceof Promise)).map(([k,v]) => [k,this.logical(v)]));
    return value;
  }
  private async physical(value: unknown): Promise<unknown> {
    if (Array.isArray(value)) return Promise.all(value.map(v => this.physical(v)));
    if (value && typeof value === 'object') {
      const obj = value as Config;
      if (typeof obj.snapshotPath === 'string') {
        const d = this.source.manifest.files.find(d => d.path === obj.snapshotPath);
        if (!d) throw new Error('Session refers to a file outside the pinned snapshot');
        return this.resource(d);
      }
      if (obj.localFile) throw new Error(`Reselect local track ${String(obj.localFile)}`);
      return Object.fromEntries(await Promise.all(Object.entries(obj).map(async ([k,v]) => [k,await this.physical(v)])));
    }
    return value;
  }
  async create(state: BrowserState, saved?: Config) {
    const native = this.source.manifest.igv;
    if (!native) throw new Error('This snapshot requires the archived version-1 viewer');
    const r = native.reference;
    const reference: Config = { id: 'OpenSpliceAI-GRCh38p14', name: 'GRCh38.p14 · pinned scoring reference',
      fastaURL: await this.resource(r.fasta), chromosomeOrder: Array.from({ length: 22 }, (_, i) => `chr${i + 1}`).concat(['chrX', 'chrY', 'chrM']), wholeGenomeView: true };
    if (r.index) reference.indexURL = await this.resource(r.index);
    if (r.compressedIndex) reference.compressedIndexURL = await this.resource(r.compressedIndex);
    const config = { reference, locus: saved?.locus || formatView(state), tracks: [],
      showNavigation: true, showRuler: true, showIdeogram: true, showCursorGuide: true,
      showCenterGuide: true, showROIOverlays: saved?.showROIOverlays !== false, minimumBases: 20,
      loadDefaultGenomes: false, queryParametersSupported: false, showSVGButton: false, search: false };
    this.browser = await igv.createBrowser(this.element, config as unknown as Parameters<typeof igv.createBrowser>[1]) as unknown as Engine;
    const rawSearch = this.browser.search.bind(this.browser);
    const genes = await this.source.genes();
    this.browser.search = async text => {
      const terms = text.trim().split(/\s+/).map(term => {
        if (term.includes(':') || term === 'all' || this.source.manifest.contigs.some(c => c.name === term)) return term;
        const matches = genes.filter(g => g.name.toLowerCase() === term.toLowerCase());
        const gene = matches.find(g => g.chrom === this.views[0]?.chrom) || matches[0];
        if (!gene) throw new Error(`Gene not found in the scoring annotation: ${term}`);
        return formatView(gene).replaceAll(',', '');
      });
      await rawSearch(terms.join(' '));
    };
    this.browser.on('locuschange', ((frames: Frame[]) => {
      this.geometryChanged();
      const views = frames.filter(f => f.chr !== 'all').map(f => ({ chrom: f.chr, start: Math.max(0, Math.floor(f.start)), end: Math.ceil(f.end) }));
      const changed = JSON.stringify(views) !== JSON.stringify(this.views); this.views = views;
      if (!this.updating && changed) this.onLoci(views);
    }) as never);
    this.element.addEventListener('pointerdown', this.rememberPane);
    this.element.addEventListener('pointerup',this.rememberPresentation);
    this.browser.on('trackclick', ((track: Track, _popup: unknown, pos: number, features?: { chr?: string }[]) => {
      if (!track.id?.startsWith('osai-') || track.type === 'annotation' || track.type === 'sequence') return;
      const chrom = features?.find(f => f.chr)?.chr || this.views[this.pointerPane]?.chrom;
      if (chrom && Number.isFinite(pos)) this.onPosition(chrom, Math.floor(pos), this.pointerPane);
      return false;
    }) as never);
    this.browser.on('trackorderchanged', (() => { if (!this.updating) this.onPresentation(); }) as never);
    this.browser.on('trackdrag', this.geometryChanged as never);
    this.browser.on('trackdragend', this.geometryChanged as never);
    this.resizeObserver = new ResizeObserver(this.geometryChanged);
    this.resizeObserver.observe(this.element);
    const columns = this.element.shadowRoot?.querySelector('.igv-column-container');
    if (columns) {
      // Native repaint replaces canvases after async track loads, independently of locuschange.
      this.mutationObserver = new MutationObserver(this.geometryChanged);
      this.mutationObserver.observe(columns, { childList: true, subtree: true });
    }
    if (Array.isArray(saved?.tracks)) this.restored = new Map((saved.tracks as Config[]).map((t,i) => {const id=String(t.id || (t.type==='sequence'?'osai-reference':`user-saved-${i}`));return [id,{...t,id}];}));
    await this.sync(state);
    for (const track of this.restored.values()) {
      if (String(track.id).startsWith('osai-') || track.type === 'sequence' || track.id === 'ruler' || track.id === 'ideogram') continue;
      try { await this.loadTrackConfig(await this.physical(track) as Config); }
      catch (error) { if (!JSON.stringify(track).includes('localFile')) throw error; }
    }
    if (Array.isArray(saved?.roi)) await this.browser.loadROI(await this.physical(saved.roi) as Config[]);
    this.restored.clear();
    this.views=this.browser.referenceFrameList.filter(f=>f.chr!=='all').map(f=>({chrom:f.chr,start:Math.max(0,Math.floor(f.start)),end:Math.ceil(f.end)}));
    this.element.dataset.ready = 'true';
    this.geometryChanged();
  }
  private geometryChanged = () => {
    if (this.disposed || this.geometryFrame) return;
    this.geometryFrame = requestAnimationFrame(() => { this.geometryFrame = 0; if (!this.disposed) this.onGeometry(); });
  };
  geometry(): PaneGeometry[] {
    if (!this.browser) return [];
    const columns = this.element.shadowRoot?.querySelectorAll<HTMLElement>('.igv-column');
    return this.browser.referenceFrameList.flatMap((frame, index) => {
      if (frame.chr === 'all' || !columns?.[index]) return [];
      const column = columns[index];
      const canvas = [...column.querySelectorAll<NativeCanvas>('.igv-viewport canvas')].find(c =>
        c._data?.referenceFrame === frame && c._data.bpPerPixel === frame.bpPerPixel);
      const viewport = canvas?.closest<HTMLElement>('.igv-viewport');
      const rect = (viewport || column).getBoundingClientRect();
      const left = rect.left + (viewport?.clientLeft || 0);
      return [{ id: frame.id, index, chrom: frame.chr, start: frame.start, end: frame.end,
        bpPerPixel: frame.bpPerPixel, left, width: viewport?.clientWidth ?? column.clientWidth,
        originBase: canvas?._data?.bpStart ?? frame.start,
        originPixel: canvas ? canvas.getBoundingClientRect().left - left : 0, ready: !!canvas }];
    });
  }
  private rememberPresentation = () => {clearTimeout(this.presentationTimer);this.presentationTimer=setTimeout(()=>{if(!this.disposed&&!this.updating)this.onPresentation();},200);};
  private rememberPane = (event: Event) => {
    const column = event.composedPath().find(node => node instanceof Element && node.classList.contains('igv-column'));
    const index = Array.from(this.element.shadowRoot?.querySelectorAll('.igv-column') || []).indexOf(column as Element);
    if (index >= 0) this.pointerPane = index;
  };
  sync(state: BrowserState) {
    const frozen = structuredClone(state);
    return this.enqueue(async () => {
      const native = this.source.manifest.igv!, desired: Config[] = [];
      const order = (id: string) => frozen.tracks.indexOf(id) * 10;
      if (frozen.tracks.includes('reference')) desired.push({ id: 'osai-reference', type: 'sequence', name: 'Pinned GRCh38 reference', order: order('reference'), height: frozen.heights.reference || 50 });
      if (frozen.tracks.includes('genes')) desired.push({ id: 'osai-genes', name: 'GENCODE v24 · scoring genes', type: 'annotation', format: 'bed',
        url: await this.resource(native.genes), indexURL: await this.resource(native.genes.index), displayMode: 'EXPANDED', height: frozen.heights.genes || 100, order: order('genes'), color: '#456b8d' });
      const models: Model[] = frozen.compare ? [...new Set<Model>([frozen.model, 'r10', 'r13'])] : [frozen.model];
      for (const [i,event] of EVENTS.entries()) {
        if (!frozen.tracks.includes(event)) continue;
        const tracks = await Promise.all(models.map(async model => ({ type: 'wig', format: 'bigwig', name: `${model} ${event}`,
          url: await this.resource(native.tracks.find(t => t.model === model && t.event === event)!),
          color: model === frozen.model ? colors[i] : model === 'baseline' ? '#626d86' : '#da7396', windowFunction: 'max' })));
        desired.push({ id: `osai-${event}`, name: `${event} · ${models.join(' / ')} · max ALT/gene`, type: 'merged', tracks,
          height: frozen.heights[event] || 65, min: 0, max: 1, autoscale: frozen.autoscale, order: order(event), windowFunction: 'max' });
      }
      if (frozen.tracks.includes('coverage')) for (const [i,d] of native.coverageTracks.entries()) desired.push({ id: `osai-coverage-${d.model}`,
        name: `${d.model === 'input' ? 'Input' : d.model + ' predicted'} · REF-matching ALT count`, type: 'wig', format: 'bigwig', url: await this.resource(d), height: frozen.heights.coverage || 35, min: 0, max: 3,
        color: ['#7b8998','#6697b5','#159584'][i], order: order('coverage') + i, windowFunction: 'mean' });
      const ids = new Set(desired.map(t => t.id));
      for (const {track} of [...this.browser.trackViews]) if ((track.id?.startsWith('osai-') || track.type === 'sequence') && !ids.has(track.id)) {
        this.presentations.set(track.id,this.capture(track));this.browser.removeTrack(track);this.signatures.delete(track.id);
      }
      for (const derived of desired) {
        const id=String(derived.id),control=id.startsWith('osai-coverage-')?'coverage':id.replace('osai-','');
        let track=this.browser.findTracks('id',id)[0];
        const restored=this.restored.get(id);
        const saved=restored?await this.physical(restored) as Config:track?this.capture(track):this.presentations.get(id);
        // Data identities always come from the pinned manifest; only presentation is restored.
        const signature=JSON.stringify(this.logical({id,type:derived.type,url:derived.url,indexURL:derived.indexURL,tracks:derived.tracks,autoscale:derived.autoscale}));
        const config:Config={...derived,...saved,id,type:derived.type,format:derived.format,url:derived.url,indexURL:derived.indexURL};
        if(derived.tracks){config.tracks=derived.tracks;if(!saved?.name || String(saved.name).startsWith(`${control} ·`))config.name=derived.name;}
        if(this.lastState){
          if(JSON.stringify(frozen.tracks)!==JSON.stringify(this.lastState.tracks))config.order=derived.order;
          if(frozen.heights[control]!==this.lastState.heights[control])config.height=derived.height;
          if(frozen.autoscale!==this.lastState.autoscale && derived.tracks)config.autoscale=derived.autoscale;
        }
        if (!track || this.signatures.get(id)!==signature) {
          if(track)this.browser.removeTrack(track);
          track=await this.browser.loadTrack(config);track.id=id;track.config.id=id;this.signatures.set(id,signature);
        }
        track.order=Number(config.order);track.trackView.setTrackHeight(Number(config.height),true);
      }
      this.browser.reorderTracks();this.lastState=frozen;
    });
  }
  search(view: View | string) { return this.enqueue(() => this.browser.search(typeof view === 'string' ? view : formatView(view))); }
  loci() { return this.browser.currentLoci(); }
  private capture(track:Track):Config {
    return {...track.getState?.() || track.config,id:track.id,order:track.order,height:track.height,...(track.type==='sequence'?{reversed:track.reversed,frameTranslate:track.frameTranslate}:{})};
  }
  async session(): Promise<Config> {
    await this.pending;
    const tracks=this.browser.trackViews.filter(({track})=>!['ruler','ideogram'].includes(track.id)).map(({track})=>this.logical(this.capture(track)));
    return {locus:this.loci(),tracks,roi:this.logical(this.browser.roiManager.toJSON()||[]),showROIOverlays:this.browser.roiManager.showOverlays};
  }
  svg() { return this.browser.toSVG(); }
  roi(view: View) { return this.enqueue(() => this.browser.loadROI([{ name: 'Marked region', color: 'rgba(15,156,136,0.18)', features: [{ chr: view.chrom, start: view.start, end: view.end }] }])); }
  clearRois() { this.browser.clearROIs(); }
  async rois() { const result=[]; for(const set of this.browser.roiManager.roiSets) for(const features of Object.values(await set.getAllFeatures())) result.push(...features.map(f=>({...f,name:f.name||set.name}))); return result; }
  importRois(features: {chr:string;start:number;end:number;name?:string}[]) {return this.enqueue(()=>this.browser.loadROI([{name:'Imported regions',features,color:'rgba(15,156,136,0.18)'}]));}
  private loadTrackConfig(config: Config) { if(typeof config.id!=='string')config={...config,id:`user-${crypto.randomUUID()}`};return this.enqueue(async () => { const track=await this.browser.loadTrack(config);track.id=String(config.id);track.config.id=track.id; }).then(()=>this.onPresentation()); }
  loadTrack(url: string | File, index?: string | File) { return this.loadTrackConfig({ url, indexURL: index, name: typeof url === 'string' ? url.split('/').at(-1) : url.name }); }
  destroy() { this.disposed = true;clearTimeout(this.presentationTimer);cancelAnimationFrame(this.geometryFrame);this.resizeObserver?.disconnect();this.mutationObserver?.disconnect();delete this.element.dataset.ready;this.element.removeEventListener('pointerup',this.rememberPresentation); this.element.removeEventListener('pointerdown',this.rememberPane); if (this.browser) igv.removeBrowser(this.browser as unknown as Parameters<typeof igv.removeBrowser>[0]); this.files.clear(); }
}
