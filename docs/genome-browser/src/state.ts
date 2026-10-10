import { MODELS, type BrowserState, type Contig, type View } from './types';

export const THEMES = ['light', 'dark', 'nord', 'monokai', 'cyberdeck', 'parchment'] as const;
export const TRACKS = ['genes', 'reference', 'coverage', 'AG', 'AL', 'DG', 'DL', 'heatmap', 'difference'] as const;
export const EXACT_BP = 32768;
export const DEFAULT_STATE: BrowserState = {
  chrom: 'chr1', start: 69040, end: 70060, snapshot: '', model: 'r10', compare: true,
  theme: 'light', density: 'comfortable', autoscale: false, tracks: [...TRACKS], heights: {},
  threshold: 0, gene: '', roi: null, selected: '', alt: '*',
};
export function clampView(view: View, contigs: Contig[], minimumSpan = 20): View {
  const contig = contigs.find(c => c.name === view.chrom);
  if (!contig || !Number.isFinite(view.start) || !Number.isFinite(view.end)) throw new Error('Unknown contig or invalid coordinates');
  const span = Math.min(contig.length, Math.max(Math.min(minimumSpan, contig.length), Math.round(view.end - view.start)));
  const start = Math.max(0, Math.min(contig.length - span, Math.round(view.start)));
  return { chrom: contig.name, start, end: start + span };
}
export function parseLocus(input: string, contigs: Contig[]): View | null {
  const match = /^([^:\s]+):([\d,]+)(?:-([\d,]+))?$/.exec(input.trim());
  if (!match) return null;
  const chrom = contigs.find(c => c.name === match[1] || c.name === `chr${match[1]}`)?.name;
  const start = Number(match[2].replaceAll(',', ''));
  const end = match[3] ? Number(match[3].replaceAll(',', '')) : start;
  if (!chrom || !Number.isSafeInteger(start) || start < 1 || !Number.isSafeInteger(end) || end < start) throw new Error('Use a valid 1-based inclusive locus, for example chr7:117504153-117504464');
  return clampView({ chrom, start: start - 1, end: match[3] ? end : start + 100 }, contigs);
}
export const formatView = (v: View) => `${v.chrom}:${(v.start + 1).toLocaleString('en-US')}-${v.end.toLocaleString('en-US')}`;
export const variantKey = (v: { chrom: string; pos: number; ref: string; alt: string }) => `${v.chrom}:${v.pos + 1}:${v.ref}>${v.alt}`;
export function orderedContigs(contigs: Contig[]): Contig[] {
  const primary = new Map([...Array.from({ length: 22 }, (_, i) => `chr${i + 1}`), 'chrX', 'chrY', 'chrM'].map((name, i) => [name, i]));
  return [...contigs].sort((a, b) => (primary.get(a.name) ?? 100) - (primary.get(b.name) ?? 100) || a.name.localeCompare(b.name));
}
export function serializeState(state: BrowserState): string {
  const p = new URLSearchParams({ v: '1', dataset: state.snapshot, chr: state.chrom, start: String(state.start), end: String(state.end),
    model: state.model, compare: state.compare ? '1' : '0', theme: state.theme, density: state.density, autoscale: state.autoscale ? '1' : '0',
    tracks: state.tracks.join(','), heights: JSON.stringify(state.heights), threshold: String(state.threshold), gene: state.gene, selected: state.selected, alt: state.alt });
  if (state.roi) p.set('roi', JSON.stringify(state.roi));
  return p.toString();
}
export function snapshotLink(pageUrl: string, manifestUrl: string, state: BrowserState): string {
  const url = new URL(pageUrl);
  url.searchParams.set('manifest', new URL(manifestUrl, url).href);
  url.hash = serializeState(state);
  return url.href;
}
export function restoreState(hash: string, fallback: BrowserState, contigs: Contig[]): BrowserState {
  const p = new URLSearchParams(hash.replace(/^#/, ''));
  if (!p.has('v')) return { ...fallback };
  const state = { ...fallback, ...clampView({ chrom: p.get('chr') || fallback.chrom, start: Number(p.get('start')), end: Number(p.get('end')) }, contigs) };
  state.snapshot = p.get('dataset') || fallback.snapshot;
  const model = p.get('model'); if (MODELS.includes(model as never)) state.model = model as BrowserState['model'];
  const theme = p.get('theme'); if (THEMES.includes(theme as never)) state.theme = theme as BrowserState['theme'];
  const density = p.get('density'); if (['comfortable', 'compact', 'dense'].includes(density || '')) state.density = density as BrowserState['density'];
  state.compare = p.get('compare') === '1'; state.autoscale = p.get('autoscale') === '1';
  if (p.has('tracks')) state.tracks = (p.get('tracks') || '').split(',').filter(t => TRACKS.includes(t as never));
  state.threshold = Math.max(0, Math.min(1, Number(p.get('threshold')) || 0));
  state.gene = p.get('gene') || ''; state.selected = p.get('selected') || ''; state.alt = p.get('alt') || '*';
  if (!['*', 'A', 'C', 'G', 'T', 'N'].includes(state.alt)) state.alt = '*';
  try {
    const heights = JSON.parse(p.get('heights') || '{}');
    state.heights = Object.fromEntries(Object.entries(heights).filter(([k, v]) => TRACKS.includes(k as never) && typeof v === 'number' && v >= 32 && v <= 240)) as Record<string, number>;
    if (p.has('roi')) state.roi = clampView(JSON.parse(p.get('roi')!), contigs, 1);
  } catch { state.heights = {}; state.roi = null; }
  return state;
}
export class History {
  private entries: BrowserState[] = []; private cursor = -1;
  clear() { this.entries = []; this.cursor = -1; }
  push(s: BrowserState) { if (this.cursor >= 0 && serializeState(this.entries[this.cursor]) === serializeState(s)) return; this.entries = this.entries.slice(0, this.cursor + 1); this.entries.push(structuredClone(s)); if (this.entries.length > 100) this.entries.shift(); this.cursor = this.entries.length - 1; }
  back(): BrowserState | null { return this.cursor > 0 ? structuredClone(this.entries[--this.cursor]) : null; }
  forward(): BrowserState | null { return this.cursor + 1 < this.entries.length ? structuredClone(this.entries[++this.cursor]) : null; }
}
