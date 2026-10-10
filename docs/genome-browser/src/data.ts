import { collectVariants } from './codec';
import { Resources } from './resources';
import { EXACT_BP } from './state';
import type { Gene, LoadedView, Manifest, RegionIndex, Summary, View } from './types';

export class DataSource {
  readonly resources: Resources;
  constructor(readonly manifest: Manifest, readonly manifestUrl: string) {
    if (manifest.format !== 'OSGB1' || manifest.scoreScale !== 100000) throw new Error('Unsupported dataset schema');
    this.resources = new Resources(manifest.baseUrl ? new URL(manifest.baseUrl, manifestUrl).href : new URL('.', manifestUrl).href);
  }
  genes(signal?: AbortSignal) { return this.resources.json<Gene[]>(this.manifest.genes.path, this.manifest.genes, signal); }
  async overview(signal?: AbortSignal) {
    const descriptor = this.manifest.overview;
    if (!descriptor) return;
    const data = await this.resources.json<Record<string, Summary[]>>(descriptor.path, descriptor, signal);
    for (const contig of this.manifest.contigs) contig.overview = data[contig.name] || [];
  }
  async sequence(view: View, signal?: AbortSignal): Promise<string | null> {
    const contig = this.manifest.contigs.find(c => c.name === view.chrom)!;
    const ref = contig.reference; if (!ref) return null;
    if (ref.ranges && !ref.ranges.some(([a, b]) => a <= view.start && b >= view.end)) return null;
    const first = Math.floor(view.start / ref.blockBp), last = Math.floor((view.end - 1) / ref.blockBp);
    if (last - first > 1024) throw new Error('Sequence range is too large; use indexed whole-genome motif search');
    const pages: string[] = [];
    // Small batches bound temporary buffers even for long gene searches.
    for (let start = first; start <= last; start += 6) {
      const result = await Promise.all(Array.from({ length: Math.min(6, last - start + 1) }, (_, i) =>
        this.resources.page(ref.index, ref.path, start + i, signal).then(p => new TextDecoder().decode(p))));
      pages.push(...result);
    }
    return pages.join('').slice(view.start - first * ref.blockBp, view.end - first * ref.blockBp);
  }
  async load(view: View, signal?: AbortSignal): Promise<LoadedView> {
    const contig = this.manifest.contigs.find(c => c.name === view.chrom);
    if (!contig) throw new Error('Unknown contig');
    const exact = view.end - view.start <= EXACT_BP;
    const features = contig.genes ? this.resources.json<Gene[]>(contig.genes.path, contig.genes, signal) : Promise.resolve([]);
    if (view.end - view.start > this.manifest.indexBp * 2) return { view, exact: false, variants: [], summaries: contig.overview.filter(s => s.end > view.start && s.start < view.end), sequence: null, genes: await features };
    const indexes = Array.from({ length: Math.floor((view.end - 1) / this.manifest.indexBp) - Math.floor(view.start / this.manifest.indexBp) + 1 }, (_, i) => Math.floor(view.start / this.manifest.indexBp) + i)
      .map(i => contig.indexes[String(i)]).filter(Boolean);
    const [data, featureGenes] = await Promise.all([Promise.all(indexes.map(d => this.resources.json<RegionIndex>(d.path, d, signal))), features]);
    const summaries = data.flatMap(i => i.summaries.filter(s => s.end > view.start && s.start < view.end));
    if (!exact) return { view, exact, variants: [], summaries, sequence: null, genes: featureGenes };
    const descriptors = data.flatMap(i => i.blocks.filter(b => b.start < view.end && b.end > view.start));
    const mobile = matchMedia('(max-width: 700px)').matches;
    if (descriptors.reduce((n, d) => n + d.rawBytes, 0) > (mobile ? 8 : 16) * 1048576) throw new Error('Dense locus exceeds the exact-view memory budget; zoom in');
    const blocks = await Promise.all(descriptors.map(d => this.resources.block(d, signal)));
    let occurrences = 0;
    for (const block of blocks) for (const pos of block.columns.pos) if (pos >= view.start && pos < view.end) occurrences++;
    if (occurrences > (mobile ? 30000 : 100000)) throw new Error('Dense locus exceeds the visible-allele memory budget; zoom in');
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
    const sequence = view.end - view.start <= 4096 ? await this.sequence(view, signal) : null;
    return { view, exact, variants: collectVariants(blocks, view.start, view.end), summaries, sequence, genes: featureGenes };
  }
  destroy() { this.resources.destroy(); }
}

/** Overview/summary counts are occurrences and annotation entries, not distinct variants. */
export function summaryTotals(summaries: Summary[]) {
  return summaries.reduce((out, s) => ({ rows: out.rows + s.rows, accepted: out.accepted + s.acceptedR13,
    mismatch: out.mismatch + s.refMismatch }), { rows: 0, accepted: 0, mismatch: 0 });
}
