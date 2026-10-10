import type { DataSource } from './data';
import type { Contig, SearchDescriptor, SearchHit, View } from './types';

export const IUPAC: Record<string, string> = { A: 'A', C: 'C', G: 'G', T: 'T', R: 'AG', Y: 'CT', W: 'AT', S: 'CG', K: 'GT', M: 'AC', B: 'CGT', D: 'AGT', H: 'ACT', V: 'ACG', N: 'ACGT' };
const COMPLEMENT: Record<string, string> = { A: 'T', C: 'G', G: 'C', T: 'A', R: 'Y', Y: 'R', W: 'W', S: 'S', K: 'M', M: 'K', B: 'V', V: 'B', D: 'H', H: 'D', N: 'N' };
export function motif(input: string) { const value = input.replace(/\s/g, '').toUpperCase(); if (!value || value.length > 256 || [...value].some(c => !IUPAC[c])) throw new Error('Enter 1–256 DNA IUPAC bases (A C G T R Y W S K M B D H V N)'); return value; }
export const reverseComplement = (sequence: string) => [...sequence].reverse().map(c => COMPLEMENT[c]).join('');
export function scan(sequence: string, query: string, offset = 0): number[] {
  const hits = [];
  for (let i = 0; i <= sequence.length - query.length; i++) { let matches = true; for (let j = 0; j < query.length; j++) if (!IUPAC[query[j]].includes(sequence[i + j])) { matches = false; break; } if (matches) hits.push(i + offset); }
  return hits;
}
interface Page { bwt: Uint8Array; rank: number[]; samples: DataView }
class FMQuery {
  private pending = new Map<number, Promise<Page>>();
  constructor(private source: DataSource, private descriptor: SearchDescriptor, private signal?: AbortSignal) {}
  private page(index: number): Promise<Page> {
    if (this.signal?.aborted) return Promise.reject(new DOMException('Aborted', 'AbortError'));
    let pending = this.pending.get(index);
    if (!pending) {
      pending = this.source.resources.page(this.descriptor.index, this.descriptor.path, index, this.signal).then(data => {
        const view = new DataView(data), n = view.getUint32(0, true), count = this.descriptor.alphabet.length;
        if (data.byteLength !== 4 + count * 4 + n * 5 || n > this.descriptor.pageBp) throw new Error('Corrupt sequence-index page');
        return { bwt: new Uint8Array(data, 4 + count * 4, n), rank: Array.from({ length: count }, (_, i) => view.getUint32(4 + i * 4, true)), samples: new DataView(data, 4 + count * 4 + n) };
      });
      this.pending.set(index, pending);
      // Promise cache only coalesces current work; byte LRU retains decoded pages.
      pending.finally(() => this.pending.delete(index)).catch(() => {});
    }
    return pending;
  }
  async rank(char: string, end: number): Promise<number> {
    if (!end) return 0;
    if (end === this.descriptor.rows) return this.descriptor.counts[char] || 0;
    const page = await this.page(Math.floor(end / this.descriptor.pageBp)), index = this.descriptor.alphabet.indexOf(char);
    if (index < 0) return 0;
    let rank = page.rank[index];
    for (let i = 0; i < end % this.descriptor.pageBp; i++) rank += Number(page.bwt[i] === char.charCodeAt(0));
    return rank;
  }
  async intervals(query: string): Promise<[number, number][]> {
    let ranges: [number, number][] = [[0, this.descriptor.rows]];
    for (const symbol of [...query].reverse()) {
      const next: [number, number][] = [];
      for (const [lo, hi] of ranges) {
        for (const char of IUPAC[symbol]) {
          if (!(char in this.descriptor.cumulative)) continue;
          const [l, h] = await Promise.all([this.rank(char, lo), this.rank(char, hi)]);
          if (l < h) next.push([this.descriptor.cumulative[char] + l, this.descriptor.cumulative[char] + h]);
        }
      }
      if (next.length > 4096) throw new Error('Motif is too ambiguous for indexed search; use a longer or more specific sequence');
      ranges = next; if (!ranges.length) break;
    }
    return ranges;
  }
  async locate(row: number): Promise<number> {
    for (let steps = 0; steps < this.descriptor.sampleRate; steps++) {
      const page = await this.page(Math.floor(row / this.descriptor.pageBp)), local = row % this.descriptor.pageBp;
      const sample = page.samples.getUint32(local * 4, true);
      if (sample !== 0xffffffff) return (sample + steps) % this.descriptor.rows;
      const char = String.fromCharCode(page.bwt[local]);
      row = this.descriptor.cumulative[char] + await this.rank(char, row);
    }
    throw new Error('Invalid sequence-index SA samples');
  }
}
export async function searchSequence(source: DataSource, input: string, scope: View | 'chromosome' | 'genome', current: View, signal: AbortSignal,
  progress: (text: string) => void, maxHits = 200): Promise<{ hits: SearchHit[]; strandHits: number; truncated: boolean }> {
  const query = motif(input), reversed = reverseComplement(query), queries = [{ value: query, strand: '+' as const }, { value: reversed, strand: '-' as const }];
  const hits: SearchHit[] = []; let strandHits = 0;
  if (typeof scope !== 'string') {
    const sequence = await source.sequence(scope, signal); if (sequence === null) throw new Error('Reference sequence is not yet published for this contig');
    for (const { value, strand } of queries) { const found = await source.resources.decoder.scan(sequence, value, scope.start, maxHits - hits.length, signal); strandHits += found.count; hits.push(...found.positions.map(pos => ({ chrom: scope.chrom, start: pos, end: pos + value.length, strand }))); }
  } else {
    const contigs = scope === 'chromosome' ? source.manifest.contigs.filter(c => c.name === current.chrom) : source.manifest.contigs;
    // Missing indexes fail clearly; partial reference data is never called a whole-genome search.
    if (contigs.some(c => !c.search)) throw new Error('This dataset has not published the sequence index for every requested contig. Choose visible region or gene scope.');
    for (const contig of contigs) {
      if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
      progress(`Searching ${contig.name}…`);
      const fm = new FMQuery(source, contig.search!, signal);
      for (const { value, strand } of queries) {
        const intervals = await fm.intervals(value);
        strandHits += intervals.reduce((n, [lo, hi]) => n + hi - lo, 0);
        for (const [lo, hi] of intervals) {
          for (let row = lo; row < hi && hits.length < maxHits; row += 6) {
            const n = Math.min(6, hi - row, maxHits - hits.length);
            const positions = await Promise.all(Array.from({ length: n }, (_, i) => fm.locate(row + i)));
            for (const pos of positions) if (pos + value.length <= contig.length) hits.push({ chrom: contig.name, start: pos, end: pos + value.length, strand });
          }
        }
      }
    }
  }
  return { hits: hits.sort((a, b) => a.chrom.localeCompare(b.chrom) || a.start - b.start || a.strand.localeCompare(b.strand)), strandHits, truncated: strandHits > hits.length };
}
