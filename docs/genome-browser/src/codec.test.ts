import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { collectVariants, decodeBlock } from './codec';
import type { Manifest, RegionIndex } from './types';
const base = new URL('../public/data/review/', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('manifest.json', base), 'utf8')) as Manifest;
const blocks = (chrom: string, pos: number) => {
  const contig = manifest.contigs.find(c => c.name === chrom)!;
  const descriptor = contig.indexes[String(Math.floor(pos / manifest.indexBp))];
  const index = JSON.parse(gunzipSync(readFileSync(new URL(descriptor.path, base))).toString()) as RegionIndex;
  return index.blocks.filter(b => b.start <= pos && b.end > pos).map(b => { const pack = readFileSync(new URL(b.path, base)); const raw = gunzipSync(pack.subarray(b.offset, b.offset + b.bytes)); return decodeBlock(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer); });
};
describe('real VCF → Python columns → TypeScript variants', () => {
  it('preserves the original five-decimal r10 values and offsets', () => { const v = collectVariants(blocks('chr1', 69090), 69090, 69091).find(v => v.alt === 'G')!; expect(v.models.r10[0].gene).toBe('OR4F5'); expect(v.models.r10[0].ds).toEqual([0.00009, 0, 0.00001, 0]); expect(v.models.r10[0].dp).toEqual([42, 2, 42, 2]); expect(v.models.r13[0].ds).toEqual([0, 0, 0, 0]); });
  it('retains alternate-contig REF mismatches and conflicting baseline annotations', () => { const rows = collectVariants(blocks('chr2_KI270773v1_alt', 18693), 18693, 18694); const incorrect = rows.find(v => v.ref === 'C' && v.alt === 'A')!, correct = rows.find(v => v.ref === 'T' && v.alt === 'A')!; expect(incorrect.refMismatch).toBe(true); expect(correct.refMismatch).toBe(false); expect(incorrect.models.r10).toHaveLength(0); expect(incorrect.models.baseline.filter(a => a.gene === 'SNTG2').length).toBeGreaterThan(1); expect(incorrect.models.baseline.map(a => a.dp)).toContainEqual([27, 28, -43, 27]); expect(incorrect.models.baseline.map(a => a.dp)).toContainEqual([-36, 28, 27, 46]); });
  it('detects corruption and does not synthesize zero-filled columns', () => { expect(() => decodeBlock(new ArrayBuffer(8))).toThrow(); const b = blocks('chr1', 69090)[0]; expect(b.columns['r13.DS_AG'].length).toBeGreaterThan(0); });
});
