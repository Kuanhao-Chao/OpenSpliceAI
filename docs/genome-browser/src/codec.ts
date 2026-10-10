import type { Annotation, DecodedBlock, Variant } from './types';
const BASES = 'ACGTNRYWSKMBDHV.';
export async function inflate(data: ArrayBuffer): Promise<ArrayBuffer> {
  const stream = new Blob([data]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}
export function decodeBlock(data: ArrayBuffer): DecodedBlock {
  const view = new DataView(data);
  if (data.byteLength < 12 || new TextDecoder().decode(new Uint8Array(data, 0, 8)) !== 'OSGB1\0\0\0') throw new Error('Unsupported score block');
  const length = view.getUint32(8, true);
  if (length + 12 > data.byteLength) throw new Error('Truncated score header');
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(data, 12, length)));
  if (header.scale !== 100000 || !Array.isArray(header.genes)) throw new Error('Invalid score schema');
  const columns: DecodedBlock['columns'] = {};
  for (const column of header.columns) {
    const width = column.type === 'I' ? 4 : column.type === 'h' ? 2 : column.type === 'B' ? 1 : 0;
    if (!width || column.bytes !== column.count * width || column.offset < 0 || 12 + length + column.offset + column.bytes > data.byteLength) throw new Error('Truncated or invalid score column');
    const bytes = data.slice(12 + length + column.offset, 12 + length + column.offset + column.bytes);
    // Schema is little endian. Explicit reads also work on a big-endian host.
    const source = new DataView(bytes);
    const target = width === 4 ? new Uint32Array(column.count) : width === 2 ? new Int16Array(column.count) : new Uint8Array(column.count);
    for (let i = 0; i < column.count; i++) target[i] = width === 4 ? source.getUint32(i * 4, true) : width === 2 ? source.getInt16(i * 2, true) : source.getUint8(i);
    columns[column.name] = target;
  }
  for (const name of ['pos', 'ref', 'alt', 'chunk', 'ordinal', 'flags']) if (!columns[name] || columns[name].length !== header.rows) throw new Error('Invalid row column');
  return { chrom: header.chrom, rows: header.rows, genes: header.genes, columns, bytes: data.byteLength };
}
export function collectVariants(blocks: DecodedBlock[], start: number, end: number): Variant[] {
  const variants = new Map<string, Variant>();
  for (const block of blocks) {
    const c = block.columns;
    const rowVariants: (Variant | undefined)[] = new Array(block.rows);
    for (let i = 0; i < block.rows; i++) {
      const pos = c.pos[i]; if (pos < start || pos >= end) continue;
      const ref = BASES[c.ref[i]], alt = BASES[c.alt[i]];
      if (!ref || !alt) throw new Error('Invalid allele code');
      const key = `${pos}:${ref}>${alt}`;
      let v = variants.get(key);
      if (!v) { v = { chrom: block.chrom, pos, ref, alt, refMismatch: !!(c.flags[i] & 1), r13Accepted: true, occurrences: [], models: { r10: [], r13: [], baseline: [] } }; variants.set(key, v); }
      v.refMismatch ||= !!(c.flags[i] & 1); v.r13Accepted &&= !!(c.flags[i] & 2);
      const occurrence = `${c.chunk[i]}:${c.ordinal[i]}`;
      if (!v.occurrences.includes(occurrence)) v.occurrences.push(occurrence);
      rowVariants[i] = v;
    }
    for (const model of ['r10', 'r13', 'baseline'] as const) {
      const rows = c[`${model}.row`]; if (!rows) throw new Error('Missing annotation columns');
      const signatures = new Map<Variant, Map<string, Annotation>>();
      for (let j = 0; j < rows.length; j++) {
        const row = rows[j]; if (row >= block.rows) throw new Error('Invalid annotation row');
        const v = rowVariants[row]; if (!v) continue;
        const gene = block.genes[c[`${model}.gene`][j]];
        if (gene === undefined) throw new Error('Invalid gene code');
        const ds = ['AG', 'AL', 'DG', 'DL'].map(e => c[`${model}.DS_${e}`][j] / 100000);
        const dp = ['AG', 'AL', 'DG', 'DL'].map(e => c[`${model}.DP_${e}`][j]);
        const signature = `${gene}|${ds.join('|')}|${dp.join('|')}`;
        let dict = signatures.get(v);
        if (!dict) { dict = new Map(v.models[model].map(a => [`${a.gene}|${a.ds.join('|')}|${a.dp.join('|')}`, a])); signatures.set(v, dict); }
        let a = dict.get(signature);
        if (!a) { a = { gene, ds, dp, occurrences: [] }; v.models[model].push(a); dict.set(signature, a); }
        const occurrence = `${c.chunk[row]}:${c.ordinal[row]}`;
        if (!a.occurrences.includes(occurrence)) a.occurrences.push(occurrence);
      }
    }
  }
  return [...variants.values()].sort((a, b) => a.pos - b.pos || a.ref.localeCompare(b.ref) || a.alt.localeCompare(b.alt));
}
