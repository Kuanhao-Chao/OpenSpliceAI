import type { Annotation, BrowserState, Model, Variant } from './types';

export function predictionStatus(variant: Variant, model: Model): string {
  if (variant.models[model].length) {
    const result = variant.models[model].every(a => a.ds.every(v => v === 0)) ? 'scored zero' : 'scored';
    return model === 'r13' && !variant.r13Accepted ? `${result} (partial preview)` : result;
  }
  return model === 'r13' && !variant.r13Accepted ? 'preview pending' : 'prediction absent';
}
export function value(variant: Variant, model: Model, channel: number, gene = ''): number | null {
  const entries = variant.models[model].filter(a => !gene || a.gene === gene);
  return entries.length ? Math.max(...entries.map(a => a.ds[channel])) : null;
}
export function visibleVariants(variants: Variant[], state: BrowserState): Variant[] {
  return variants.filter(v => (state.alt === '*' || state.alt === v.alt) && (!state.gene || v.models[state.model].some(a => a.gene === state.gene)) &&
    (state.threshold <= 0 || v.models[state.model].some(a => Math.max(...a.ds) >= state.threshold)));
}
export function matchedAnnotations(variant: Variant): { gene: string; r10: Annotation; r13: Annotation }[] {
  if (variant.refMismatch || !variant.r13Accepted) return [];
  const genes = new Set(variant.models.r10.map(a => a.gene));
  const matches = [];
  for (const gene of genes) {
    const left = variant.models.r10.filter(a => a.gene === gene), right = variant.models.r13.filter(a => a.gene === gene);
    // Conflicting source annotations are inspectable but cannot form a unique comparison.
    if (left.length === 1 && right.length === 1) matches.push({ gene, r10: left[0], r13: right[0] });
  }
  return matches;
}
export function comparisonStatistics(variants: Variant[], channel: number) {
  const points = variants.flatMap(v => matchedAnnotations(v).map(m => ({ x: m.r10.ds[channel], y: m.r13.ds[channel], variant: v, gene: m.gene })));
  const n = points.length;
  if (!n) return { n, meanDifference: null, correlation: null, points };
  const mx = points.reduce((s, p) => s + p.x, 0) / n, my = points.reduce((s, p) => s + p.y, 0) / n;
  let covariance = 0, vx = 0, vy = 0;
  for (const p of points) { covariance += (p.x - mx) * (p.y - my); vx += (p.x - mx) ** 2; vy += (p.y - my) ** 2; }
  return { n, meanDifference: my - mx, correlation: vx > 0 && vy > 0 ? covariance / Math.sqrt(vx * vy) : null, points };
}
export function affectedSite(pos0: number, annotation: Annotation, channel: number): number | null {
  return annotation.ds[channel] > 0 ? pos0 + 1 + annotation.dp[channel] : null;
}
export const csvCell = (value: unknown) => { const s = value === null || value === undefined ? '' : String(value); return /[",\r\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s; };
export function exactCsv(variants: Variant[], snapshot: string): string {
  const lines: unknown[][] = [['snapshot', 'chrom', 'pos_1based', 'start_0based', 'end_0based', 'ref', 'alt', 'model', 'gene', 'status',
    'DS_AG', 'DS_AL', 'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL', 'site_AG_1based', 'site_AL_1based', 'site_DG_1based', 'site_DL_1based', 'ref_mismatch', 'source_occurrences']];
  for (const v of variants) for (const model of ['r10', 'r13', 'baseline'] as const) {
    const entries = v.models[model];
    if (!entries.length) lines.push([snapshot, v.chrom, v.pos + 1, v.pos, v.pos + 1, v.ref, v.alt, model, '', predictionStatus(v, model), ...Array(12).fill(null), v.refMismatch, v.occurrences.join(';')]);
    for (const a of entries) lines.push([snapshot, v.chrom, v.pos + 1, v.pos, v.pos + 1, v.ref, v.alt, model, a.gene, predictionStatus(v, model), ...a.ds.map(s => s.toFixed(model === 'baseline' ? 2 : 5)), ...a.dp, ...[0, 1, 2, 3].map(c => affectedSite(v.pos, a, c)), v.refMismatch, a.occurrences.join(';')]);
  }
  return lines.map(line => line.map(csvCell).join(',')).join('\r\n') + '\r\n';
}
