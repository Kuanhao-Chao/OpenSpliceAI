import type { View } from './types';

/** Published intervals describe availability, independently of input alleles. */
export function publishedCoverage(view: View, published: readonly View[] = []) {
  const intervals: View[] = [];
  const overlapping = published.filter(r => r.chrom === view.chrom && r.start < view.end && r.end > view.start)
    .map(r => ({ chrom: r.chrom, start: Math.max(r.start, view.start), end: Math.min(r.end, view.end) }))
    .sort((a, b) => a.start - b.start || a.end - b.end);
  for (const next of overlapping) {
    const last = intervals.at(-1);
    if (last && next.start <= last.end) last.end = Math.max(last.end, next.end);
    else intervals.push({ ...next });
  }
  const coveredBp = intervals.reduce((total, r) => total + r.end - r.start, 0);
  const totalBp = view.end - view.start;
  const kind = coveredBp === 0 ? 'outside' : coveredBp === totalBp ? 'full' : 'partial';
  return { kind, coveredBp, totalBp, intervals } as const;
}
