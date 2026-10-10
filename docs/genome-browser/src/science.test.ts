import { describe, expect, it } from 'vitest';
import { affectedSite, comparisonStatistics, csvCell, exactCsv, matchedAnnotations, predictionStatus } from './science';
import type { Variant } from './types';
const variant = (): Variant => ({ chrom: 'chr1', pos: 100, ref: 'A', alt: 'C', refMismatch: false, r13Accepted: true, occurrences: ['1:1'], models: { r10: [{ gene: 'G', ds: [0, 0.00001, 0.8, 0], dp: [-50, -4, 12, 0], occurrences: ['1:1'] }], r13: [{ gene: 'G', ds: [0, 0.00002, 0.9, 0], dp: [-2, -4, 12, 3], occurrences: ['1:1'] }], baseline: [] } });
describe('scientific semantics', () => {
  it('distinguishes absent, pending and measured zero', () => { const v = variant(); expect(predictionStatus(v, 'baseline')).toBe('prediction absent'); v.r13Accepted = false; v.models.r13 = []; expect(predictionStatus(v, 'r13')).toBe('preview pending'); v.models.r10[0].ds = [0, 0, 0, 0]; expect(predictionStatus(v, 'r10')).toBe('scored zero'); });
  it('uses genomic POS+DP on either strand and no site for zero', () => { const v = variant(); expect(affectedSite(v.pos, v.models.r10[0], 1)).toBe(97); expect(affectedSite(v.pos, v.models.r10[0], 0)).toBeNull(); });
  it('excludes conflicts, REF mismatches and partial previews from comparisons', () => { const v = variant(); expect(matchedAnnotations(v)).toHaveLength(1); v.refMismatch = true; expect(matchedAnnotations(v)).toHaveLength(0); v.refMismatch = false; v.r13Accepted = false; expect(matchedAnnotations(v)).toHaveLength(0); v.r13Accepted = true; v.models.r10.push({ ...v.models.r10[0], dp: [1, 2, 3, 4] }); expect(matchedAnnotations(v)).toHaveLength(0); });
  it('does not report correlation when all scores have no variance', () => expect(comparisonStatistics([variant()], 0).correlation).toBeNull());
  it('exports original precision, signed offsets and blank missing values', () => { const csv = exactCsv([variant()], 'snap'); expect(csv).toContain('0.00001'); expect(csv).toContain('-50,-4,12,0'); expect(csv).toContain('baseline,,prediction absent,,,,'); expect(csv).not.toContain('NaN'); });
  it('escapes commas, quotes and actual newlines in CSV', () => { expect(csvCell('a,"b"\nc')).toBe('"a,""b""\nc"'); });
});
