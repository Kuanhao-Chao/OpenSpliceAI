import { describe, expect, it } from 'vitest';
import { publishedCoverage } from './coverage';
const view = { chrom: 'chr1', start: 100, end: 200 };
describe('published demonstration coverage', () => {
  it('counts a clipped union, preserving gaps without double counting', () => {
    const intervals = [{ ...view, start: 50, end: 120 }, { ...view, start: 110, end: 140 }, { ...view, start: 160, end: 250 }];
    expect(publishedCoverage(view, intervals)).toEqual({ kind: 'partial', coveredBp: 80, totalBp: 100,
      intervals: [{ ...view, end: 140 }, { ...view, start: 160 }] });
    expect(intervals[0].start).toBe(50);
  });
  it('recognizes adjacent intervals as complete coverage', () => {
    expect(publishedCoverage(view, [{ ...view, end: 150 }, { ...view, start: 150 }]).kind).toBe('full');
  });
  it('does not include a different contig or a touching endpoint', () => {
    expect(publishedCoverage(view, [{ ...view, chrom: 'chr2' }, { ...view, start: 200, end: 300 }]).coveredBp).toBe(0);
    expect(publishedCoverage(view).kind).toBe('outside');
  });
  it('treats one published base as available without filling its neighbors', () => {
    const coverage = publishedCoverage(view, [{ ...view, start: 150, end: 151 }]);
    expect(coverage.kind).toBe('partial'); expect(coverage.coveredBp).toBe(1);
    expect(publishedCoverage({ ...view, start: 150, end: 151 }, coverage.intervals).kind).toBe('full');
  });
});
