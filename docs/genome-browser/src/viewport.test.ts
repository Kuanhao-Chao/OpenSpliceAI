import { describe, expect, it } from 'vitest';
import { baseAt, baseX, covers, requestedView, type PaneGeometry } from './viewport';

const pane: PaneGeometry = { id: 'one', index: 0, chrom: 'chr1', start: 100.4, end: 200.4,
  left: 293, width: 800, bpPerPixel: .125, originBase: .375, originPixel: -800, ready: true };

describe('native viewport coordinates', () => {
  it('uses the rendered fractional canvas origin and its translation', () => {
    expect(baseX(pane, 101)).toBe(298);
    expect(baseX(pane, 102) - baseX(pane, 101)).toBe(8);
    const second = { ...pane, left: 672.5, width: 374 };
    expect(baseX(second, 101)).toBe(677.5);
  });
  it('selects the containing base and excludes gutters and right edges', () => {
    expect(baseAt(pane, 293)).toBe(100);
    expect(baseAt(pane, 298)).toBe(101);
    expect(baseAt(pane, 305.99)).toBe(101);
    expect(baseAt(pane, 306)).toBe(102);
    expect(baseAt(pane, 292.99)).toBeNull();
    expect(baseAt(pane, 1093)).toBeNull();
    expect(baseAt({ ...pane, ready: false }, 298)).toBeNull();
  });
  it('requests whole edge bases without rounding the drawing frame', () => {
    expect(requestedView(pane, 1000)).toEqual({ chrom: 'chr1', start: 100, end: 201 });
    expect(requestedView(pane, 200).end).toBe(200);
    expect(covers({ chrom: 'chr1', start: 100, end: 201 }, requestedView(pane, 1000))).toBe(true);
    expect(covers({ chrom: 'chr2', start: 100, end: 201 }, requestedView(pane, 1000))).toBe(false);
  });
  it('round-trips human-genome coordinates at fractional scales', () => {
    const large = { ...pane, start: 117504123.31, end: 117504221.5, originBase: 117504000.17, originPixel: -734, bpPerPixel: .1675 };
    for (const position of [117504125, 117504150, 117504200]) {
      expect(baseAt(large, baseX(large, position + .5))).toBe(position);
      expect(baseAt(large, baseX(large, position))).toBe(position);
    }
  });
});
