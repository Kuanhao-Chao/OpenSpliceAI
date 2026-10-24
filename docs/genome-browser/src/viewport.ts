import type { View } from './types';

/** CSS screen coordinates from IGV's actual viewport and translated backing canvas. */
export interface PaneGeometry extends View {
  id: string;
  index: number;
  left: number;
  width: number;
  bpPerPixel: number;
  originBase: number;
  originPixel: number;
  ready: boolean;
}

export const baseX = (pane: PaneGeometry, position: number) =>
  pane.left + pane.originPixel + (position - pane.originBase) / pane.bpPerPixel;

export const genomicPosition = (pane: PaneGeometry, screenX: number) =>
  pane.originBase + (screenX - pane.left - pane.originPixel) * pane.bpPerPixel;

export function baseAt(pane: PaneGeometry, screenX: number): number | null {
  if (!pane.ready || screenX < pane.left || screenX >= pane.left + pane.width) return null;
  const position = genomicPosition(pane, screenX);
  // Cancel floating-point noise at integer boundaries without rounding a real fractional base.
  return Math.floor(position + Number.EPSILON * Math.max(1, Math.abs(position)) * 8);
}

export function requestedView(pane: PaneGeometry, contigLength: number): View {
  // IGV rounds canvas translation to pixels. Include the resulting partial edge bases too.
  return { chrom: pane.chrom,
    start: Math.max(0, Math.floor(Math.min(pane.start, genomicPosition(pane, pane.left)))),
    end: Math.min(contigLength, Math.ceil(Math.max(pane.end, genomicPosition(pane, pane.left + pane.width)))) };
}

export const covers = (outer: View, inner: View) =>
  outer.chrom === inner.chrom && outer.start <= inner.start && outer.end >= inner.end;
