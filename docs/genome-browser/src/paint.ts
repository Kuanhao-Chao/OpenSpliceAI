import { formatView } from './state';
import type { BrowserState, ExactPlot, View } from './types';
import { baseX, type PaneGeometry } from './viewport';

export interface Palette { bg: string; ink: string; muted: string; line: string; accent: string; compare: string; missing: string }
export interface Lane { id: string; y: number; height: number }
export interface DetailPane {
  geometry: PaneGeometry;
  view?: View;
  plot?: ExactPlot;
  exact?: boolean;
  pending?: boolean;
  error?: string;
}
export function layout(state: BrowserState): Lane[] {
  let y = 64;
  return state.tracks.map(id => {
    const base = id === 'genes' ? 84 : id === 'reference' ? 36 : id === 'heatmap' ? 92 : 58;
    const height = state.heights[id] || Math.max(32, Math.round(base * (state.density === 'dense' ? .65 : state.density === 'compact' ? .82 : 1)));
    const lane = { id, y, height }; y += height + 8; return lane;
  });
}
export function detailHeight(state: BrowserState) {
  const lanes = layout({ ...state, tracks: state.tracks.filter(t => t === 'heatmap' || t === 'difference') });
  return (lanes.at(-1)?.y || 64) + (lanes.at(-1)?.height || 0) + 32;
}

/** The only x transform is the one used by the native IGV canvas, including its pixel shift. */
export function draw(ctx: CanvasRenderingContext2D, width: number, state: BrowserState, panes: DetailPane[], activePane: number, palette: Palette, screenLeft: number, pixelRatio: number) {
  const lanes = layout({ ...state, tracks: state.tracks.filter(t => t === 'heatmap' || t === 'difference') });
  const height = detailHeight(state), minimumPixel = 1 / pixelRatio;
  ctx.fillStyle = palette.bg; ctx.fillRect(0, 0, width, height);
  ctx.textBaseline = 'middle'; ctx.font = '11px Inter, system-ui, sans-serif';
  ctx.fillStyle = palette.muted;
  ctx.fillText(`${state.model} · exact ALT scores · missing ≠ zero`, 8, 14);
  if (!panes.length) { ctx.fillText('Choose a chromosome region for exact scores.', 8, 42); return; }
  for (const pane of panes) {
    const g = pane.geometry, left = g.left - screenLeft, right = left + g.width;
    const x = (pos: number) => baseX(g, pos) - screenLeft;
    ctx.save(); ctx.beginPath(); ctx.rect(left, 24, g.width, height - 46); ctx.clip();
    ctx.fillStyle = g.index === activePane ? palette.accent : palette.muted;
    ctx.font = '10px Inter, system-ui, sans-serif';
    ctx.fillText(`${g.index === activePane ? '● ' : ''}${formatView({ chrom: g.chrom, start: Math.floor(g.start), end: Math.ceil(g.end) })}`, left + 3, 34);
    ctx.fillStyle = palette.line; ctx.fillRect(left, 47, g.width, minimumPixel);
    const status = !g.ready ? 'Aligning to IGV…' : pane.error ? 'Data failed · Retry data' : pane.pending ? 'Loading newly visible bases…' : pane.exact === false ? 'Zoom in for exact scores' : '';
    if (status) { ctx.fillStyle = pane.error ? palette.compare : palette.muted; ctx.fillText(status, left + 3, 56); }
    for (const { id, y, height: h } of lanes) {
      const top = y + 20, bottom = y + h;
      ctx.fillStyle = palette.muted; ctx.font = '10px Inter, system-ui, sans-serif';
      ctx.fillText(id === 'heatmap' ? `ALT max DS · ${state.model}` : 'Matched max DS · r13 − r10', left + 3, y + 9);
      ctx.strokeStyle = palette.line; ctx.beginPath(); ctx.moveTo(left, bottom); ctx.lineTo(right, bottom); ctx.stroke();
      ctx.save(); ctx.beginPath(); ctx.rect(left, top, g.width, bottom - top); ctx.clip();
      if (id === 'difference') {
        const middle = (top + bottom) / 2;
        ctx.strokeStyle = palette.line; ctx.beginPath(); ctx.moveTo(left, middle); ctx.lineTo(right, middle); ctx.stroke();
      }
      if (g.ready && pane.view && pane.exact && pane.plot) {
        // Reproject cached absolute positions while panning; newly exposed bases stay blank.
        ctx.beginPath(); ctx.rect(x(pane.view.start), top, x(pane.view.end) - x(pane.view.start), bottom - top); ctx.clip();
        const plot = pane.plot, cellWidth = Math.max(minimumPixel, 1 / g.bpPerPixel);
        if (id === 'heatmap') {
          const bandHeight = (bottom - top) / 4;
          const pixels = new Map<string, { x: number; band: number; score: number; mismatch: boolean }>();
          for (let i = 0; i < plot.pos.length; i++) {
            const px = x(plot.pos[i]); if (px + cellWidth <= left || px >= right) continue;
            const key = `${Math.floor(px * pixelRatio)}:${plot.alt[i]}`, old = pixels.get(key);
            if (!old || old.score < plot.score[i]) pixels.set(key, { x: px, band: plot.alt[i], score: plot.score[i], mismatch: !!plot.mismatch[i] });
          }
          for (const p of pixels.values()) {
            ctx.fillStyle = p.mismatch ? palette.compare : palette.accent; ctx.globalAlpha = .12 + .88 * p.score;
            ctx.fillRect(p.x, top + p.band * bandHeight, cellWidth, Math.max(minimumPixel, bandHeight - 1));
          }
          ctx.globalAlpha = 1;
        } else {
          const middle = (top + bottom) / 2, amplitude = Math.max(1, (bottom - top) / 2 - 2);
          for (let i = 0; i < plot.difference.length; i++) {
            const px = x(plot.differencePos[i]); if (px + cellWidth <= left || px >= right) continue;
            const delta = plot.difference[i]; ctx.fillStyle = delta >= 0 ? palette.accent : palette.compare;
            ctx.fillRect(px, middle - Math.max(0, delta) * amplitude, cellWidth, Math.max(minimumPixel, Math.abs(delta) * amplitude));
          }
        }
      }
      if (g.ready && state.roi?.chrom === g.chrom) {
        ctx.fillStyle = palette.accent; ctx.globalAlpha = .10;
        ctx.fillRect(x(state.roi.start), top, x(state.roi.end) - x(state.roi.start), bottom - top); ctx.globalAlpha = 1;
      }
      const selected = /^([^:]+):(\d+):/.exec(state.selected);
      if (g.ready && selected?.[1] === g.chrom) {
        const px = x(Number(selected[2]) - .5);
        ctx.strokeStyle = palette.ink; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(px, top); ctx.lineTo(px, bottom); ctx.stroke(); ctx.setLineDash([]);
      }
      ctx.restore();
    }
    ctx.restore();
  }
  // Row labels occupy IGV's axis gutter, never an extra inset in the genomic columns.
  const gutter = Math.max(0, panes[0].geometry.left - screenLeft);
  ctx.save(); ctx.beginPath(); ctx.rect(0, 0, gutter, height); ctx.clip();
  ctx.font = '10px Inter, system-ui, sans-serif'; ctx.fillStyle = palette.muted;
  for (const lane of lanes) if (lane.id === 'heatmap') {
    const top = lane.y + 20, band = (lane.height - 20) / 4;
    for (const [i, alt] of ['A', 'C', 'G', 'T'].entries()) ctx.fillText(alt, Math.max(4, gutter - 14), top + (i + .5) * band);
  }
  ctx.restore();
  ctx.fillStyle = palette.muted; ctx.font = '10px Inter, system-ui, sans-serif';
  ctx.fillText('Columns follow IGV · select a column to inspect its base', 8, height - 12);
}
