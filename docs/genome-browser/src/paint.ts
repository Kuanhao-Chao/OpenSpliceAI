import { formatView, variantKey } from './state';
import { matchedAnnotations, value, visibleVariants } from './science';
import type { BrowserState, Gene, LoadedView, Model, Variant } from './types';

export interface Palette { bg: string; ink: string; muted: string; line: string; accent: string; compare: string; missing: string }
export interface Lane { id: string; y: number; height: number }
export const CHANNEL_COLORS = ['#208978', '#66a68d', '#7662ad', '#b27aaf'];
export function layout(state: BrowserState): Lane[] {
  let y = 64;
  const lanes = state.tracks.map(id => { const base = id === 'genes' ? 84 : id === 'reference' ? 36 : id === 'heatmap' ? 92 : 58;
    const height = state.heights[id] || Math.max(32, Math.round(base * (state.density === 'dense' ? 0.65 : state.density === 'compact' ? 0.82 : 1)));
    const lane = { id, y, height }; y += height + 8; return lane; });
  return lanes;
}
export const margin = (width: number) => width < 600 ? 94 : 144;
export function draw(ctx: CanvasRenderingContext2D, width: number, state: BrowserState, loaded: LoadedView, genes: Gene[], palette: Palette): number {
  const lanes = layout(state), height = (lanes.at(-1)?.y || 64) + (lanes.at(-1)?.height || 0) + 42;
  const left = margin(width), right = width - 14, span = state.end - state.start;
  const x = (p: number) => left + (p - state.start) / span * (right - left);
  const visible = visibleVariants(loaded.variants, state);
  ctx.fillStyle = palette.bg; ctx.fillRect(0, 0, width, height);
  ctx.font = '12px Inter, system-ui, sans-serif'; ctx.textBaseline = 'middle'; ctx.fillStyle = palette.ink;
  ctx.fillText(formatView(state), 12, 16); ctx.fillStyle = palette.muted;
  ctx.fillText(`${loaded.exact ? 'Exact variants' : 'Summary maxima'} · ${state.model} · ${state.snapshot}`, 12, 35);
  const targetTicks = Math.max(2, Math.floor((right - left) / 100));
  const power = 10 ** Math.floor(Math.log10(span / targetTicks));
  const step = [1, 2, 5, 10].map(n => n * power).find(n => n >= span / targetTicks)!;
  ctx.font = '10px Inter, system-ui, sans-serif';
  for (let pos = Math.ceil(state.start / step) * step; pos < state.end; pos += step) { ctx.strokeStyle = palette.line; ctx.beginPath(); ctx.moveTo(x(pos), 51); ctx.lineTo(x(pos), height - 27); ctx.stroke(); ctx.fillStyle = palette.muted; ctx.fillText((pos + 1).toLocaleString('en-US'), x(pos) + 3, 52); }
  const selected = loaded.variants.find(v => variantKey(v) === state.selected);
  for (const lane of lanes) {
    const { y, height: h, id } = lane;
    ctx.fillStyle = palette.ink; ctx.font = '11px Inter, system-ui, sans-serif'; ctx.fillText(id === 'genes' ? 'MANE genes' : id === 'heatmap' ? 'ALT max DS' : id === 'difference' ? 'r13 − r10' : id === 'coverage' ? 'r13 coverage' : id === 'reference' ? 'Reference' : `DS_${id}`, 10, y + 14);
    ctx.strokeStyle = palette.line; ctx.beginPath(); ctx.moveTo(left, y + h); ctx.lineTo(right, y + h); ctx.stroke();
    ctx.save(); ctx.beginPath(); ctx.rect(left, y, right - left, h); ctx.clip();
    if (id === 'genes') {
      const ends: number[] = [];
      for (const gene of genes.filter(g => g.chrom === state.chrom && g.end > state.start && g.start < state.end)) {
        let row = ends.findIndex(end => end < gene.start); if (row < 0) row = ends.length; ends[row] = gene.end;
        const gy = y + 18 + row * 24; if (gy > y + h - 6) continue;
        ctx.strokeStyle = palette.accent; ctx.beginPath(); ctx.moveTo(x(gene.start), gy + 8); ctx.lineTo(x(gene.end), gy + 8); ctx.stroke();
        ctx.fillStyle = palette.accent;
        for (const [a, b] of gene.exons) ctx.fillRect(x(a), gy + 3, Math.max(1, x(b) - x(a)), 10);
        ctx.fillStyle = palette.ink; ctx.font = '10px Inter, system-ui, sans-serif';
        ctx.fillText(`${gene.name} ${gene.strand === '+' ? '→' : '←'}`, Math.max(left + 2, x(gene.start)), gy - 4);
        for (let px = Math.max(left, x(gene.start)) + 24; px < Math.min(right, x(gene.end)); px += 36) { const dir = gene.strand === '+' ? 1 : -1; ctx.strokeStyle = palette.accent; ctx.beginPath(); ctx.moveTo(px - dir * 3, gy + 5); ctx.lineTo(px, gy + 8); ctx.lineTo(px - dir * 3, gy + 11); ctx.stroke(); }
      }
    } else if (id === 'reference') {
      if (loaded.sequence && (right - left) / span >= 7) {
        ctx.font = '11px monospace';
        const colors: Record<string, string> = { A: '#208978', C: '#488ac5', G: '#ae8833', T: '#cb635e' };
        for (let i = 0; i < loaded.sequence.length; i++) { ctx.fillStyle = colors[loaded.sequence[i]] || palette.muted; ctx.fillText(loaded.sequence[i], x(state.start + i) + 1, y + h / 2); }
      } else { ctx.fillStyle = palette.muted; ctx.fillText('Zoom to base letters', left + 5, y + h / 2); }
    } else if (id === 'coverage') {
      for (const s of loaded.summaries) { ctx.fillStyle = s.acceptedR13 === s.rows ? palette.accent : s.acceptedR13 ? palette.compare : palette.missing; ctx.globalAlpha = 0.6; ctx.fillRect(x(Math.max(s.start, state.start)), y + 7, Math.max(1, x(Math.min(s.end, state.end)) - x(Math.max(s.start, state.start))), h - 14); }
      ctx.globalAlpha = 1; ctx.fillStyle = palette.ink; if (!loaded.summaries.length) ctx.fillText('Outside source collection', left + 5, y + 22);
    } else if (id === 'heatmap') {
      const alts = ['A', 'C', 'G', 'T']; const band = (h - 12) / 4;
      for (let i = 0; i < 4; i++) { ctx.fillStyle = palette.muted; ctx.fillText(alts[i], left + 3, y + 7 + i * band + band / 2); }
      if (loaded.exact) {
        const pixels = new Map<string, { v: number; mismatch: boolean; px: number; band: number }>();
        for (const v of visible) { const band = alts.indexOf(v.alt); if (band < 0) continue; const values = [0, 1, 2, 3].map(c => value(v, state.model, c, state.gene)).filter(a => a !== null) as number[]; if (!values.length) continue; const px = Math.floor(x(v.pos)), key = `${px}:${band}`; const score = Math.max(...values); const old = pixels.get(key); if (!old || old.v < score) pixels.set(key, { v: score, mismatch: v.refMismatch, px, band }); }
        for (const p of pixels.values()) { ctx.fillStyle = p.mismatch ? palette.compare : palette.accent; ctx.globalAlpha = 0.12 + 0.88 * p.v; ctx.fillRect(p.px, y + 6 + p.band * band, Math.max(1, (right - left) / span), band - 1); }
        ctx.globalAlpha = 1;
      } else { ctx.fillStyle = palette.muted; ctx.fillText('Zoom for exact alleles', left + 20, y + h / 2); }
    } else if (id === 'difference') {
      const middle = y + h / 2;
      ctx.strokeStyle = palette.line; ctx.beginPath(); ctx.moveTo(left, middle); ctx.lineTo(right, middle); ctx.stroke();
      if (loaded.exact) {
        for (const v of visible) for (const match of matchedAnnotations(v)) { if (state.gene && match.gene !== state.gene) continue; const delta = Math.max(...match.r13.ds) - Math.max(...match.r10.ds); ctx.fillStyle = delta >= 0 ? palette.accent : palette.compare; ctx.fillRect(x(v.pos), middle - Math.max(0, delta) * (h / 2 - 4), Math.max(1, (right - left) / span), Math.max(1, Math.abs(delta) * (h / 2 - 4))); }
      } else { ctx.fillStyle = palette.muted; ctx.fillText('Exact matched gene comparisons require zoom', left + 4, y + 12); }
    } else {
      const channel = ['AG', 'AL', 'DG', 'DL'].indexOf(id);
      const scores = (model: Model) => loaded.exact ? visible.map(v => value(v, model, channel, state.gene)).filter(v => v !== null) as number[] : loaded.summaries.filter(s => (s.models[model]?.annotations || 0) > 0).map(s => s.models[model]!.max[channel] / 100000);
      const all = scores(state.model);
      const colors: [Model, string][] = [[state.model, CHANNEL_COLORS[channel]]];
      if (state.compare && state.model !== 'baseline') colors.push([state.model === 'r10' ? 'r13' : 'r10', palette.compare]);
      const max = state.autoscale ? colors.reduce((maximum, [model]) => scores(model).reduce((m, score) => Math.max(m, score), maximum), 0.001) : 1;
      for (const [model, color] of colors) {
        ctx.fillStyle = color; ctx.globalAlpha = model === state.model ? 0.85 : 0.45;
        const pixels = new Map<number, number>();
        if (loaded.exact) for (const v of visible) { const score = value(v, model, channel, state.gene); if (score === null) continue; const pixel = Math.floor(x(v.pos)); pixels.set(pixel, Math.max(pixels.get(pixel) ?? 0, score)); }
        if (loaded.exact) { for (const [px, score] of pixels) ctx.fillRect(px, y + h - 4 - Math.min(score / max, 1) * (h - 13), Math.max(1, (right - left) / span), Math.max(1, Math.min(score / max, 1) * (h - 13))); }
        else for (const s of loaded.summaries) { const stats = s.models[model]; if (!stats || !stats.annotations) continue; const score = stats.max[channel] / 100000; ctx.fillRect(x(s.start), y + h - 4 - score / max * (h - 13), Math.max(1, x(s.end) - x(s.start)), Math.max(1, score / max * (h - 13))); }
      }
      ctx.globalAlpha = 1; ctx.fillStyle = palette.muted; ctx.font = '9px Inter, system-ui, sans-serif'; ctx.fillText(max.toFixed(max < 0.01 ? 5 : 2), left + 2, y + 6);
      if (!all.length) ctx.fillText(state.model === 'r13' ? 'Preview pending / prediction absent' : 'Prediction absent', left + 3, y + h / 2);
    }
    if (state.roi?.chrom === state.chrom) { ctx.fillStyle = palette.accent; ctx.globalAlpha = 0.10; ctx.fillRect(x(state.roi.start), y, x(state.roi.end) - x(state.roi.start), h); ctx.globalAlpha = 1; }
    if (selected) {
      ctx.strokeStyle = palette.ink; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(x(selected.pos), y); ctx.lineTo(x(selected.pos), y + h); ctx.stroke(); ctx.setLineDash([]);
      if (['AG', 'AL', 'DG', 'DL'].includes(id)) { const channel = ['AG', 'AL', 'DG', 'DL'].indexOf(id); for (const a of selected.models[state.model]) if (a.ds[channel] > 0) { const site = selected.pos + a.dp[channel]; if (site >= state.start && site < state.end) { ctx.fillStyle = palette.accent; ctx.fillRect(x(site) - 2, y + 4, 4, 8); } } }
    }
    ctx.restore();
  }
  ctx.fillStyle = palette.muted; ctx.font = '10px Inter, system-ui, sans-serif'; ctx.fillText('DS: variant delta · missing ≠ zero · summaries: annotation maxima · GRCh38.p14', 10, height - 14);
  return height;
}
