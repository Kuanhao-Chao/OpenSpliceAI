import { comparisonStatistics, exactCsv, matchedAnnotations, visibleVariants } from './science';
import { variantKey } from './state';
import type { BrowserState, ExactPlot, ExactQuery, Variant } from './types';

export function queryPlot(all: Variant[], state: BrowserState): ExactPlot {
  const variants = visibleVariants(all, state);
  const score = (v: Variant) => v.models[state.model].filter(a => !state.gene || a.gene === state.gene).reduce((m, a) => Math.max(m, ...a.ds), -1);
  const chart = variants.filter(v => 'ACGT'.includes(v.alt) && score(v) >= 0);
  const differencePos: number[] = [], difference: number[] = [];
  for (const v of variants) for (const m of matchedAnnotations(v)) {
    if (state.gene && m.gene !== state.gene) continue;
    differencePos.push(v.pos); difference.push(Math.max(...m.r13.ds) - Math.max(...m.r10.ds));
  }
  return { pos: Uint32Array.from(chart, v => v.pos), alt: Uint8Array.from(chart, v => 'ACGT'.indexOf(v.alt)),
    score: Float32Array.from(chart, score), mismatch: Uint8Array.from(chart, v => Number(v.refMismatch)),
    differencePos: Uint32Array.from(differencePos), difference: Float32Array.from(difference) };
}

/** All selection, ranking and statistics operate on the entire exact query. */
export function queryVariants(all: Variant[], state: BrowserState, page: number, channel: number, position?: number): ExactQuery {
  const variants = visibleVariants(all,state);
  const actualPage = Math.max(0,Math.min(page,Math.ceil(variants.length / 100) - 1));
  const score = (v: Variant) => v.models[state.model].filter(a => !state.gene || a.gene === state.gene).reduce((m,a) => Math.max(m,...a.ds),-1);
  const top = [...variants].sort((a,b) => score(b)-score(a)).slice(0,400);
  const selected = all.find(v => variantKey(v) === state.selected);
  if (selected && !top.includes(selected)) top.unshift(selected);
  const compared = state.gene ? variants.map(v => ({ ...v, models: { ...v.models,
    r10:v.models.r10.filter(a=>a.gene===state.gene),r13:v.models.r13.filter(a=>a.gene===state.gene) } })) : variants;
  const stats = comparisonStatistics(compared,channel);
  const pixels = new Set(stats.points.map(p => Math.round(p.x*300) + Math.round(p.y*184)*301));
  return { total:variants.length,page:actualPage,rows:variants.slice(actualPage*100,(actualPage+1)*100),top,selected,
    atPosition:position===undefined?[]:variants.filter(v=>v.pos===position),
    plot:queryPlot(variants,state),
    stats:{n:stats.n,meanDifference:stats.meanDifference,correlation:stats.correlation,pixels:Uint32Array.from(pixels)} };
}
export const queryCsv = (all: Variant[],state: BrowserState) => exactCsv(visibleVariants(all,state),state.snapshot);
