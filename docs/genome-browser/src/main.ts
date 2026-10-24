import './style.css';
import { checksum } from './resources';
import { GenomeEngine, type Config } from './igv';
import { publishedCoverage } from './coverage';
import { DataSource, summaryTotals } from './data';
import { CHANNEL_COLORS, draw, layout, margin, type Palette } from './paint';
import { comparisonStatistics, csvCell, exactCsv, predictionStatus, visibleVariants } from './science';
import { searchSequence } from './search';
import { clampView, DEFAULT_STATE, EXACT_BP, formatView, History, orderedContigs, parseLocus, restoreState, serializeState, snapshotLink, TRACKS, variantKey } from './state';
import type { BrowserState, Gene, LoadedView, Manifest, SearchHit, Variant, View } from './types';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id)! as T;
const select = (id: string) => $<HTMLSelectElement>(id);
const input = (id: string) => $<HTMLInputElement>(id);
const esc = (text: unknown) => String(text).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
const canvas = $<HTMLCanvasElement>('track-canvas');
const ctx = canvas.getContext('2d')!;
const history = new History();
let source: DataSource;
let state: BrowserState = { ...DEFAULT_STATE };
let loaded: LoadedView | null = null;
let genes: Gene[] = [];
let viewRequest: AbortController | null = null;
let searchRequest: AbortController | null = null;
let epoch = 0, successfulEpoch = -1;
let datasetEpoch = 0, datasetReady = false;
let datasetRequest: AbortController | null = null;
let engine: GenomeEngine | null = null;
let locusTimer: ReturnType<typeof setTimeout>;
let activePane = 0;
let allelePage = 0, navigationId = 0, paintId = 0, hashId = 0;
let nativeSession: Config | undefined, manifestSha256 = '';
let painting: Promise<void> = Promise.resolve();
let wholeGenome = false;
let loading: Promise<void> = Promise.resolve();
let searchHits: SearchHit[] = [], hitCursor = -1;
let catalog: { id: string; label: string; manifest: string }[] = [];
const descriptions: Record<string, string> = {
  genes: 'GENCODE v24 scoring annotation', reference: 'Pinned GRCh38 reference bases', coverage: 'Input and predicted ALT counts; missing predictions are distinct from zeros',
  AG: 'Acceptor gain delta', AL: 'Acceptor loss delta', DG: 'Donor gain delta', DL: 'Donor loss delta', heatmap: 'Largest DS by ALT allele', difference: 'Difference in maximum DS for matched genes',
};
function palette(): Palette { const css = getComputedStyle(document.body); const get = (name: string) => css.getPropertyValue(`--${name}`).trim(); return { bg: get('panel'), ink: get('ink'), muted: get('muted'), line: get('line'), accent: get('accent'), compare: get('compare'), missing: get('missing') }; }
function report(error: unknown) { if (error instanceof Error && error.name === 'AbortError') return; const message = error instanceof Error ? error.message : String(error); $('load-status').textContent = message; $('retry').hidden = false; }
function on(id: string, action: () => void | Promise<void>) { $(id).addEventListener('click', () => { Promise.resolve().then(action).catch(report); }); }
function updateHash(push = true) {
  const revision = ++hashId;
  const write = () => { const hash = new URLSearchParams(serializeState(state)); hash.set('v','2'); hash.set('manifestSha256',manifestSha256); hash.set('pane',String(activePane)); if (nativeSession) hash.set('igv',JSON.stringify(nativeSession)); window.history.replaceState(null,'',`${location.pathname}${location.search}#${hash}`); };
  write(); if (push) history.push(state);
  if (engine && datasetReady) void engine.session().then(saved => { if (revision === hashId) { nativeSession = saved; write(); } }).catch(report);
}
function syncControls() {
  document.body.dataset.theme = state.theme;
  for (const id of ['theme', 'model', 'density', 'alt']) select(id).value = state[id as 'theme' | 'model' | 'density' | 'alt'];
  select('chromosome').value = state.chrom;
  input('compare').checked = state.compare; input('autoscale').checked = state.autoscale;
  input('threshold').value = String(state.threshold); input('gene-filter').value = state.gene;
  $('view-label').textContent = formatView(state);
  input('locus').value = formatView(state); trackControls();
}
function trackControls() {
  const filter = input('track-filter').value.toLowerCase();
  const ordered = [...state.tracks, ...TRACKS.filter(t => !state.tracks.includes(t))];
  $('track-list').innerHTML = ordered.filter(t => `${t} ${descriptions[t]}`.toLowerCase().includes(filter)).map(t => `<div class="track-item" title="${esc(descriptions[t])}"><label class="check"><input type="checkbox" data-track="${t}" ${state.tracks.includes(t) ? 'checked' : ''}/>${t}</label><button data-move="${t}" data-dir="-1" aria-label="Move ${t} up">↑</button><button data-move="${t}" data-dir="1" aria-label="Move ${t} down">↓</button><input type="number" min="32" max="240" value="${state.heights[t] || layout({ ...state, tracks: [t] })[0].height}" data-height="${t}" aria-label="${t} height in pixels" /></div>`).join('');
}
function paint(): Promise<void> { painting = render(); painting.catch(report); return painting; }
async function render() {
  const data = loaded, revision = ++paintId;
  if (!data || wholeGenome || data.view.chrom !== state.chrom || data.view.start !== state.start || data.view.end !== state.end) return;
  if (data.exact && data.queryToken !== undefined) {
    try {
      const query = await source.resources.decoder.query<import('./types').ExactQuery>(data.queryToken,structuredClone(state),allelePage,Number(select('stats-event').value));
      if (revision !== paintId || loaded !== data) return;
      data.query = query; allelePage = query.page;
    } catch(error) { if(revision !== paintId || loaded !== data) return; throw error; }
  }
  const detailState = { ...state, tracks: state.tracks.filter(t => ['heatmap', 'difference'].includes(t)) };
  const width = Math.max(280, canvas.parentElement!.clientWidth), lanes = layout(detailState), height = (lanes.at(-1)?.y || 64) + (lanes.at(-1)?.height || 0) + 42;
  const dpr = Math.min(2, devicePixelRatio || 1);
  canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr); canvas.style.height = `${height}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); draw(ctx, width, detailState, data, data.genes || genes, palette());
  const variants = data.query?.rows || [];
  const total = data.query?.total || 0, top = data.query?.top || [], selected = data.query?.selected;
  select('variant-select').innerHTML = `<option value="">${data.exact ? `${total.toLocaleString()} alleles; showing top ${Math.min(400, total)}` : 'Zoom in for exact alleles'}</option>` + top.map(v => `<option value="${esc(variantKey(v))}">${esc(variantKey(v))} · ${esc(predictionStatus(v, state.model))}</option>`).join('');
  select('variant-select').value = state.selected;
  renderInspector(selected); paintOverview(); paintScatter();
  const pages = Math.max(1, Math.ceil(total / 100)); allelePage = Math.min(allelePage, pages - 1);
  $('allele-page').textContent = `${total.toLocaleString()} alleles · page ${allelePage + 1} / ${pages}`;
  $('allele-rows').innerHTML = variants.map(v => `<tr><td><button data-allele="${esc(variantKey(v))}">${esc(variantKey(v))}</button></td><td>${esc(predictionStatus(v, state.model))}</td><td>${v.models[state.model].map(a => esc(a.gene)).join(', ') || '—'}</td></tr>`).join('');
  $('filter-note').hidden = data.exact;
  for (const id of ['alt', 'gene-filter', 'threshold']) (document.getElementById(id) as HTMLInputElement).disabled = !data.exact;
  $('gene-details').innerHTML = (data.genes || genes).filter(g => g.chrom === state.chrom && g.start < state.end && g.end > state.start).slice(0, 30).map(g => `<div class="prediction"><h3>${esc(g.name)} · ${g.strand} strand</h3><p class="caption">${esc(formatView(g))} · ${g.exons.length} exons</p>${g.exons.map(([a, b], i) => `<button data-exon-start="${a}" data-exon-end="${b}" aria-label="Navigate to ${esc(g.name)} exon ${g.strand === '+' ? i + 1 : g.exons.length - i}">Exon ${g.strand === '+' ? i + 1 : g.exons.length - i}: ${(a + 1).toLocaleString()}–${b.toLocaleString()}</button>`).join('')}</div>`).join('') || '<p class="caption">No scoring annotation gene overlaps this view.</p>';
  const r = source.resources; $('resource-status').textContent = `${(r.transferred / 1048576).toFixed(2)} MiB received after HTTP decoding · ${(r.cache.bytes / 1048576).toFixed(1)} MiB cache · ${r.requests} requests`;
  const outside = source.manifest.scope === 'demonstration' && publishedCoverage(state, source.manifest.publishedIntervals || []).kind === 'outside';
  $<HTMLButtonElement>('csv').disabled = !data.exact || outside; $<HTMLButtonElement>('summary-csv').disabled = outside;
}
function showPublishedCoverage(view: View) {
  const node = $('demo-coverage');
  node.hidden = source.manifest.scope !== 'demonstration';
  if (node.hidden) return null;
  const coverage = publishedCoverage(view, source.manifest.publishedIntervals || []);
  node.dataset.coverage = coverage.kind;
  const count = `${coverage.coveredBp.toLocaleString()} / ${coverage.totalBp.toLocaleString()} bp within published intervals`;
  node.textContent = coverage.kind === 'outside' ? 'Scores are unavailable here. Choose a scored example above.'
    : `${coverage.kind === 'partial' ? 'Partially covered demonstration' : 'Published demonstration interval'} · ${count}. Input SNVs are shown where represented.`;
  if (coverage.kind === 'partial') for (const [i, bookmark] of (source.manifest.bookmarks || []).entries()) {
    if (bookmark.chrom !== view.chrom || bookmark.start >= view.end || bookmark.end <= view.start) continue;
    const button = document.createElement('button'); button.dataset.bookmark = String(i); button.textContent = `Focus ${bookmark.label}`;
    node.append(' ', button);
  }
  return coverage;
}
async function navigate(view: View, changes: Partial<BrowserState> = {}, push = true, native = true) {
  const request = ++navigationId; wholeGenome = false;
  clearTimeout(locusTimer); viewRequest?.abort(); successfulEpoch = -1;
  $('browser').classList.add('loading');
  state = { ...state, ...changes, ...clampView(view, source.manifest.contigs) };
  allelePage = 0;
  syncControls(); updateHash(push);
  if (native) await engine?.search(state);
  if (request !== navigationId) return;
  if (native && engine?.views.length) {
    activePane = Math.max(0, Math.min(activePane, engine.views.length - 1));
    select('locus-pane').innerHTML = engine.views.map((v, i) => `<option value="${i}">${esc(formatView(v))}</option>`).join('');
    select('locus-pane').value = String(activePane);
  }
  updateHash(false); await refresh();
}
function clearLoadedView() {
  $('demo-coverage').hidden = true;
  loaded = null; successfulEpoch = -1;
  ctx.clearRect(0, 0, canvas.width, canvas.height); renderInspector();
  select('variant-select').innerHTML = '<option>Loading exact alleles…</option>';
  $('gene-details').textContent = 'Waiting for current-view data.';
  $('stats').textContent = 'Waiting for current-view data.';
  $('allele-rows').innerHTML = ''; $('allele-page').textContent = 'Waiting for current-view data.';
  const scatter = $<HTMLCanvasElement>('scatter'); scatter.getContext('2d')!.clearRect(0, 0, scatter.width, scatter.height);
}
function refresh(): Promise<void> {
  viewRequest?.abort(); viewRequest = new AbortController(); const signal = viewRequest.signal, local = ++epoch;
  successfulEpoch = -1; $('retry').hidden = true; $('browser').classList.add('loading'); $('load-status').textContent = 'Loading data…';
  // Retain the previous frame with an explicit label until this generation commits.
  if (loaded) $('load-status').textContent = `Loading ${formatView(state)} · previous frame ${formatView(loaded.view)}`;
  paintOverview();
  loading = source.load({ chrom: state.chrom, start: state.start, end: state.end }, signal).then(result => {
    if (local !== epoch || signal.aborted) return;
    loaded = result; successfulEpoch = local;
    const totals = summaryTotals(result.summaries);
    const coverage = showPublishedCoverage(state);
    $('load-status').textContent = coverage?.kind === 'outside' ? 'Outside published demonstration intervals · scores unavailable' : `${result.exact ? 'Exact' : 'Binned maxima · zoom in for exact alleles'} · ${totals.rows.toLocaleString()} source occurrences in overlapping bins${coverage?.kind === 'partial' ? ' · partially covered demonstration' : ''}`;
    return paint();
  }).catch(error => { if (local === epoch && !signal.aborted) { clearLoadedView(); report(error); } }).finally(() => { if (local === epoch) $('browser').classList.remove('loading'); });
  return loading;
}
function renderInspector(v?: Variant) {
  if (!v) { $('variant-details').innerHTML = '<p class="caption">Select an exact allele. All source annotations will appear here, including zero scores and conflicts.</p>'; return; }
  const contig = source.manifest.contigs.find(c => c.name === v.chrom)!;
  let html = `<div class="variant-title">${esc(variantKey(v))}</div><span class="badge">${v.occurrences.length} source occurrence${v.occurrences.length === 1 ? '' : 's'}</span><span class="badge">${v.r13Accepted ? 'Final audited r13 input' : 'r13 evidence unavailable'}</span>`;
  if (v.refMismatch) html += '<p class="warning caption">REF differs from the pinned FASTA. Preserved for inspection; excluded from ordinary model comparisons.</p>';
  for (const model of ['r10', 'r13', 'baseline'] as const) {
    html += `<div class="prediction"><h3>${esc(source.manifest.models[model].label)} <span class="badge">${esc(predictionStatus(v, model))}</span></h3>`;
    if (!v.models[model].length) {
      const overlapping = (loaded?.genes || genes).some(g => g.chrom === v.chrom && g.start <= v.pos && g.end > v.pos);
      const supported = genes.some(g => g.chrom === v.chrom);
      const reason = !supported ? 'This contig has no configured scoring annotation.' : !overlapping ? 'This position is outside the configured scoring genes.' : v.refMismatch ? 'The input REF does not match the scoring reference.' : 'No prediction annotation was recorded; this remains visible for audit.';
      html += `<p class="caption">${esc(reason)} This is an absent prediction, not a delta of zero.</p>`;
    }
    for (const a of v.models[model]) {
      html += `<p><strong>${esc(a.gene)}</strong> <span class="caption">${a.occurrences.length} occurrence${a.occurrences.length === 1 ? '' : 's'}</span></p><table><thead><tr><th>Event</th><th>DS</th><th>DP</th><th>Site</th></tr></thead><tbody>`;
      for (let c = 0; c < 4; c++) { const site = v.pos + 1 + a.dp[c]; const nonzero = a.ds[c] > 0; html += `<tr><td>${['AG', 'AL', 'DG', 'DL'][c]}</td><td>${a.ds[c].toFixed(source.manifest.models[model].precision)}</td><td>${a.dp[c] > 0 ? '+' : ''}${a.dp[c]}</td><td>${nonzero && site >= 1 && site <= contig.length ? `<button class="site-link" data-site="${site}">${site.toLocaleString()}</button>` : nonzero ? 'outside contig' : '—'}</td></tr>`; }
      html += '</tbody></table>';
    }
    if (new Set(v.models[model].map(a => a.gene)).size < v.models[model].length) html += '<p class="warning caption">Distinct annotations for the same gene are preserved; this gene is excluded from the comparison.</p>';
    html += '</div>';
  }
  html += `<details><summary>Source identities</summary><p class="caption">chunk:row ${esc(v.occurrences.join(', '))}</p></details>`;
  html += '<details><summary>Audit evidence</summary><div id="source-evidence" class="caption">Loading source evidence…</div></details>';
  $('variant-details').innerHTML = html;
  const key=variantKey(v),chunks=[...new Set(v.occurrences.map(o=>Number(o.split(':')[0])))];
  void source.provenance(chunks).then(evidence=>{if(state.selected!==key)return;const node=document.getElementById('source-evidence');if(!node)return;
    node.innerHTML=chunks.map(chunk=>`<p>Chunk ${chunk}</p>`+Object.entries(evidence[String(chunk)]||{}).map(([model,e])=>`<p>${esc(model)} · ${esc(e.state)}<br>Output SHA-256 <code>${esc(e.outputSha256)}</code>${e.receiptSha256?`<br>Receipt SHA-256 <code>${esc(e.receiptSha256)}</code>`:'<br>Model identity is not bound by a recorded receipt.'}</p>`).join('')).join('')||'Per-chunk evidence is not published in this snapshot.';
  }).catch(error=>{const node=document.getElementById('source-evidence');if(node&&state.selected===key)node.textContent=`Evidence unavailable: ${String(error)}`;});
}
function prepareSmallCanvas(id: string, height: number) { const c = $<HTMLCanvasElement>(id), width = c.clientWidth || 320, dpr = Math.min(2, devicePixelRatio || 1); c.width = width * dpr; c.height = height * dpr; const cx = c.getContext('2d')!; cx.setTransform(dpr, 0, 0, dpr, 0, 0); return { c, cx, width }; }
function paintOverview() {
  if (!source) return;
  const p = palette(), primary = orderedContigs(source.manifest.contigs).filter(c => /^chr(\d+|X|Y|M)$/.test(c.name));
  const { cx, width } = prepareSmallCanvas('genome-overview', 55);
  const cell = width / Math.max(primary.length, 1); cx.font = '9px system-ui';
  primary.forEach((c, i) => { const max = c.overview.reduce((m, s) => Math.max(m, ...(s.models[state.model]?.max || [0])), 0) / 100000; cx.fillStyle = c.name === state.chrom ? p.accent : p.line; cx.fillRect(i * cell + 2, 13, Math.max(2, cell - 4), 26); cx.fillStyle = p.compare; cx.fillRect(i * cell + 2, 39 - max * 24, Math.max(2, cell - 4), Math.max(1, max * 24)); cx.fillStyle = p.muted; cx.fillText(c.name.replace('chr', ''), i * cell + 3, 51); });
  const mini = prepareSmallCanvas('minimap', 32), contig = source.manifest.contigs.find(c => c.name === state.chrom)!;
  mini.cx.fillStyle = p.line; mini.cx.fillRect(0, 9, mini.width, 16);
  for (const s of contig.overview) { mini.cx.fillStyle = s.acceptedR13 === s.rows ? p.accent : p.missing; mini.cx.fillRect(s.start / contig.length * mini.width, 10, Math.max(1, (s.end - s.start) / contig.length * mini.width), 14); }
  mini.cx.strokeStyle = p.ink; mini.cx.lineWidth = 2; mini.cx.strokeRect(state.start / contig.length * mini.width, 5, Math.max(3, (state.end - state.start) / contig.length * mini.width), 24);
}
function paintScatter() {
  const c = $<HTMLCanvasElement>('scatter'), cx = c.getContext('2d')!, p = palette(), result = loaded?.query?.stats || {n:0,meanDifference:null,correlation:null,pixels:new Uint32Array()};
  cx.clearRect(0, 0, c.width, c.height); cx.fillStyle = p.bg; cx.fillRect(0, 0, c.width, c.height); cx.strokeStyle = p.line; cx.beginPath(); cx.moveTo(40, 10); cx.lineTo(40, 194); cx.lineTo(340, 194); cx.moveTo(40, 194); cx.lineTo(340, 10); cx.stroke();
  cx.fillStyle = p.accent; cx.globalAlpha = 0.5;
  // Deterministic pixel occupancy avoids millions of overdraws in dense loci.
  for (const pixel of result.pixels) cx.fillRect(39 + pixel % 301,193 - Math.floor(pixel / 301),3,3);
  cx.globalAlpha = 1; cx.fillStyle = p.muted; cx.font = '11px system-ui'; cx.fillText('r10 delta score →', 140, 220); cx.fillText('r13', 3, 14);
  $('stats').textContent = loaded?.exact ? `${result.n.toLocaleString()} matched allele/gene pairs · mean r13−r10 ${result.meanDifference?.toFixed(5) ?? '—'} · Pearson r ${result.correlation?.toFixed(3) ?? 'undefined (no variance or no matches)'}` : 'Zoom in for exact matched allele/gene statistics. Wide-view maxima are not used as exact comparisons.';
}
async function inspectPosition(chrom: string,pos: number) {
  if (!datasetReady || wholeGenome) return;
  if (!loaded || loaded.view.chrom !== chrom || pos < loaded.view.start || pos >= loaded.view.end) {
    const pane = engine?.views.findIndex(v => v.chrom===chrom && v.start<=pos && pos<v.end) ?? -1;
    if (pane>=0) {activePane=pane; select('locus-pane').value=String(pane); await navigate(engine!.views[pane],{},true,false);}
    else await navigate({chrom,start:pos-100,end:pos+101},{selected:''});
  }
  if (loaded?.queryToken===undefined || successfulEpoch!==epoch) return;
  const query = await source.resources.decoder.query<import('./types').ExactQuery>(loaded.queryToken,structuredClone(state),allelePage,0,pos);
  const candidates = query.atPosition;
  if (candidates.length===1) {state.selected=variantKey(candidates[0]);updateHash();await paint();}
  else if(candidates.length>1) {
    state.selected=''; await paint();
    select('variant-select').innerHTML='<option value="">Choose an ALT allele at this position</option>'+candidates.map(v=>`<option value="${esc(variantKey(v))}">${esc(variantKey(v))}</option>`).join('');
    $('load-status').textContent=`${candidates.length} alleles at ${chrom}:${pos+1}; choose the exact REF/ALT`;select('variant-select').focus();
  }
}

async function goTo(inputText: string) {
  const variant = /^([^:]+):(\d[\d,]*):([A-Z])>([A-Z])$/i.exec(inputText.trim());
  if (variant) { const locus = parseLocus(`${variant[1]}:${variant[2]}`, source.manifest.contigs)!; const selected = `${locus.chrom}:${Number(variant[2].replaceAll(',', ''))}:${variant[3].toUpperCase()}>${variant[4].toUpperCase()}`; await navigate(locus, { selected, gene: '' }); return; }
  const region = parseLocus(inputText, source.manifest.contigs); if (region) { await navigate(region, { selected: '', gene: '' }); return; }
  const matches = genes.filter(g => g.name.toLowerCase() === inputText.trim().toLowerCase());
  if (!matches.length) throw new Error('Gene not found in the scoring annotation. Enter an exact gene symbol or a genomic locus.');
  const gene = matches.find(g => g.chrom === state.chrom) || matches[0];
  await navigate({ chrom: gene.chrom, start: Math.max(0, gene.start - 250), end: gene.end + 250 }, { selected: '', gene: '' });
}
async function stableExport(action: (data: LoadedView, metadata: BrowserState) => void | Promise<void>, allowOverview = false) {
  const current = epoch; await loading; await painting; await engine?.session();
  const metadata=structuredClone(state);
  if(wholeGenome && allowOverview) {metadata.chrom='all';await action({view:metadata,exact:false,variants:[],summaries:[],sequence:null},metadata);return;}
  if (!loaded || successfulEpoch !== current || current !== epoch || wholeGenome) throw new Error('Export requires a successfully loaded, stable view. Retry the data or wait for the new locus.');
  await action(loaded,metadata);
}
function download(blob: Blob, extension: string, metadata = state) { const a = document.createElement('a'); const url = URL.createObjectURL(blob); a.href = url; a.download = `OpenSpliceAI-${metadata.snapshot}-${metadata.chrom==='all'?'whole-genome':`${metadata.chrom}-${metadata.start+1}-${metadata.end}`}.${extension}`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 30000); }

async function loadDataset(url: string, restore = true) {
  datasetRequest?.abort(); datasetRequest = new AbortController();
  const datasetSignal = datasetRequest.signal, thisDataset = ++datasetEpoch;
  datasetReady = false; nativeSession = undefined; wholeGenome = false; ++hashId;
  history.clear(); clearLoadedView();
  viewRequest?.abort(); searchRequest?.abort(); epoch++; engine?.destroy(); engine = null; source?.destroy();
  $('load-status').textContent = 'Loading dataset…'; $('retry').hidden = true;
  $('dataset-note').classList.remove('error');
  $('dataset-note').textContent = 'Loading dataset information…';
  const manifestUrl = new URL(url, location.href).href;
  const response = await fetch(manifestUrl, { signal: datasetSignal }); if (!response.ok) throw new Error(`Dataset manifest failed (${response.status})`);
  const manifestBytes = await response.arrayBuffer();
  manifestSha256 = await checksum(manifestBytes);
  const expectedHash = restore ? new URLSearchParams(location.hash.slice(1)).get('manifestSha256') : null;
  if (expectedHash && expectedHash !== manifestSha256) throw new Error('The manifest differs from the shared immutable snapshot hash');
  const manifest = JSON.parse(new TextDecoder().decode(manifestBytes)) as Manifest;
  if (manifest.manifestVersion !== 2) { location.replace(`${import.meta.env.BASE_URL}legacy/?manifest=${encodeURIComponent(manifestUrl)}${location.hash}`); return; }
  const nextSource = new DataSource(manifest, manifestUrl);
  const result = await Promise.all([nextSource.genes(datasetSignal), nextSource.overview(datasetSignal)]).catch(error => { nextSource.destroy(); throw error; });
  if (thisDataset !== datasetEpoch || datasetSignal.aborted) { nextSource.destroy(); return; }
  source = nextSource;
  genes = result[0];
  const fallback = { ...DEFAULT_STATE, ...manifest.defaultView, snapshot: manifest.id };
  state = restore ? restoreState(location.hash, fallback, manifest.contigs) : fallback;
  if (state.snapshot !== manifest.id) throw new Error('The shared link identifies a different immutable snapshot. Choose that dataset before restoring the link.');
  const global = manifest.coverage?.global;
  const prediction = global?.models.r13.uniqueCanonicalPredicted || 0;
  $('dataset-note').innerHTML = `<strong>${esc(manifest.label)}</strong> · ${manifest.scope === 'demonstration' ? `Small published demonstration · ${manifest.publishedIntervals?.length || 0} scored regions. Full-data browser packaging is a separate preparation step.` : 'Complete original SpliceAI source collection'}<br>r10 and r13 processing complete · ${prediction.toLocaleString()} / ${global?.uniqueCanonicalInput.toLocaleString()} canonical input SNVs have predictions per model (${global ? (100 * prediction / global.uniqueCanonicalInput).toFixed(2) : '—'}%). Missing predictions remain explicit.<br><span class="caption">Snapshot ${esc(manifest.id)} · ${esc(manifest.created.split('T')[0])}. ${esc(manifest.r13Evidence)}</span>`;
  select('model').querySelector<HTMLOptionElement>('[value="r13"]')!.textContent = 'OpenSpliceAI r13';
  $('example-loci').innerHTML = (manifest.bookmarks || []).map((b, i) => `<button data-bookmark="${i}" title="${esc(formatView(b))}">${esc(b.label)}</button>`).join('');
  select('chromosome').innerHTML = orderedContigs(manifest.contigs).map(c => `<option value="${esc(c.name)}">${esc(c.name)} · ${(c.length / 1e6).toFixed(2)} Mb</option>`).join('');
  $('gene-list').innerHTML = genes.map(g => `<option value="${esc(g.name)}"></option>`).join('');
  $('provenance').innerHTML = `<p>Snapshot <code>${esc(manifest.id)}</code> · ${esc(manifest.sourceOccurrences.toLocaleString())} prepared source occurrences, ${esc(manifest.acceptedR13Occurrences.toLocaleString())} with accepted r13 evidence.</p><p>Reference SHA-256 <code>${esc(manifest.referenceSha256)}</code><br>Annotation SHA-256 <code>${esc(manifest.annotationSha256)}</code></p>` + Object.entries(manifest.models).map(([id, m]) => `<p>${esc(id)}: ${esc(m.label)}; ${m.precision}-decimal source precision${m.seed !== undefined ? `; random seed ${m.seed}` : ''}${m.sha256 ? `<br><code>${esc(m.sha256)}</code>` : ''}</p>`).join('');
  syncControls();
  engine = new GenomeEngine(source, $('igv-browser'), views => {
    if (!datasetReady) return;
    if (!views.length) { wholeGenome = true; ++epoch; viewRequest?.abort(); clearLoadedView(); $('browser').classList.remove('loading'); select('locus-pane').innerHTML='<option>Whole genome</option>'; updateHash(); $('view-label').textContent = 'Whole genome'; $('load-status').textContent = 'Whole-genome overview · choose a chromosome for exact alleles'; $<HTMLButtonElement>('csv').disabled = true; $<HTMLButtonElement>('summary-csv').disabled = true; return; }
    select('locus-pane').innerHTML = views.map((v, i) => `<option value="${i}">${esc(formatView(v))}</option>`).join('');
    activePane = Math.min(activePane, views.length - 1); select('locus-pane').value = String(activePane);
    clearTimeout(locusTimer); viewRequest?.abort(); successfulEpoch = -1;
    $('load-status').textContent = 'Updating locus…';
    locusTimer = setTimeout(() => navigate(views[activePane], {}, true, false).catch(report), 120);
  }, (chrom, pos) => {
    if (!datasetReady) return;
    inspectPosition(chrom,pos).catch(report);
  }, () => { if(!datasetReady || !engine)return;const current=engine;void current.session().then(saved=>{if(engine!==current||!datasetReady)return;nativeSession=saved;for(const t of saved.tracks as Config[]){const id=String(t.id||'').replace('osai-','');if(TRACKS.includes(id as never)&&typeof t.height==='number')state.heights[id]=t.height;}trackControls();updateHash(false);}).catch(report); });
  const saved = new URLSearchParams(location.hash.slice(1)).get('igv');
  await engine.create(state, saved ? JSON.parse(saved) : undefined);
  activePane = Math.min(Number(new URLSearchParams(location.hash.slice(1)).get('pane')) || 0,Math.max(0,engine.views.length-1));
  select('locus-pane').innerHTML = engine.views.map((v,i) => `<option value="${i}">${esc(formatView(v))}</option>`).join('');
  select('locus-pane').value = String(activePane);
  if (engine.views[activePane]) state = {...state,...clampView(engine.views[activePane],manifest.contigs)};
  else if(saved && JSON.parse(saved).locus?.includes('all')) wholeGenome=true;
  datasetReady = true; syncControls(); history.push(state); updateHash(false); if(wholeGenome){clearLoadedView();$('view-label').textContent='Whole genome';$('load-status').textContent='Whole-genome overview · choose a chromosome for exact alleles';$<HTMLButtonElement>('csv').disabled=true;}else await refresh();
}

async function initialize() {
  const params = new URLSearchParams(location.search), manifestUrl = params.get('manifest');
  if (manifestUrl) { catalog = [{ id: 'custom', label: 'Configured dataset', manifest: manifestUrl }]; }
  else { const configResponse = await fetch(`${import.meta.env.BASE_URL}settings.json`); if (!configResponse.ok) throw new Error('Browser settings are missing'); const config = await configResponse.json(); const response = await fetch(new URL(config.catalogUrl, new URL(`${import.meta.env.BASE_URL}settings.json`, location.href))); if (!response.ok) throw new Error('Dataset catalog failed'); const data = await response.json(); catalog = data.datasets; }
  if (!catalog.length) throw new Error('No published datasets are configured');
  select('dataset').innerHTML = catalog.map(c => `<option value="${esc(c.id)}">${esc(c.label)}</option>`).join('');
  const requested = new URLSearchParams(location.hash.slice(1)).get('dataset');
  const choice = catalog.find(c => c.id === requested) || catalog[0]; select('dataset').value = choice.id;
  await loadDataset(choice.manifest);
}

$('locus-form').addEventListener('submit', e => { e.preventDefault(); goTo(input('locus').value).catch(report); });
document.querySelectorAll<HTMLButtonElement>('[data-gene]').forEach(b => b.addEventListener('click', () => goTo(b.dataset.gene!).catch(report)));
select('dataset').addEventListener('change', () => loadDataset(catalog.find(c => c.id === select('dataset').value)!.manifest, false).catch(report));
select('chromosome').addEventListener('change', () => { const c = source.manifest.contigs.find(c => c.name === select('chromosome').value)!; navigate({ chrom: c.name, start: 0, end: c.length }, { selected: '', gene: '' }).catch(report); });
for (const id of ['theme', 'model', 'density', 'alt']) select(id).addEventListener('change', () => { state = { ...state, [id]: select(id).value }; document.body.dataset.theme = state.theme; updateHash(); engine?.sync(state).catch(report); paint(); });
for (const id of ['compare', 'autoscale']) input(id).addEventListener('change', () => { state = { ...state, [id]: input(id).checked }; updateHash(); engine?.sync(state).catch(report); paint(); });
input('threshold').addEventListener('change', () => { state.threshold = Math.min(1, Math.max(0, Number(input('threshold').value) || 0)); updateHash(); paint(); });
input('gene-filter').addEventListener('change', () => { state.gene = input('gene-filter').value.trim(); updateHash(); paint(); });
input('track-filter').addEventListener('input', trackControls);
$('track-list').addEventListener('change', e => { const t = e.target as HTMLInputElement; if (t.dataset.track) state.tracks = t.checked ? [...state.tracks, t.dataset.track] : state.tracks.filter(id => id !== t.dataset.track); if (t.dataset.height) state.heights = { ...state.heights, [t.dataset.height]: Math.max(32, Math.min(240, Number(t.value) || 58)) }; updateHash(); engine?.sync(state).catch(report); paint(); });
$('track-list').addEventListener('click', e => { const b = (e.target as HTMLElement).closest<HTMLButtonElement>('[data-move]'); if (!b) return; const tracks = [...state.tracks], from = tracks.indexOf(b.dataset.move!), to = from + Number(b.dataset.dir); if (from < 0 || to < 0 || to >= tracks.length) return; [tracks[from], tracks[to]] = [tracks[to], tracks[from]]; state.tracks = tracks; updateHash(); trackControls(); engine?.sync(state).catch(report); paint(); });
select('variant-select').addEventListener('change', () => { state.selected = select('variant-select').value; updateHash(); paint(); });
$('variant-details').addEventListener('click', e => { const b = (e.target as HTMLElement).closest<HTMLElement>('[data-site]'); if (b) { const site = Number(b.dataset.site) - 1; navigate({ chrom: state.chrom, start: site - 80, end: site + 81 }).catch(report); } });
$('gene-details').addEventListener('click', e => { const b = (e.target as HTMLElement).closest<HTMLElement>('[data-exon-start]'); if (b) navigate({ chrom: state.chrom, start: Number(b.dataset.exonStart) - 70, end: Number(b.dataset.exonEnd) + 70 }, { selected: '' }).catch(report); });
on('retry', () => datasetReady ? refresh() : catalog.length ? loadDataset((catalog.find(c => c.id === select('dataset').value) || catalog[0]).manifest) : initialize());
on('back', async () => { const s = history.back(); if (s) { state = s; await navigate(s, {}, false); } });
on('forward', async () => { const s = history.forward(); if (s) { state = s; await navigate(s, {}, false); } });
const shift = (fraction: number) => navigate({ ...state, start: state.start + (state.end - state.start) * fraction, end: state.end + (state.end - state.start) * fraction });
const zoom = (factor: number) => { const center = (state.start + state.end) / 2, half = (state.end - state.start) * factor / 2; return navigate({ ...state, start: center - half, end: center + half }); };
on('pan-left', () => shift(-0.5)); on('pan-right', () => shift(0.5)); on('zoom-in', () => zoom(0.5)); on('zoom-out', () => zoom(2));
on('whole', () => navigate({ chrom: state.chrom, start: 0, end: source.manifest.contigs.find(c => c.name === state.chrom)!.length }));
on('share', async () => {
  if (!source || !datasetReady) throw new Error('Load a dataset before sharing');
  const url = new URL(snapshotLink(location.href, source.manifestUrl, state));
  const hash = new URLSearchParams(url.hash.slice(1)); hash.set('v', '2'); hash.set('manifestSha256',manifestSha256); hash.set('pane',String(activePane)); hash.set('igv', JSON.stringify(await engine!.session())); url.hash = hash.toString();
  await navigator.clipboard.writeText(url.href); $('load-status').textContent = 'Snapshot and complete view link copied';
});
on('mark-roi', async () => { state.roi = { chrom: state.chrom, start: state.start, end: state.end }; await engine?.roi(state.roi); updateHash(); paint(); });
on('clear-roi', () => { state.roi = null; engine?.clearRois(); updateHash(); paint(); });
for (const direction of ['prev', 'next']) on(`${direction}-gene`, () => { const list = genes.filter(g => g.chrom === state.chrom).sort((a, b) => a.start - b.start); const center = (state.start + state.end) / 2; const gene = direction === 'next' ? list.find(g => g.start > center) : [...list].reverse().find(g => g.end < center); if (gene) return goTo(gene.name); });
on('tracks-open', () => { $('track-panel').classList.toggle('open'); $('tracks-open').setAttribute('aria-expanded', String($('track-panel').classList.contains('open'))); });
on('tracks-close', () => { $('track-panel').classList.remove('open'); $('tracks-open').setAttribute('aria-expanded', 'false'); $('tracks-open').focus(); });
on('svg', () => stableExport((_data,metadata) => download(new Blob([engine!.svg()], { type: 'image/svg+xml' }), 'svg',metadata),true));
on('png', () => stableExport(async (_data,metadata) => {
  const svg = engine!.svg(); const url = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml' }));
  try {
    const image = new Image(); image.src = url; await image.decode();
    const figure = document.createElement('canvas'); figure.width = image.width * 2; figure.height = image.height * 2;
    const cx = figure.getContext('2d')!; cx.fillStyle = '#fff'; cx.fillRect(0, 0, figure.width, figure.height); cx.drawImage(image, 0, 0, figure.width, figure.height);
    const blob = await new Promise<Blob>((resolve, reject) => figure.toBlob(b => b ? resolve(b) : reject(new Error('PNG encoding failed')), 'image/png'));
    download(blob, 'png',metadata);
  } finally { URL.revokeObjectURL(url); }
},true));
on('csv', () => stableExport(async (data,metadata) => { if (!data.exact || data.queryToken === undefined) throw new Error('Zoom in for exact CSV'); const csv = await source.resources.decoder.query<string>(data.queryToken,metadata,0,0,undefined,'csv'); download(new Blob([csv], {type:'text/csv'}),'exact.csv',metadata); }));
on('summary-csv', () => stableExport((data,metadata) => { const rows: unknown[][] = [['snapshot', 'chrom', 'bin_start_0based', 'bin_end_0based', 'source_occurrences', 'accepted_r13_occurrences', 'ref_mismatch_occurrences', 'model', 'annotation_entries', 'zero_annotations', 'max_DS_AG', 'max_DS_AL', 'max_DS_DG', 'max_DS_DL']]; for (const s of data.summaries) for (const [model, stats] of Object.entries(s.models)) rows.push([metadata.snapshot, metadata.chrom, s.start, s.end, s.rows, s.acceptedR13, s.refMismatch, model, stats!.annotations, stats!.zero, ...stats!.max.map(v => (v / 100000).toFixed(5))]); download(new Blob([rows.map(r => r.map(csvCell).join(',')).join('\r\n')], { type: 'text/csv' }), 'summary.csv',metadata); }));
select('stats-event').addEventListener('change', () => { if (loaded) paint(); });
$('genome-overview').addEventListener('click', e => { const c = orderedContigs(source.manifest.contigs).filter(c => /^chr(\d+|X|Y|M)$/.test(c.name)), target = c[Math.min(c.length - 1, Math.floor((e.offsetX / $<HTMLCanvasElement>('genome-overview').clientWidth) * c.length))]; if (target) navigate({ chrom: target.name, start: 0, end: target.length }).catch(report); });
$('minimap').addEventListener('click', e => { const length = source.manifest.contigs.find(c => c.name === state.chrom)!.length, center = e.offsetX / $<HTMLCanvasElement>('minimap').clientWidth * length, span = state.end - state.start; navigate({ chrom: state.chrom, start: center - span / 2, end: center + span / 2 }).catch(report); });
canvas.addEventListener('click', e => {
  if (!loaded?.exact) return;
  const rect = canvas.getBoundingClientRect(), pos = Math.round(state.start + (e.clientX - rect.left - margin(rect.width)) / (rect.width - margin(rect.width) - 14) * (state.end - state.start));
  inspectPosition(state.chrom,pos).catch(report);
});
$('browser').addEventListener('keydown', e => { if (!(e instanceof KeyboardEvent) || e.target !== $('browser')) return; if (['ArrowLeft', 'ArrowRight', '+', '=', '-', 'r', 'R'].includes(e.key)) e.preventDefault(); if (e.key === 'ArrowLeft') shift(-0.2).catch(report); if (e.key === 'ArrowRight') shift(0.2).catch(report); if (e.key === '+' || e.key === '=') zoom(0.5).catch(report); if (e.key === '-') zoom(2).catch(report); if (e.key.toLowerCase() === 'r') { state.roi = { chrom: state.chrom, start: state.start, end: state.end }; updateHash(); paint(); } });
$('sequence-form').addEventListener('submit', e => { e.preventDefault(); searchRequest?.abort(); searchRequest = new AbortController(); const signal = searchRequest.signal;
  const scope = select('sequence-scope').value; let requested: View | 'chromosome' | 'genome' = scope === 'genome' ? 'genome' : scope === 'chromosome' ? 'chromosome' : { chrom: state.chrom, start: state.start, end: state.end };
  if (scope === 'gene') { const gene = genes.find(g => g.name === state.gene && g.chrom === state.chrom) || genes.find(g => g.chrom === state.chrom && g.start < state.end && g.end > state.start); if (!gene) { $('sequence-status').textContent = 'Choose a gene locus or gene filter first'; return; } requested = { chrom: gene.chrom, start: gene.start, end: gene.end }; }
  $('sequence-status').textContent = 'Searching…'; $('sequence-results').innerHTML = ''; searchHits = []; hitCursor = -1;
  searchSequence(source, input('motif').value, requested, state, signal, text => { if (!signal.aborted) $('sequence-status').textContent = text; }).then(result => { if (signal.aborted) return; searchHits = result.hits; $('sequence-status').textContent = `${result.strandHits.toLocaleString()} strand hits; displaying ${result.hits.length}${result.truncated ? ' (limited to 200)' : ''}. Palindromes appear on both strands.`; $('sequence-results').innerHTML = searchHits.map((h, i) => `<li><button data-hit="${i}">${esc(formatView(h))} (${h.strand})</button></li>`).join(''); }).catch(error => { if (!signal.aborted) $('sequence-status').textContent = error.message; });
});
on('cancel-search', () => { searchRequest?.abort(); $('sequence-status').textContent = 'Search cancelled'; });
async function showHit(index: number) { if (!searchHits.length) return; hitCursor = (index + searchHits.length) % searchHits.length; const hit = searchHits[hitCursor]; $('hit-position').textContent = `${hitCursor + 1} / ${searchHits.length}`; await navigate({ chrom: hit.chrom, start: hit.start - 70, end: hit.end + 70 }, { roi: hit, selected: '' }); }
on('prev-hit', () => showHit(hitCursor - 1)); on('next-hit', () => showHit(hitCursor + 1));
$('sequence-results').addEventListener('click', e => { const b = (e.target as HTMLElement).closest<HTMLElement>('[data-hit]'); if (b) showHit(Number(b.dataset.hit)).catch(report); });
function openBookmark(e: Event) { const b = (e.target as HTMLElement).closest<HTMLElement>('[data-bookmark]'); if (!b) return; const view = source.manifest.bookmarks?.[Number(b.dataset.bookmark)]; if (view) navigate(view, { selected: '', gene: '' }).catch(report); }
$('example-loci').addEventListener('click', openBookmark);
$('demo-coverage').addEventListener('click', openBookmark);
on('allele-prev', () => { allelePage = Math.max(0, allelePage - 1); paint(); });
on('allele-next', () => { allelePage++; paint(); });
$('allele-rows').addEventListener('click', e => { const b = (e.target as HTMLElement).closest<HTMLElement>('[data-allele]'); if (b) { state.selected = b.dataset.allele!; updateHash(); paint(); $('variant-details').scrollIntoView({ block: 'nearest' }); } });
$('multilocus-form').addEventListener('submit', e => { e.preventDefault(); engine?.search(input('multilocus').value).then(async () => {
  const views = engine!.views; if (!views.length) return; activePane = 0;
  select('locus-pane').innerHTML = views.map((v, i) => `<option value="${i}">${esc(formatView(v))}</option>`).join('');
  await navigate(views[0], {}, true, false);
}).catch(report); });
select('locus-pane').addEventListener('change', () => { activePane = Number(select('locus-pane').value); const view = engine?.views[activePane]; if (view) navigate(view, {}, true, false).catch(report); });
on('save-session', async () => { if (!datasetReady) return; download(new Blob([JSON.stringify({ version: 2, manifest: source.manifestUrl, manifestSha256, activePane, state, igv: await engine!.session() }, null, 2)], { type: 'application/json' }), 'session.json'); });
input('session-file').addEventListener('change', async () => {
  try { const file = input('session-file').files?.[0]; if (!file) return; const saved = JSON.parse(await file.text());
    if (saved.version !== 2 || !saved.manifest || !saved.state || !saved.igv) throw new Error('Invalid OpenSpliceAI session');
    const url = new URL(snapshotLink(location.href, saved.manifest, saved.state)); const hash = new URLSearchParams(url.hash.slice(1)); hash.set('v', '2'); if (saved.manifestSha256) hash.set('manifestSha256',saved.manifestSha256); hash.set('pane',String(saved.activePane || 0)); hash.set('igv', JSON.stringify(saved.igv)); url.hash = hash.toString(); location.assign(url);
  } catch (error) { report(error); }
});
on('roi-bed', async () => { const rois = await engine!.rois(); const rows = rois.map(r => [r.chr, r.start, r.end, r.name || 'IGV region']); if (state.roi) rows.push([state.roi.chrom, state.roi.start, state.roi.end, 'Marked region']); download(new Blob([rows.map(r => r.join('\t')).join('\n') + '\n'], { type: 'text/plain' }), 'regions.bed'); });
$('add-track-form').addEventListener('submit', e => { e.preventDefault(); engine?.loadTrack(input('track-url').value, input('track-index').value || undefined).then(() => { $('track-status').textContent = 'Track loaded'; }).catch(report); });
input('track-files').addEventListener('change', async () => { try { const files = [...(input('track-files').files || [])]; const indexes = files.filter(f => /\.(bai|crai|tbi|csi|idx)$/i.test(f.name)); for (const file of files.filter(f => !indexes.includes(f))) { const index = indexes.find(i => i.name.startsWith(file.name) || i.name.replace(/\.(bai|crai)$/, '') === file.name.replace(/\.(bam|cram)$/, '')); await engine?.loadTrack(file, index); } $('track-status').textContent = 'Local tracks loaded; reselect these files after restoring a session'; } catch (error) { report(error); } });
window.addEventListener('hashchange', () => { if (!source) return; try { const restored = restoreState(location.hash, state, source.manifest.contigs); if (restored.snapshot !== source.manifest.id) throw new Error('Shared link requires another dataset snapshot'); state = restored; navigate(restored).catch(report); } catch (error) { report(error); } });
new ResizeObserver(() => paint()).observe(canvas.parentElement!);
window.addEventListener('pagehide', () => { viewRequest?.abort(); searchRequest?.abort(); engine?.destroy(); source?.destroy(); });
initialize().catch(error => { if (error.name === 'AbortError') return; $('dataset-note').classList.add('error'); $('dataset-note').textContent = error.message; report(error); });

input('roi-file').addEventListener('change',async()=>{try{
  const file=input('roi-file').files?.[0];if(!file)return;if(file.size>2*1048576)throw new Error('Region file exceeds 2 MiB');
  const features=(await file.text()).split(/\r?\n/).filter(line=>line.trim()&&!/^(#|track|browser)/.test(line)).map(line=>{const [chr,a,b,name]=line.split('\t');const start=Number(a),end=Number(b),contig=source.manifest.contigs.find(c=>c.name===chr);if(!contig||!Number.isSafeInteger(start)||!Number.isSafeInteger(end)||start<0||end<=start||end>contig.length)throw new Error('BED region has invalid contig or 0-based coordinates');return{chr,start,end,name};});
  if(features.length>10000)throw new Error('Region file exceeds 10,000 intervals');await engine?.importRois(features);updateHash();$('track-status').textContent=`Imported ${features.length} regions`;
}catch(error){report(error);}});
