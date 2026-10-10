import './style.css';
import { createSvgRecorder } from './canvasSvg';
import { DataSource, summaryTotals } from './data';
import { CHANNEL_COLORS, draw, layout, margin, type Palette } from './paint';
import { comparisonStatistics, csvCell, exactCsv, predictionStatus, visibleVariants } from './science';
import { searchSequence } from './search';
import { clampView, DEFAULT_STATE, EXACT_BP, formatView, History, orderedContigs, parseLocus, restoreState, serializeState, TRACKS, variantKey } from './state';
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
let loading: Promise<void> = Promise.resolve();
let searchHits: SearchHit[] = [], hitCursor = -1;
let catalog: { id: string; label: string; manifest: string }[] = [];
const descriptions: Record<string, string> = {
  genes: 'Exons, introns and strand from the scoring annotation', reference: 'Pinned GRCh38 reference bases', coverage: 'Accepted r13 source occurrences; pending is not zero',
  AG: 'Acceptor gain delta', AL: 'Acceptor loss delta', DG: 'Donor gain delta', DL: 'Donor loss delta', heatmap: 'Largest DS by ALT allele', difference: 'Difference in maximum DS for matched genes',
};
function palette(): Palette { const css = getComputedStyle(document.body); const get = (name: string) => css.getPropertyValue(`--${name}`).trim(); return { bg: get('panel'), ink: get('ink'), muted: get('muted'), line: get('line'), accent: get('accent'), compare: get('compare'), missing: get('missing') }; }
function report(error: unknown) { if (error instanceof Error && error.name === 'AbortError') return; const message = error instanceof Error ? error.message : String(error); $('load-status').textContent = message; $('retry').hidden = false; }
function on(id: string, action: () => void | Promise<void>) { $(id).addEventListener('click', () => { Promise.resolve().then(action).catch(report); }); }
function updateHash(push = true) { window.history.replaceState(null, '', `${location.pathname}${location.search}#${serializeState(state)}`); if (push) history.push(state); }
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
function paint() {
  if (!loaded) return;
  const width = Math.max(280, canvas.parentElement!.clientWidth), lanes = layout(state), height = (lanes.at(-1)?.y || 64) + (lanes.at(-1)?.height || 0) + 42;
  const dpr = Math.min(2, devicePixelRatio || 1);
  canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr); canvas.style.height = `${height}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); draw(ctx, width, state, loaded, loaded.genes || genes, palette());
  const variants = visibleVariants(loaded.variants, state);
  const maxima = new Map(variants.map(v => [v, v.models[state.model].reduce((m, a) => Math.max(m, ...a.ds), 0)]));
  const top = [...variants].sort((a, b) => maxima.get(b)! - maxima.get(a)!).slice(0, 400);
  const selected = loaded.variants.find(v => variantKey(v) === state.selected); if (selected && !top.includes(selected)) top.unshift(selected);
  select('variant-select').innerHTML = `<option value="">${loaded.exact ? `${variants.length.toLocaleString()} alleles; showing top ${Math.min(400, variants.length)}` : 'Zoom in for exact alleles'}</option>` + top.map(v => `<option value="${esc(variantKey(v))}">${esc(variantKey(v))} · ${esc(predictionStatus(v, state.model))}</option>`).join('');
  select('variant-select').value = state.selected;
  renderInspector(selected); paintOverview(); paintScatter(variants);
  $('filter-note').hidden = loaded.exact;
  for (const id of ['alt', 'gene-filter', 'threshold']) (document.getElementById(id) as HTMLInputElement).disabled = !loaded.exact;
  $('gene-details').innerHTML = (loaded.genes || genes).filter(g => g.chrom === state.chrom && g.start < state.end && g.end > state.start).slice(0, 30).map(g => `<div class="prediction"><h3>${esc(g.name)} · ${g.strand} strand</h3><p class="caption">${esc(formatView(g))} · ${g.exons.length} exons</p>${g.exons.map(([a, b], i) => `<button data-exon-start="${a}" data-exon-end="${b}" aria-label="Navigate to ${esc(g.name)} exon ${g.strand === '+' ? i + 1 : g.exons.length - i}">Exon ${g.strand === '+' ? i + 1 : g.exons.length - i}: ${(a + 1).toLocaleString()}–${b.toLocaleString()}</button>`).join('')}</div>`).join('') || '<p class="caption">No scoring annotation gene overlaps this view.</p>';
  const r = source.resources; $('resource-status').textContent = `${(r.transferred / 1048576).toFixed(2)} MiB received after HTTP decoding · ${(r.cache.bytes / 1048576).toFixed(1)} MiB cache · ${r.requests} requests`;
  $<HTMLButtonElement>('csv').disabled = !loaded.exact;
}
async function navigate(view: View, changes: Partial<BrowserState> = {}, push = true) {
  state = { ...state, ...changes, ...clampView(view, source.manifest.contigs) };
  syncControls(); updateHash(push); await refresh();
}
function refresh(): Promise<void> {
  viewRequest?.abort(); viewRequest = new AbortController(); const signal = viewRequest.signal, local = ++epoch;
  successfulEpoch = -1; $('retry').hidden = true; $('browser').classList.add('loading'); $('load-status').textContent = 'Loading data…';
  loaded = null;
  // Clear old pixels and inspection immediately: an old locus must not look like new data.
  ctx.clearRect(0, 0, canvas.width, canvas.height); renderInspector(); select('variant-select').innerHTML = '<option>Loading exact alleles…</option>';
  loading = source.load({ chrom: state.chrom, start: state.start, end: state.end }, signal).then(result => {
    if (local !== epoch || signal.aborted) return;
    loaded = result; successfulEpoch = local;
    const totals = summaryTotals(result.summaries);
    $('load-status').textContent = `${result.exact ? 'Exact' : 'Binned maxima'} · ${totals.rows.toLocaleString()} source occurrences in overlapping bins · r13 ${totals.rows ? (100 * totals.accepted / totals.rows).toFixed(1) : '—'}% accepted`;
    paint();
  }).catch(error => { if (local === epoch && !signal.aborted) report(error); }).finally(() => { if (local === epoch) $('browser').classList.remove('loading'); });
  return loading;
}
function renderInspector(v?: Variant) {
  if (!v) { $('variant-details').innerHTML = '<p class="caption">Select an exact allele. All source annotations will appear here, including zero scores and conflicts.</p>'; return; }
  const contig = source.manifest.contigs.find(c => c.name === v.chrom)!;
  let html = `<div class="variant-title">${esc(variantKey(v))}</div><span class="badge">${v.occurrences.length} source occurrence${v.occurrences.length === 1 ? '' : 's'}</span><span class="badge">${v.r13Accepted ? 'r13 accepted evidence' : 'r13 preview pending'}</span>`;
  if (v.refMismatch) html += '<p class="warning caption">REF differs from the pinned FASTA. Preserved for inspection; excluded from ordinary model comparisons.</p>';
  for (const model of ['r10', 'r13', 'baseline'] as const) {
    html += `<div class="prediction"><h3>${esc(source.manifest.models[model].label)} <span class="badge">${esc(predictionStatus(v, model))}</span></h3>`;
    if (!v.models[model].length) html += `<p class="caption">${model === 'r13' && !v.r13Accepted ? 'This source occurrence is outside the accepted preview snapshot.' : 'No prediction annotation was recorded. This is an absent value, not a delta of zero.'}</p>`;
    for (const a of v.models[model]) {
      html += `<p><strong>${esc(a.gene)}</strong> <span class="caption">${a.occurrences.length} occurrence${a.occurrences.length === 1 ? '' : 's'}</span></p><table><thead><tr><th>Event</th><th>DS</th><th>DP</th><th>Site</th></tr></thead><tbody>`;
      for (let c = 0; c < 4; c++) { const site = v.pos + 1 + a.dp[c]; const nonzero = a.ds[c] > 0; html += `<tr><td>${['AG', 'AL', 'DG', 'DL'][c]}</td><td>${a.ds[c].toFixed(source.manifest.models[model].precision)}</td><td>${a.dp[c] > 0 ? '+' : ''}${a.dp[c]}</td><td>${nonzero && site >= 1 && site <= contig.length ? `<button class="site-link" data-site="${site}">${site.toLocaleString()}</button>` : nonzero ? 'outside contig' : '—'}</td></tr>`; }
      html += '</tbody></table>';
    }
    if (new Set(v.models[model].map(a => a.gene)).size < v.models[model].length) html += '<p class="warning caption">Distinct annotations for the same gene are preserved; this gene is excluded from the comparison.</p>';
    html += '</div>';
  }
  html += `<details><summary>Source identities</summary><p class="caption">chunk:row ${esc(v.occurrences.join(', '))}</p></details>`;
  $('variant-details').innerHTML = html;
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
function paintScatter(variants: Variant[]) {
  const c = $<HTMLCanvasElement>('scatter'), cx = c.getContext('2d')!, p = palette(), result = comparisonStatistics(variants, Number(select('stats-event').value));
  cx.clearRect(0, 0, c.width, c.height); cx.fillStyle = p.bg; cx.fillRect(0, 0, c.width, c.height); cx.strokeStyle = p.line; cx.beginPath(); cx.moveTo(40, 10); cx.lineTo(40, 194); cx.lineTo(340, 194); cx.moveTo(40, 194); cx.lineTo(340, 10); cx.stroke();
  cx.fillStyle = p.accent; cx.globalAlpha = 0.5;
  // Deterministic pixel occupancy avoids millions of overdraws in dense loci.
  const pixels = new Set(result.points.map(point => `${Math.round(40 + point.x * 300)}:${Math.round(194 - point.y * 184)}`));
  for (const pixel of pixels) { const [x, y] = pixel.split(':').map(Number); cx.fillRect(x - 1, y - 1, 3, 3); }
  cx.globalAlpha = 1; cx.fillStyle = p.muted; cx.font = '11px system-ui'; cx.fillText('r10 delta score →', 140, 220); cx.fillText('r13', 3, 14);
  $('stats').textContent = loaded?.exact ? `${result.n.toLocaleString()} matched allele/gene pairs · mean r13−r10 ${result.meanDifference?.toFixed(5) ?? '—'} · Pearson r ${result.correlation?.toFixed(3) ?? 'undefined (no variance or no matches)'}` : 'Zoom in for exact matched allele/gene statistics. Wide-view maxima are not used as exact comparisons.';
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
async function stableExport(action: (loaded: LoadedView) => void | Promise<void>) {
  const current = epoch; await loading;
  if (!loaded || successfulEpoch !== current || current !== epoch) throw new Error('Export requires a successfully loaded, stable view. Retry the data or wait for the new locus.');
  await action(loaded);
  if (current !== epoch) throw new Error('View changed during export');
}
function download(blob: Blob, extension: string) { const a = document.createElement('a'); const url = URL.createObjectURL(blob); a.href = url; a.download = `OpenSpliceAI-${state.snapshot}-${state.chrom}-${state.start + 1}-${state.end}.${extension}`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 30000); }
async function loadDataset(url: string, restore = true) {
  datasetRequest?.abort(); datasetRequest = new AbortController();
  const datasetSignal = datasetRequest.signal, thisDataset = ++datasetEpoch;
  datasetReady = false;
  history.clear(); loaded = null; successfulEpoch = -1;
  ctx.clearRect(0, 0, canvas.width, canvas.height); renderInspector();
  viewRequest?.abort(); searchRequest?.abort(); epoch++; source?.destroy();
  $('dataset-note').textContent = 'Loading dataset information…';
  const manifestUrl = new URL(url, location.href).href;
  const response = await fetch(manifestUrl, { signal: datasetSignal }); if (!response.ok) throw new Error(`Dataset manifest failed (${response.status})`);
  const manifest = await response.json() as Manifest;
  const nextSource = new DataSource(manifest, manifestUrl);
  const result = await Promise.all([nextSource.genes(datasetSignal), nextSource.overview(datasetSignal)]).catch(error => { nextSource.destroy(); throw error; });
  if (thisDataset !== datasetEpoch || datasetSignal.aborted) { nextSource.destroy(); return; }
  source = nextSource;
  genes = result[0];
  const fallback = { ...DEFAULT_STATE, ...manifest.defaultView, snapshot: manifest.id };
  state = restore ? restoreState(location.hash, fallback, manifest.contigs) : fallback;
  if (state.snapshot !== manifest.id) throw new Error('The shared link identifies a different immutable snapshot. Choose that dataset before restoring the link.');
  $('dataset-note').innerHTML = `<strong>${esc(manifest.label)}</strong> · ${esc(manifest.scope === 'review-subset' ? 'Real scored review subset; genome-wide data hosting is not configured.' : 'Frozen scored-source collection')}</strong><br>r10 is the default. r13 ${manifest.r13Final ? 'passed its final publication audit' : 'is a preview with visible gaps'} · snapshot ${esc(manifest.id)} · ${esc(manifest.created.split('T')[0])}. ${esc(manifest.r13Evidence)}`;
  select('chromosome').innerHTML = orderedContigs(manifest.contigs).map(c => `<option value="${esc(c.name)}">${esc(c.name)} · ${(c.length / 1e6).toFixed(2)} Mb</option>`).join('');
  $('gene-list').innerHTML = genes.map(g => `<option value="${esc(g.name)}"></option>`).join('');
  $('provenance').innerHTML = `<p>Snapshot <code>${esc(manifest.id)}</code> · ${esc(manifest.sourceOccurrences.toLocaleString())} prepared source occurrences, ${esc(manifest.acceptedR13Occurrences.toLocaleString())} with accepted r13 evidence.</p><p>Reference SHA-256 <code>${esc(manifest.referenceSha256)}</code><br>Annotation SHA-256 <code>${esc(manifest.annotationSha256)}</code></p>` + Object.entries(manifest.models).map(([id, m]) => `<p>${esc(id)}: ${esc(m.label)}; ${m.precision}-decimal source precision${m.seed !== undefined ? `; random seed ${m.seed}` : ''}${m.sha256 ? `<br><code>${esc(m.sha256)}</code>` : ''}</p>`).join('');
  datasetReady = true; history.push(state); syncControls(); await refresh();
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
for (const id of ['theme', 'model', 'density', 'alt']) select(id).addEventListener('change', () => { state = { ...state, [id]: select(id).value }; document.body.dataset.theme = state.theme; updateHash(); paint(); });
for (const id of ['compare', 'autoscale']) input(id).addEventListener('change', () => { state = { ...state, [id]: input(id).checked }; updateHash(); paint(); });
input('threshold').addEventListener('change', () => { state.threshold = Math.min(1, Math.max(0, Number(input('threshold').value) || 0)); updateHash(); paint(); });
input('gene-filter').addEventListener('change', () => { state.gene = input('gene-filter').value.trim(); updateHash(); paint(); });
input('track-filter').addEventListener('input', trackControls);
$('track-list').addEventListener('change', e => { const t = e.target as HTMLInputElement; if (t.dataset.track) state.tracks = t.checked ? [...state.tracks, t.dataset.track] : state.tracks.filter(id => id !== t.dataset.track); if (t.dataset.height) state.heights = { ...state.heights, [t.dataset.height]: Math.max(32, Math.min(240, Number(t.value) || 58)) }; updateHash(); paint(); });
$('track-list').addEventListener('click', e => { const b = (e.target as HTMLElement).closest<HTMLButtonElement>('[data-move]'); if (!b) return; const tracks = [...state.tracks], from = tracks.indexOf(b.dataset.move!), to = from + Number(b.dataset.dir); if (from < 0 || to < 0 || to >= tracks.length) return; [tracks[from], tracks[to]] = [tracks[to], tracks[from]]; state.tracks = tracks; updateHash(); trackControls(); paint(); });
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
on('share', async () => { await navigator.clipboard.writeText(location.href); $('load-status').textContent = 'Snapshot and complete view link copied'; });
on('mark-roi', () => { state.roi = { chrom: state.chrom, start: state.start, end: state.end }; updateHash(); paint(); });
on('clear-roi', () => { state.roi = null; updateHash(); paint(); });
for (const direction of ['prev', 'next']) on(`${direction}-gene`, () => { const list = genes.filter(g => g.chrom === state.chrom).sort((a, b) => a.start - b.start); const center = (state.start + state.end) / 2; const gene = direction === 'next' ? list.find(g => g.start > center) : [...list].reverse().find(g => g.end < center); if (gene) return goTo(gene.name); });
on('tracks-open', () => { $('track-panel').classList.toggle('open'); $('tracks-open').setAttribute('aria-expanded', String($('track-panel').classList.contains('open'))); });
on('tracks-close', () => { $('track-panel').classList.remove('open'); $('tracks-open').setAttribute('aria-expanded', 'false'); $('tracks-open').focus(); });
on('svg', () => stableExport(data => { const width = canvas.clientWidth, recorder = createSvgRecorder(ctx); recorder.ctx.setTransform(1, 0, 0, 1, 0, 0); const height = draw(recorder.ctx, width, state, data, data.genes || genes, palette()); download(new Blob([recorder.svg(width, height, palette().bg, `OpenSpliceAI ${state.snapshot} ${formatView(state)}`)], { type: 'image/svg+xml' }), 'svg'); paint(); }));
on('png', () => stableExport(async data => { const exportEpoch = epoch, width = canvas.clientWidth, offscreen = document.createElement('canvas'); const height = (layout(state).at(-1)?.y || 64) + (layout(state).at(-1)?.height || 0) + 42; offscreen.width = width * 2; offscreen.height = height * 2; const cx = offscreen.getContext('2d')!; cx.scale(2, 2); draw(cx, width, state, data, data.genes || genes, palette()); const blob = await new Promise<Blob>((resolve, reject) => offscreen.toBlob(b => b ? resolve(b) : reject(new Error('PNG encoding failed')), 'image/png')); if (exportEpoch !== epoch) throw new Error('View changed during figure encoding'); download(blob, 'png'); }));
on('csv', () => stableExport(data => { if (!data.exact) throw new Error('Zoom in for exact CSV'); download(new Blob([exactCsv(visibleVariants(data.variants, state), state.snapshot)], { type: 'text/csv' }), 'exact.csv'); }));
on('summary-csv', () => stableExport(data => { const rows: unknown[][] = [['snapshot', 'chrom', 'bin_start_0based', 'bin_end_0based', 'source_occurrences', 'accepted_r13_occurrences', 'ref_mismatch_occurrences', 'model', 'annotation_entries', 'zero_annotations', 'max_DS_AG', 'max_DS_AL', 'max_DS_DG', 'max_DS_DL']]; for (const s of data.summaries) for (const [model, stats] of Object.entries(s.models)) rows.push([state.snapshot, state.chrom, s.start, s.end, s.rows, s.acceptedR13, s.refMismatch, model, stats!.annotations, stats!.zero, ...stats!.max.map(v => (v / 100000).toFixed(5))]); download(new Blob([rows.map(r => r.map(csvCell).join(',')).join('\r\n')], { type: 'text/csv' }), 'summary.csv'); }));
select('stats-event').addEventListener('change', () => { if (loaded) paintScatter(visibleVariants(loaded.variants, state)); });
$('genome-overview').addEventListener('click', e => { const c = orderedContigs(source.manifest.contigs).filter(c => /^chr(\d+|X|Y|M)$/.test(c.name)), target = c[Math.min(c.length - 1, Math.floor((e.offsetX / $<HTMLCanvasElement>('genome-overview').clientWidth) * c.length))]; if (target) navigate({ chrom: target.name, start: 0, end: target.length }).catch(report); });
$('minimap').addEventListener('click', e => { const length = source.manifest.contigs.find(c => c.name === state.chrom)!.length, center = e.offsetX / $<HTMLCanvasElement>('minimap').clientWidth * length, span = state.end - state.start; navigate({ chrom: state.chrom, start: center - span / 2, end: center + span / 2 }).catch(report); });
let drag: { x: number; y: number; view: View; shift: boolean; pointer: number } | null = null;
canvas.addEventListener('pointerdown', e => { if (e.button !== 0 || !loaded) return; drag = { x: e.clientX, y: e.clientY, view: { chrom: state.chrom, start: state.start, end: state.end }, shift: e.shiftKey, pointer: e.pointerId }; if (e.pointerType === 'mouse') canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointercancel', () => { drag = null; });
canvas.addEventListener('pointermove', e => { if (!loaded || drag) return; const rect = canvas.getBoundingClientRect(), pos = Math.floor(state.start + (e.clientX - rect.left - margin(rect.width)) / (rect.width - margin(rect.width) - 14) * (state.end - state.start)); const v = loaded.variants.find(v => v.pos === pos); if (v) { $('tooltip').textContent = `${variantKey(v)} · ${predictionStatus(v, state.model)}`; $('tooltip').hidden = false; $('tooltip').style.left = `${Math.min(rect.width - 220, Math.max(0, e.clientX - rect.left))}px`; $('tooltip').style.top = `${e.clientY - rect.top + 35}px`; } else $('tooltip').hidden = true; });
canvas.addEventListener('pointerleave', () => { $('tooltip').hidden = true; });
canvas.addEventListener('pointerup', e => { if (!drag || !loaded) return; const d = drag; drag = null; const rect = canvas.getBoundingClientRect(), plot = rect.width - margin(rect.width) - 14, delta = e.clientX - d.x;
  if (Math.abs(e.clientY - d.y) > Math.abs(delta) && e.pointerType !== 'mouse') return;
  if (Math.abs(delta) > 6) { if (d.shift) { const a = d.view.start + (d.x - rect.left - margin(rect.width)) / plot * (d.view.end - d.view.start), b = d.view.start + (e.clientX - rect.left - margin(rect.width)) / plot * (d.view.end - d.view.start); state.roi = clampView({ chrom: state.chrom, start: Math.min(a, b), end: Math.max(a, b) }, source.manifest.contigs, 1); updateHash(); paint(); } else { const bases = delta / plot * (d.view.end - d.view.start); navigate({ chrom: d.view.chrom, start: d.view.start - bases, end: d.view.end - bases }).catch(report); } }
  else { const pos = d.view.start + (e.clientX - rect.left - margin(rect.width)) / plot * (d.view.end - d.view.start); const candidates = visibleVariants(loaded.variants, state).filter(v => Math.abs(v.pos - pos) < Math.max(1, (state.end - state.start) / plot * 5)); const v = candidates.sort((a, b) => Math.abs(a.pos - pos) - Math.abs(b.pos - pos))[0]; if (v) { state.selected = variantKey(v); updateHash(); paint(); } else if (!loaded.exact) navigate({ chrom: state.chrom, start: pos - 500, end: pos + 500 }).catch(report); }
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
window.addEventListener('hashchange', () => { if (!source) return; try { const restored = restoreState(location.hash, state, source.manifest.contigs); if (restored.snapshot !== source.manifest.id) throw new Error('Shared link requires another dataset snapshot'); state = restored; navigate(restored).catch(report); } catch (error) { report(error); } });
new ResizeObserver(() => paint()).observe(canvas.parentElement!);
window.addEventListener('pagehide', () => { viewRequest?.abort(); searchRequest?.abort(); source?.destroy(); });
initialize().catch(error => { if (error.name === 'AbortError') return; $('dataset-note').classList.add('error'); $('dataset-note').textContent = error.message; report(error); });
