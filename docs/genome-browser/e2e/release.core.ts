import { expect, test, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';

const catalog = JSON.parse(await readFile(new URL('../public/catalog.json', import.meta.url), 'utf8'));
const manifestPath = catalog.datasets[0].manifest.replace(/^\.\//, '');
const manifest = JSON.parse(await readFile(new URL('../public/' + manifestPath, import.meta.url), 'utf8'));
const errors = new WeakMap<Page, string[]>();
const bookmarkIndex = (gene: string) => manifest.bookmarks.findIndex((b: {label: string}) => b.label.startsWith(gene + ' ·'));

async function ready(page: Page) {
  await expect(page.locator('#igv-browser')).toHaveAttribute('data-ready', 'true');
  await expect(page.locator('#load-status')).toContainText('Exact');
  await expect(page.locator('#browser')).not.toHaveClass(/loading/);
}
async function locus(page: Page, value: string) {
  await page.locator('#locus').fill(value);
  await page.locator('#locus-form button[type=submit]').click();
}
async function download(page: Page, id: string) {
  const wait = page.waitForEvent('download');
  await page.locator('#' + id).click();
  const file = await wait, chunks: Buffer[] = [];
  for await (const chunk of (await file.createReadStream())!) chunks.push(chunk);
  return Buffer.concat(chunks);
}
async function example(page: Page, gene: string) {
  const index = bookmarkIndex(gene);
  expect(index).toBeGreaterThanOrEqual(0);
  await page.locator(`#example-loci [data-bookmark="${index}"]`).click();
  await expect(page.locator('#view-label')).toContainText(manifest.bookmarks[index].chrom + ':');
  await ready(page);
  await expect(page.locator('#gene-details')).toContainText(gene);
}

test.beforeEach(async ({ page }) => {
  const messages: string[] = []; errors.set(page, messages);
  page.on('pageerror', error => messages.push(error.message));
  await page.goto('./');
  await ready(page);
});
test.afterEach(async ({ page }) => { expect(errors.get(page)).toEqual([]); });

test('load native tracks and inspect real r10/r13 scores in every added example', async ({ page }) => {
  await expect(page.locator('#dataset-note')).toContainText('10 scored regions');
  await expect(page.locator('#example-loci button')).toHaveCount(10);
  expect(await page.locator('#igv-browser canvas').count()).toBeGreaterThan(7);
  await locus(page, 'chr1:69091:A>G');
  await expect(page.locator('#variant-details')).toContainText('OR4F5');
  await expect(page.locator('#variant-details')).toContainText('0.00009');
  await expect(page.locator('#variant-details')).toContainText('scored zero');
  for (const gene of ['CFTR', 'HBB', 'BRCA1', 'BRCA2']) {
    await example(page, gene);
    const value = await page.locator('#variant-select option').nth(1).getAttribute('value');
    expect(value).toBeTruthy();
    await page.locator('#variant-select').selectOption(value!);
    await expect(page.locator('#variant-details')).toContainText(gene);
    await expect(page.locator('#variant-details')).toContainText('OpenSpliceAI MANE r10');
    await expect(page.locator('#variant-details')).toContainText('OpenSpliceAI MANE r13');
  }
});

test('pan continuously, zoom with the ruler and export a selected BED region', async ({ page }) => {
  const viewport = page.locator('#igv-browser .igv-viewport:has(canvas):visible').nth(4);
  await viewport.scrollIntoViewIfNeeded();
  const canvas = viewport.locator('canvas').first(), box = (await viewport.boundingBox())!;
  const before = await canvas.evaluate(el => el.style.left), oldLocus = await page.locator('#view-label').textContent();
  await page.mouse.move(box.x + box.width * .6, box.y + 10); await page.mouse.down();
  await page.mouse.move(box.x + box.width * .3, box.y + 10, { steps: 8 });
  expect(await canvas.evaluate(el => el.style.left)).not.toBe(before);
  await page.mouse.up();
  await expect(page.locator('#view-label')).not.toHaveText(oldLocus!); await ready(page);
  const ruler = page.locator('#igv-browser .igv-viewport:has(.igv-ruler-tooltip):visible').first();
  await ruler.scrollIntoViewIfNeeded();
  const old = await page.locator('#view-label').textContent(); let r = (await ruler.boundingBox())!;
  await page.mouse.move(r.x + r.width * .3, r.y + r.height * .6); await page.mouse.down();
  await page.mouse.move(r.x + r.width * .7, r.y + r.height * .6, {steps: 8}); await page.mouse.up();
  await expect(page.locator('#view-label')).not.toHaveText(old!); await ready(page);
  r = (await ruler.boundingBox())!;
  await page.keyboard.down('Shift');
  await page.mouse.move(r.x + r.width * .3, r.y + r.height * .6); await page.mouse.down();
  await page.mouse.move(r.x + r.width * .7, r.y + r.height * .6, {steps: 8}); await page.mouse.up();
  await page.keyboard.up('Shift');
  const bed = (await download(page, 'roi-bed')).toString().trim().split('\n')[0].split('\t');
  expect(bed[0]).toBe('chr1'); expect(Number(bed[1])).toBeLessThan(Number(bed[2]));
});

test('switch models and download exact CSV, PNG and SVG', async ({ page }) => {
  await locus(page, 'chr1:69091:A>G'); await ready(page);
  await page.locator('#model').selectOption('r13');
  await page.locator('#compare').uncheck(); await page.locator('#compare').check();
  await expect(page.locator('#variant-details')).toContainText('0.00009');
  const csv = (await download(page, 'csv')).toString();
  expect(csv).toContain('DS_AG,DS_AL,DS_DG,DS_DL'); expect(csv).toContain('0.00009');
  expect(csv).toContain(manifest.id); expect(csv.trim().split('\n').length).toBeGreaterThan(100);
  expect((await download(page, 'svg')).toString()).toContain('<svg');
  expect((await download(page, 'png')).subarray(0,8)).toEqual(Buffer.from([137,80,78,71,13,10,26,10]));
});

test('switch inspector panes and restore a pinned session', async ({ page }) => {
  await page.locator('#multilocus').fill('chr1:925850-926130 chr1:69040-70060');
  await page.locator('#multilocus-form button').click();
  await expect(page.locator('#locus-pane option')).toHaveCount(2);
  await page.locator('#locus-pane').selectOption('1');
  await expect(page.locator('#view-label')).toContainText('69,040'); await ready(page);
  await example(page, 'CFTR');
  await expect(page.locator('#locus-pane option')).toHaveCount(1);
  await expect(page.locator('#locus-pane')).toHaveValue('0');
  await page.locator('#model').selectOption('r13');
  const bytes = await download(page, 'save-session'), saved = JSON.parse(bytes.toString());
  expect(saved.version).toBe(2); expect(saved.state.snapshot).toBe(manifest.id);
  expect(saved.manifest).toBe(new URL(manifestPath,page.url()).href);
  expect(saved.manifestSha256).toMatch(/^[a-f0-9]{64}$/);
  await page.locator('#session-file').setInputFiles({name:'session.json',mimeType:'application/json',buffer:bytes});
  await page.waitForURL(url => url.searchParams.has('manifest')); await ready(page);
  await expect(page.locator('#model')).toHaveValue('r13');
  await expect(page.locator('#view-label')).toContainText('chr7:');
});

test('recover from data errors and operate core controls on a phone', async ({ page }) => {
  await page.route('**/*.pack', route => route.fulfill({status:503,body:'unavailable'}));
  await locus(page, 'chr2_KI270773v1_alt:18694:C>A');
  await expect(page.locator('#retry')).toBeVisible();
  await expect(page.locator('#load-status')).toContainText('failed');
  await page.unroute('**/*.pack'); await page.locator('#retry').click(); await ready(page);
  const index = bookmarkIndex('CFTR'), r = manifest.bookmarks[index];
  await locus(page, `${r.chrom}:${r.start - 199}-${r.end + 200}`);
  await expect(page.locator('#demo-coverage')).toHaveAttribute('data-coverage','partial');
  await page.locator(`#demo-coverage [data-bookmark="${index}"]`).click(); await ready(page);
  await expect(page.locator('#demo-coverage')).toHaveAttribute('data-coverage','full');
  await page.setViewportSize({width:320,height:812});
  for (const selector of ['#locus','#locus-form button[type=submit]','#model','#tracks-open']) {
    const control = page.locator(selector); await control.scrollIntoViewIfNeeded();
    await page.evaluate(() => window.scrollTo(0,window.scrollY));
    await expect(control).toBeVisible(); const box = (await control.boundingBox())!;
    expect(box.x).toBeGreaterThanOrEqual(-1); expect(box.x + box.width).toBeLessThanOrEqual(321);
  }
  await page.locator('#tracks-open').click(); await expect(page.locator('#track-panel')).toBeVisible();
  await page.locator('#tracks-close').click(); await expect(page.locator('#track-panel')).not.toBeVisible();
  await page.locator('#model').selectOption('r10');
  await locus(page,'chr1:69091:A>G'); await expect(page.locator('#variant-details')).toContainText('0.00009');
  await locus(page,'chr3:1-100');
  await expect(page.locator('#load-status')).toContainText('Outside published demonstration');
  await expect(page.locator('#csv')).toBeDisabled(); await expect(page.locator('#summary-csv')).toBeDisabled();
  await expect(page.locator('#variant-details')).not.toContainText('scored zero');
});
