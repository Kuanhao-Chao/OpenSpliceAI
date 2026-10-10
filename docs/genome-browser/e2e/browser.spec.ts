import { expect, test, type Page } from '@playwright/test';
import { writeFile } from 'node:fs/promises';

const scriptErrors = new WeakMap<Page, string[]>();

test.beforeEach(async ({ page }) => {
  const errors: string[] = [];
  scriptErrors.set(page, errors);
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('./');
  await expect(page.locator('#load-status')).toContainText('Exact', { timeout: 30000 });
});

test.afterEach(async ({ page }) => { expect(scriptErrors.get(page)).toEqual([]); });

test('real scores, model changes, missing values and source precision', async ({ page }) => {
  await page.locator('#locus').fill('chr1:69091:A>G'); await page.locator('#locus-form button[type=submit]').click();
  await expect(page.locator('#variant-details')).toContainText('0.00009');
  await expect(page.locator('#variant-details')).toContainText('OR4F5');
  await page.locator('#model').selectOption('r13');
  await expect(page.locator('#variant-details')).toContainText('scored zero');
  await page.locator('#locus').fill('chr2_KI270773v1_alt:18694:C>A'); await page.locator('#locus-form button[type=submit]').click();
  await expect(page.locator('#variant-details')).toContainText('REF differs');
  await expect(page.locator('#variant-details')).toContainText('Distinct annotations');
  await expect(page.locator('#variant-details')).toContainText('prediction absent');
});

test('state, navigation, filters, empty track selection and themes', async ({ page }) => {
  await page.locator('#theme').selectOption('nord');
  await page.locator('#alt').selectOption('G');
  await page.locator('#threshold').fill('0.25'); await page.locator('#threshold').dispatchEvent('change');
  await page.locator('#mark-roi').click();
  for (const checkbox of await page.locator('#track-list input[type=checkbox]').all()) await checkbox.uncheck();
  const before = await page.evaluate(() => location.hash);
  await page.reload(); await expect(page.locator('#load-status')).toContainText('Exact');
  await expect(page.locator('body')).toHaveAttribute('data-theme', 'nord');
  await expect(page.locator('#alt')).toHaveValue('G');
  await expect(page.locator('#track-list input[type=checkbox]:checked')).toHaveCount(0);
  expect(await page.evaluate(() => location.hash)).toBe(before);
  for (const theme of ['light', 'dark', 'nord', 'monokai', 'cyberdeck', 'parchment']) { await page.locator('#theme').selectOption(theme); await expect(page.locator('body')).toHaveAttribute('data-theme', theme); }
  await page.locator('#zoom-in').click(); await expect(page.locator('#load-status')).toContainText('Exact');
  await page.locator('#back').click(); await expect(page.locator('#load-status')).toContainText('Exact');
});

test('PNG, vector SVG, exact CSV and summary CSV download', async ({ page }) => {
  for (const id of ['png', 'svg', 'csv', 'summary-csv']) {
    const download = page.waitForEvent('download'); await page.locator(`#${id}`).click(); const file = await download;
    const stream = await file.createReadStream(); const chunks = []; for await (const chunk of stream!) chunks.push(chunk); const content = Buffer.concat(chunks);
    expect(content.length).toBeGreaterThan(100);
    if (id === 'svg') { expect(content.toString()).toContain('<svg'); expect(content.toString()).not.toContain('<image'); expect(content.toString()).toContain('GRCh38'); }
    if (id === 'csv') expect(content.toString()).toContain('DS_AG,DS_AL,DS_DG,DS_DL');
    if (id === 'summary-csv') expect(content.toString()).toContain('annotation_entries');
  }
});

test('sequence search, cancellation and honest unavailable genome index', async ({ page }) => {
  await page.locator('#motif').fill('GT'); await page.locator('#sequence-form button[type=submit], #sequence-form button.primary').click();
  await expect(page.locator('#sequence-status')).toContainText('strand hits');
  expect(await page.locator('#sequence-results li').count()).toBeGreaterThan(0);
  await page.locator('#sequence-scope').selectOption('genome');
  await page.locator('#sequence-form button.primary').click();
  await expect(page.locator('#sequence-status')).toContainText('not published');
  await page.locator('#cancel-search').click(); await expect(page.locator('#sequence-status')).toContainText('cancelled');
});

test('failed ranges are visible and prevent exports, then retry succeeds', async ({ page }) => {
  await page.route('**/*.pack', route => route.fulfill({ status: 503, body: 'unavailable' }));
  await page.locator('#locus').fill('chr2_KI270773v1_alt:18694:C>A'); await page.locator('#locus-form button[type=submit]').click();
  await expect(page.locator('#retry')).toBeVisible();
  await expect(page.locator('#load-status')).toContainText('failed');
  await page.locator('#svg').click(); await expect(page.locator('#load-status')).toContainText('successfully loaded');
  await page.unroute('**/*.pack'); await page.locator('#retry').click(); await expect(page.locator('#load-status')).toContainText('Exact');
});

test('responsive layouts have no overflow and retain accessible controls', async ({ page }, testInfo) => {
  for (const [name, width, height] of [['phone', 320, 812], ['tablet', 768, 1024], ['laptop', 1440, 900]] as const) {
    await page.setViewportSize({ width, height });
    await page.waitForTimeout(150);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1); expect(overflow, name).toBe(false);
    if (width < 1024) { await page.locator('#tracks-open').click(); await expect(page.locator('#track-panel')).toBeVisible(); await page.locator('#tracks-close').click(); }
    await page.screenshot({ path: testInfo.outputPath(`${name}.png`), fullPage: true });
  }
});

test('wide views use summaries and refuse exact CSV', async ({ page }) => {
  await page.locator('#whole').click(); await expect(page.locator('#load-status')).toContainText('Binned');
  await expect(page.locator('#csv')).toBeDisabled();
  await expect(page.locator('#stats')).toContainText('Zoom in');
});

test('paged FM search agrees with the actual mitochondrial reference', async ({ page }) => {
  await page.locator('#locus').fill('chrM:1-20'); await page.locator('#locus-form button[type=submit]').click();
  await expect(page.locator('#view-label')).toContainText('chrM');
  await expect(page.locator('#load-status')).toContainText('Exact');
  await page.locator('#motif').fill('GATCACAGGTCTATCACCCT');
  await page.locator('#sequence-scope').selectOption('chromosome');
  await page.route('**/search-chrM.*', async route => {
    await new Promise(resolve => setTimeout(resolve, 500));
    await route.continue().catch(() => {}); // Cancellation can close this request.
  });
  await page.locator('#sequence-form button.primary').click();
  await page.locator('#cancel-search').click();
  await expect(page.locator('#sequence-status')).toContainText('cancelled');
  await page.waitForTimeout(650);
  await expect(page.locator('#sequence-status')).toContainText('cancelled');
  await page.unroute('**/search-chrM.*');
  await page.locator('#sequence-form button.primary').click();
  await expect(page.locator('#sequence-status')).toContainText('strand hits', { timeout: 30000 });
  await expect(page.locator('#sequence-results')).toContainText('chrM:1-20');
});

test('default view stays within the small initial data budget and has no script errors', async ({ page }, testInfo) => {
  const metrics = await page.evaluate(() => performance.getEntriesByType('resource').filter(e => /\/data\//.test(e.name)).map(e => ({ url: e.name, encodedBytes: (e as PerformanceResourceTiming).encodedBodySize })));
  const total = metrics.reduce((n, m) => n + m.encodedBytes, 0);
  await writeFile(testInfo.outputPath('startup-budget.json'), JSON.stringify({ encodedDataBytes: total, resources: metrics }, null, 2));
  expect(total).toBeLessThan(2 * 1048576);
  expect(total).toBeGreaterThan(100000);
  expect(await page.locator('#gene-details').textContent()).toContain('SAMD11');
});
