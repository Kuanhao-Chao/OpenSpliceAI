import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.goto('./');
  await expect(page.locator('#igv-browser')).toHaveAttribute('data-ready', 'true', { timeout: 60000 });
  await expect(page.locator('#load-status')).toContainText('Exact');
});

test('native tracks and a self-contained audited demonstration', async ({ page }, info) => {
  await expect(page.locator('#dataset-note')).toContainText('processing complete');
  await expect(page.locator('#dataset-note')).toContainText('97.94%');
  await expect(page.locator('#dataset-note')).toContainText('demonstration');
  expect(await page.locator('#igv-browser canvas').count()).toBeGreaterThan(7);
  await expect(page.locator('#igv-browser')).toContainText('GENCODE v24');
  await expect(page.locator('#allele-page')).toContainText('page 1');
  await page.screenshot({ path: info.outputPath('igv-default.png'), fullPage: true });
});

test('continuous native dragging updates the viewport before release', async ({ page }) => {
  const viewport=page.locator('#igv-browser .igv-viewport:has(canvas):visible').nth(4);
  await viewport.scrollIntoViewIfNeeded();
  const target=viewport.locator('canvas').first();
  const box = await viewport.boundingBox(); expect(box).toBeTruthy();
  const before = await target.evaluate(el => el.style.left);
  await page.mouse.move(box!.x + box!.width * .6, box!.y + 10); await page.mouse.down();
  await page.mouse.move(box!.x + box!.width * .3, box!.y + 10, { steps: 8 });
  expect(await target.evaluate(el => el.style.left)).not.toBe(before);
  await page.mouse.up();
  await expect(page.locator('#view-label')).not.toHaveText('chr1:925,851-926,130');
  await expect(page.locator('#load-status')).toContainText('Exact');
  expect(before).toBeDefined();
});

test('multiple native regions and inspector-pane selection', async ({ page }) => {
  await page.locator('#multilocus').fill('chr1:925850-926130 chr1:69040-70060');
  await page.locator('#multilocus-form button').click();
  await expect(page.locator('#locus-pane option')).toHaveCount(2);
  await page.locator('#locus-pane').selectOption('1');
  await expect(page.locator('#view-label')).toContainText('69,040');
  await expect(page.locator('#load-status')).toContainText('Exact');
});

test('session saves native tracks and exact snapshot identity', async ({ page }) => {
  const waiting = page.waitForEvent('download'); await page.locator('#save-session').click();
  const download = await waiting; const stream = await download.createReadStream(); const parts: Buffer[] = [];
  for await (const part of stream!) parts.push(part);
  const saved = JSON.parse(Buffer.concat(parts).toString());
  expect(saved.version).toBe(2); expect(saved.manifest).toContain('/data/final-demo-');
  expect(saved.igv.tracks.length).toBeGreaterThan(4); expect(saved.state.snapshot).toContain('final');
});

test('outside-demo regions remain explicitly unavailable', async ({ page }) => {
  await page.locator('#locus').fill('chr7:117504153-117504464');
  await page.locator('#locus-form button[type=submit]').click();
  await expect(page.locator('#load-status')).toContainText('Outside published demonstration');
  await expect(page.locator('#variant-details')).not.toContainText('scored zero');
});

test('all requests remain local during demonstration navigation', async ({ page }) => {
  const external: string[] = [];
  page.on('request', request => { if (/^https?:/.test(request.url()) && new URL(request.url()).hostname !== '127.0.0.1') external.push(request.url()); });
  await page.reload(); await expect(page.locator('#load-status')).toContainText('Exact');
  await page.locator('[data-bookmark="3"]').click(); await expect(page.locator('#view-label')).toContainText('KI270773');
  await expect(page.locator('#load-status')).toContainText('Exact'); expect(external).toEqual([]);
});

test('ruler zoom, selected regions, BED export and session restoration',async({page})=>{
 const ruler=page.locator('#igv-browser .igv-viewport:has(.igv-ruler-tooltip):visible').first();await ruler.scrollIntoViewIfNeeded();const before=await page.locator('#view-label').textContent();let box=(await ruler.boundingBox())!;
 await page.mouse.move(box.x+box.width*.3,box.y+box.height*.6);await page.mouse.down();await page.mouse.move(box.x+box.width*.7,box.y+box.height*.6,{steps:10});await page.mouse.up();await expect(page.locator('#view-label')).not.toHaveText(before!);
 box=(await ruler.boundingBox())!;await page.keyboard.down('Shift');await page.mouse.move(box.x+box.width*.3,box.y+box.height*.6);await page.mouse.down();await page.mouse.move(box.x+box.width*.7,box.y+box.height*.6,{steps:10});await page.mouse.up();await page.keyboard.up('Shift');
 const download=async(id:string)=>{const waiting=page.waitForEvent('download');await page.locator(id).click();const d=await waiting;const parts=[];for await(const chunk of (await d.createReadStream())!)parts.push(chunk);return Buffer.concat(parts);};
 const bed=(await download('#roi-bed')).toString();expect(bed).toMatch(/^chr1\t\d+\t\d+\t/);
 const session=await download('#save-session');await page.locator('#session-file').setInputFiles({name:'session.json',mimeType:'application/json',buffer:session});await page.waitForURL(url=>url.searchParams.has('manifest'));await expect(page.locator('#igv-browser')).toHaveAttribute('data-ready','true',{timeout:60000});await expect(page.locator('#load-status')).toContainText('Exact',{timeout:30000});expect((await download('#roi-bed')).toString()).toBe(bed);
});

test('local annotation tracks and imported regions remain local',async({page})=>{
 await page.locator('#track-files').setInputFiles({name:'review.bed',mimeType:'text/plain',buffer:Buffer.from('chr1\t925900\t926000\tLocal_feature\n')});await expect(page.locator('#track-status')).toContainText('Local tracks loaded');
 await page.locator('#roi-file').setInputFiles({name:'regions.bed',mimeType:'text/plain',buffer:Buffer.from('chr1\t925900\t926000\tMarked\n')});await expect(page.locator('#track-status')).toContainText('Imported 1 regions');
 const waiting=page.waitForEvent('download');await page.locator('#save-session').click();const d=await waiting;const parts=[];for await(const chunk of (await d.createReadStream())!)parts.push(chunk);const saved=JSON.parse(Buffer.concat(parts).toString());expect(JSON.stringify(saved)).toContain('localFile');expect(JSON.stringify(saved)).toContain('Marked');
});

test('whole-genome native overview exports figures and disables exact CSV',async({page})=>{
 await page.locator('#igv-browser select').first().selectOption('all');await expect(page.locator('#view-label')).toHaveText('Whole genome');await expect(page.locator('#csv')).toBeDisabled();
 const waiting=page.waitForEvent('download');await page.locator('#svg').click();const d=await waiting;expect(d.suggestedFilename()).toContain('whole-genome');const parts=[];for await(const chunk of (await d.createReadStream())!)parts.push(chunk);expect(Buffer.concat(parts).toString()).toContain('<svg');
});

test('multiple URL tracks keep distinct identities after a session round trip',async({page})=>{
 await page.route('**/review-url-*.bed',route=>route.fulfill({contentType:'text/plain',body:'chr1\t925900\t926000\tURL_feature\n'}));
 for(const name of ['a','b']){await page.locator('#track-url').fill(`http://127.0.0.1:4173/review-url-${name}.bed`);await page.locator('#add-track-form button').click();await expect(page.locator('#track-status')).toContainText('Track loaded');}
 const capture=async()=>{const waiting=page.waitForEvent('download');await page.locator('#save-session').click();const d=await waiting;const chunks=[];for await(const chunk of (await d.createReadStream())!)chunks.push(chunk);return Buffer.concat(chunks);};
 const content=await capture();const saved=JSON.parse(content.toString());const user=saved.igv.tracks.filter((t:{id:string})=>t.id.startsWith('user-'));expect(user).toHaveLength(2);expect(new Set(user.map((t:{id:string})=>t.id)).size).toBe(2);
 await page.locator('#session-file').setInputFiles({name:'session.json',mimeType:'application/json',buffer:content});await page.waitForURL(url=>url.searchParams.has('manifest'));await expect(page.locator('#igv-browser')).toHaveAttribute('data-ready','true',{timeout:60000});await expect(page.locator('#load-status')).toContainText('Exact',{timeout:30000});const restored=JSON.parse((await capture()).toString());expect(restored.igv.tracks.filter((t:{id:string})=>t.id.startsWith('user-')).map((t:{id:string})=>t.id).sort()).toEqual(user.map((t:{id:string})=>t.id).sort());
});

test('restored presentation survives model changes and cannot replace pinned data',async({page})=>{
 const capture=async()=>{const wait=page.waitForEvent('download');await page.locator('#save-session').click();const d=await wait;const chunks=[];for await(const chunk of (await d.createReadStream())!)chunks.push(chunk);return JSON.parse(Buffer.concat(chunks).toString());};
 const saved=await capture();const gene=saved.igv.tracks.find((t:{id:string})=>t.id==='osai-genes');gene.name='Custom scoring genes';gene.color='#ff00ff';gene.height=160;gene.url='http://127.0.0.1:4173/never-load.bed';const sequence=saved.igv.tracks.find((t:{id:string})=>t.id==='osai-reference');sequence.reversed=true;
 let replaced=false;await page.route('**/never-load.bed',route=>{replaced=true;return route.fulfill({body:'wrong data'});});await page.locator('#session-file').setInputFiles({name:'session.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(saved))});await page.waitForURL(url=>url.searchParams.has('manifest'));await expect(page.locator('#igv-browser')).toHaveAttribute('data-ready','true',{timeout:60000});await expect(page.locator('#load-status')).toContainText('Exact',{timeout:30000});
 await page.locator('#model').selectOption('r13');await expect(page.locator('#igv-browser')).toContainText('Custom scoring genes');const restored=await capture();const g=restored.igv.tracks.find((t:{id:string})=>t.id==='osai-genes');expect(g.height).toBe(160);expect(g.name).toBe('Custom scoring genes');expect(replaced).toBe(false);expect(restored.igv.tracks.find((t:{id:string})=>t.id==='osai-reference').reversed).toBe(true);
});
