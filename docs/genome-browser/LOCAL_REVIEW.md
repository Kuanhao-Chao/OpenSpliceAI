# Browser release review and validation

The IGV design and setup were approved on 2026-10-24. The release preserves that
viewer and expands its demonstration from six to ten scored intervals. Open
https://khchao.com/OpenSpliceAI/genome/ and the guide at
https://khchao.com/OpenSpliceAI/content/genome_browser.html.

## Features to verify

1. Drag a native track to pan before mouse release. Drag the ruler to zoom;
   Shift-drag to select a region and export it as BED.
2. Use the scored-example buttons for CFTR, HBB, BRCA1 and BRCA2. These are
   compact windows across both strands, not whole-gene releases.
3. Switch r10/r13 and comparison settings. Default delta-score axes run from
   zero to one; missing predictions remain gaps.
4. Search `chr1:69091:A>G` for exact annotations, five-decimal scores, signed
   offsets, source occurrences and audit evidence. Inspect the alternate-contig
   example for preserved duplicates, absent predictions and REF mismatches.
5. Extend a view beyond a scored interval. The coverage notice identifies
   partial availability and offers a button to focus the published window.
6. Compare multiple regions, switch the inspector pane, reorder/resize native
   tracks and restore a saved session. Track presentation and marked regions
   persist; built-in data remain pinned to their manifest.
   Each region has its own exact ALT matrix and matched-difference plot directly
   beneath its native columns. Check alignment while dragging, zooming and
   resizing; clicking a column selects the corresponding pane and base.
7. Add BED, BigWig, indexed VCF, BAM or CRAM tracks. Local files stay on your
   computer and must be reselected when a saved session is reopened.
8. Page through exact alleles and export full matching CSV. Pagination does not
   truncate export or comparison statistics. Outside published intervals,
   exact/summary CSV is disabled rather than fabricating scored zeros.
9. Export PNG/SVG, including native whole-genome overviews. Wide views use
   summaries and require closer zoom for exact CSV.
10. Try six themes, narrow layouts and sequence search. Mitochondrial reference
    search is available separately from the scored demonstration intervals.

## Data and evidence

The expanded dataset has 67,579 source occurrences across ten scored intervals.
The four additions account for 22,986 records with predictions from both final
models. These counts are source occurrences, not a claim of genome-wide score
availability. `public/catalog.json` identifies the active immutable snapshot.

`validation/results.json` records actual check results, source/data identities,
transfer measurements and deployment evidence. Larger logs, screenshots and
independent source-comparison evidence remain in the ignored release results.
Old version-1 links use the archived viewer; previous immutable version-2
snapshot URLs remain accessible.

## Run locally

From `docs/genome-browser`, install Node.js 22.12 or newer, then run `npm ci`,
`npm test`, `npm run build` and `npm run dev`. Open
http://127.0.0.1:4173/OpenSpliceAI/genome/.

`BROWSER_PORT` selects an alternative local port for isolated browser tests.
`BROWSER_CACHE_DIR` isolates a shared dependency cache. `BROWSER_BASE_URL` lets
Playwright target an already served build instead of starting its own server.

## Release and extended checks

`npm test` runs 22 score/coverage/query/geometry unit checks. `npm run build` creates
the production application; `npm run audit` then runs seven essential workflows
in Chromium, Firefox and WebKit (21 checks). The two alignment workflows compare
drawn columns against native one-base markers, including high DPI and multiple
regions. The browser gate uses Vite preview
of the built files. Phone checks exercise actual controls at 320 px.

Use `npm run test:all` for all 41 frontend tests and `npm run audit:all` for the
72-check audit, including themes, all viewport widths and advanced features.
The manually triggered **Genome browser extended audit** workflow also runs
the 32 backend tests. It is independent of Pages deployment.

The WebKit hosted-runner scroll-area finding is retained in the extended
audit: a 351 px document scroll width at a 320 px viewport, with no captured
element bounding box outside the viewport. Release checks require visible,
operable core controls rather than that document-wide geometry measurement.

## Full collection preparation

Both final models completed source processing. Among 3,339,752,304 distinct
canonical input SNVs, each has predictions for 3,271,090,620; the 68,661,684
without annotations remain absent values. The input does not contain every
possible human SNV.

Full browser packaging continues independently in the existing frozen CPU jobs.
Use `python -m tools.genome_browser status CAMPAIGN_DIR --job JOB_ID` for read-only
progress and billing. Pack percentage is not total-runtime percentage. Full
readiness requires correctly bound final verification receipts, followed by a
suitable HTTPS range endpoint for online activation. The demonstration release
does not alter those running jobs or publish partially prepared full data.
