# Local genome browser feature review

The six-region browser implementation passed local feature review and was
approved on 2026-10-24. This records that reviewed baseline. The release adds
four gene examples; scoring and browser-data packaging remain separate.

## Open the production preview

The production browser and documentation are served on Rockfish `login02`,
loopback port 4174. From your computer:

```bash
ssh -J kchao10@login.rockfish.jhu.edu \
  -N -L 4174:127.0.0.1:4174 kchao10@login02
```

Keep that connection open, then visit:

- Browser: http://localhost:4174/OpenSpliceAI/genome/
- Guide: http://localhost:4174/OpenSpliceAI/content/genome_browser.html

The source development server remains on port 4173. Port 4174 uses the actual
production build, including the rebuilt archived viewer for version-1 links.

## Features to try

1. Drag a native track: pixels move before mouse release and coordinates follow.
2. Drag the ruler to zoom; Shift-drag to mark an interval. Export it as BED.
3. Use the model selector and comparison checkbox for r10, r13 and the original
   baseline. Default axes show DS from 0 to 1; missing predictions remain gaps.
4. Search `chr1:69091:A>G`. Inspect all original annotations, five-decimal
   OpenSpliceAI values, signed offsets, source occurrences and audit evidence.
5. Enter `chr1:925850-926130 chr1:69040-70060` in Compare regions and switch the
   inspector's active pane. Try gene names, variant queries, chromosomes and zoom.
6. Reorder, resize, rename and style native tracks. Save a session, restore it,
   switch models, and verify that presentation and marked regions persist.
7. Load local BED, BigWig, indexed VCF, BAM or CRAM files with their indexes.
   Import BED regions. Local data remain on your computer; reselect them after
   restoring a session. Remote user tracks retain their URLs and distinct IDs.
8. Browse the exact allele table and download full matching CSV. Pagination is
   100 rows; it does not truncate export or comparison statistics.
9. Export PNG/SVG, including a native whole-genome overview. Wide genomic views
   use summaries and disable exact CSV until sufficient detail is available.
10. Try the six themes and narrow layouts. Sequence search supports IUPAC,
    both strands, cancellation and the complete mitochondrial search example.

The default demonstration contains six real scored regions and 44,593 source
occurrences. It is not the full online dataset. Outside intervals are labeled
unavailable. Source processing completion and browser packaging are separate.

## Verified results

- TypeScript and production build: pass.
- Frontend unit tests: 31 pass.
- Python data/hosting/status tests: 28 pass, no skips.
- Browser acceptance: 21 Chromium + 21 Firefox + 21 WebKit tests pass.
- WebKit used a pinned Playwright 1.58.2 Ubuntu Noble container on four CPU
  cores; its frozen source hashes match the current frontend.
- Initial data transfer: 1,442,459 bytes (1.38 MiB), below the 2 MiB target.
- Current demonstration: 39 authenticated data files, 3,016,231 bytes.
- Sphinx build with warnings as errors: pass. All 2,114 local references resolve.
- Actual production browser, guide and legacy-link smoke checks: pass.
- BED, BigWig, VCF+tabix, BAM+BAI and CRAM+CRAI loading: pass with unique session
  IDs and no external data requests, using private software-test fixtures.
- Indexed BGZF reference and strict byte-range exact scores: pass in Chromium
  and Firefox using a separate, explicitly non-scientific protocol fixture.

Evidence is in `validation/local-review.json` and
`results/genome_browser/local_review_20261023_v4/verification/`. The latter has
`desktop-final.png`, `inspector-final.png`, `phone-final.png` and protocol evidence.
The prototype's startup transfer is measured separately from JavaScript/CSS.
The native IGV bundle is approximately 469 KB after gzip.

## Full source collection and preparation

Both final models have completed processing of 3,433,384,833 source occurrences.
Among 3,339,752,304 distinct canonical input SNVs, each model has predictions for
3,271,090,620 (97.94%). The 68,661,684 without predictions remain absent values,
not zeros or unfinished browser tasks. The collection is the original SpliceAI
input collection; it does not contain every possible SNV in the human genome.
GENCODE v24 supplies the scoring genes; MANE describes model training. The
individual seeds are not an ensemble. Historical and receipt-bound evidence
remain distinct.

Full browser packaging continues in job 32361767, using four workers, eight
allocated CPU units, 32 GiB and zero GPUs. The exact latest pack count and
accounting snapshot are in the validation JSON; percentages describe packaging,
not scoring. The source job was not interrupted or modified.

Dependent job 32417755 waits for that job to end. It verifies and reuses closed
packs, completes remaining conversion if needed, builds native tracks and
reference/provenance evidence, reconciles the census, and exports a local full
snapshot. Its allocation is eight CPUs/32 GiB, four workers and up to 36 hours
on `shared`, with zero GPUs. It does not publish. Scientific source-job failures
stop automatic continuation. Its controller was checked while pending to handle
transient accounting delays; the frozen conversion code is unchanged.

The full release is ready only when these files exist and bind the correct ID:

- `results/genome_browser/full_local_final_20261023/all-data-verified.json`
- `results/genome_browser/full_local_final_20261023/review-ready.json`

Check progress from the isolated worktree with its Python environment:

```bash
/data/ssalzbe1/khchao/OpenSpliceAI/.audit/igv-runtime/py/bin/python \
  -m tools.genome_browser status \
  /data/ssalzbe1/khchao/OpenSpliceAI/results/genome_browser/full_20261023_v2 \
  --job 32361767
```

The status command is read-only. Native tracks, reference and final verification
have different costs, so pack percentage is not total-runtime percentage.
Full packs are tens of GB, plus native tracks and reference/search artifacts.
The existing standard CPU allocation is sufficient; big-memory and GPU
allocations are not needed for the current preparation design.

After the full release is verified, it can be served locally with the same
application using `python -m tools.genome_browser serve PUBLIC_DIR --browser
BUILT_APP_DIR`. Full online activation still requires a suitable HTTPS range
endpoint and your later publication instruction. All changes remain local
until feature review is complete.
