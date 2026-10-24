# IGV browser implementation

The approved browser uses pinned IGV.js 3.8.9 for native dragging, ruler zoom and
selection, track presentation, multiple panes and local or URL tracks. Exact
source annotations remain in lossless OSGB1 packs. Worker queries return
100-row pages and compact plot buffers; full CSV and statistics use the complete
matching collection.

The default expanded demonstration contains ten scored intervals and 67,579
source occurrences. CFTR, HBB, BRCA1 and BRCA2 add four 2,048-base splice-junction
windows across both strands. Existing examples, reference-only mitochondrial
search and earlier immutable snapshots remain available. Manifest-driven
bookmarks identify scored windows; merged interval coverage distinguishes fully
published, partially published and unavailable views. A gene search can show
more context than a demonstration window.

Version-2 sessions pin the manifest URL and SHA-256 and preserve loci, the active
pane, presentation and regions. Built-in scientific tracks retain pinned data
identities while accepting presentation changes. Local user tracks serialize as
reselection placeholders. The archived viewer preserves version-1 links.

Conversion verifies lossless DS/DP, annotation and source identities; native
tracks reduce REF-matching canonical alleles while preserving recorded zero and
missing states. The expanded selection is reproducible from
`tools/genome_browser/recipes/demo-expanded-v4.json`. Its source census and every
native plotted value are checked against the audited VCFs. Only authenticated
manifest-listed files are copied into the public snapshot.

Both final source models completed processing; each has 3,271,090,620 predictions
among 3,339,752,304 distinct canonical input SNVs. The 68,661,684 absent
predictions stay absent. Source occurrences and annotation entries are not
relabeled unique variants. GENCODE v24 supplies scoring genes; MANE describes
training. Historical provenance remains explicit.

Full browser-data packaging is independent of this demonstration release. The
existing frozen jobs retain their code and allocations. Readiness requires
receipts bound to the full dataset, including final census and content checks;
intermediate packs or a queued continuation do not establish completion. Large
full data remain outside Git/GitHub Pages and require an indexed HTTPS host.

`validation/results.json` records measured release checks and immutable dataset
identities. `LOCAL_REVIEW.md` describes feature checks and local validation.

## Focused release validation

Pages uses 16 scientific/query/coverage unit checks, all public snapshot hashes,
TypeScript and production/Sphinx builds, local links, and five core workflows
per browser (15 checks across Chromium, Firefox and WebKit). The browser gate
serves the production bundle with Vite preview. Core phone checks verify usable
controls at 320 px; global scroll-area geometry stays in the extended audit.

All 35 frontend tests, 72 broader browser checks and 32 data-preparation tests
remain available through the separate manual extended-audit workflow. That
workflow has no deployment dependency or Pages publication permission.
