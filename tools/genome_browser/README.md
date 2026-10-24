# Genome browser data and local review

The application uses IGV.js and final audited r10/r13 sources. No inference or
GPU resources are required. Full browser packaging remains separate from source
scoring completion; use the campaign verification receipts to establish readiness.

The small default dataset is a self-contained six-region demonstration. Every
public file is listed and hashed in its immutable manifest. Indexed BigWigs
support native browsing; OSGB1 packs retain exact DS/DP, genes, duplicates and
source identities. Missing predictions are not zeros. The scoring annotation is
GENCODE v24; the individual models were MANE-trained.

## Verified export and local serving

```bash
python -m tools.genome_browser verify SNAPSHOT_DIR
python -m tools.genome_browser export SNAPSHOT_DIR PUBLIC_DIR
python -m tools.genome_browser serve PUBLIC_DIR \
  --browser docs/genome-browser/dist/genome --port 8765
```

Open http://127.0.0.1:8765/. Export copies only manifest-listed files and requires
an empty destination. Full source-collection exports require release verification.
The server allowlists data files and can mount the built application on the same
origin. Existing data-only serving remains available without --browser.

## Final campaign

The frozen configuration binds accepted source manifests, their census, model
metadata, reference and annotation. The run command authenticates the executing
builder. Disjoint workers commit independently flushed packs to a SQLite index.
Checkpoint adoption checks frozen code, source manifests, scientific settings,
per-shard identities and every pack hash before rebinding it to a new release.

```bash
python -m tools.genome_browser.complete freeze REPO NEW_RUN_DIR
python -m tools.genome_browser.complete run NEW_RUN_DIR --workers 4
```

A full run creates exact packs, summaries, 12 native score BigWigs, 3 coverage
BigWigs, BGZF/tabix genes, paged provenance, reference and motif indexes. Native
conversion reconciles occurrence and distinct-allele census totals. Indexed
FASTA requires compressed-file hashes, decompressed identity and contig-length
proof. Completion writes all-data-verified.json; an intermediate manifest alone
is not a release. Preserve failed evidence and use a new immutable configuration
when code or source identities change.

Use checkpointed CPU allocations on shared for small jobs; no scoring jobs are
modified. The existing full campaign continues independently. Keep a free-space
and inode margin; full packs are tens of GB plus native tracks and reference.
Resource estimates must come from completed checkpoints and measured phases.

## Future online publication

The full dataset stays outside Git and GitHub Pages. After local verification
and approval, configure an HTTPS host with byte ranges and cross-origin headers,
then generate an upload inventory and probe the actual endpoint:

```bash
python -m tools.genome_browser publication-plan PUBLIC_DIR \
  https://HOST/DATA/SNAPSHOT/ PUBLICATION_PLAN_DIR
python -m tools.genome_browser probe https://HOST/DATA/SNAPSHOT/manifest.json
```

Preserve stored bytes for indexed files. Upload data before the manifest and
catalog. These commands do not upload or activate remote data. Publication is
currently deferred until feature review is complete.

## Validation

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tools/genome_browser/tests -v
cd docs/genome-browser
npm test
npm run build
npm run audit
```

Tests cover exact numeric preservation, duplicates, missing/zero states, corrupt
checkpoints, reference identity, paged motif search, safe exports and HTTP ranges.
Browser acceptance covers native gestures, panes, sessions, local tracks, regions,
figures, full CSV, responsive controls, cancellation and transfer budgets. WebKit
requires its supported runtime libraries; a pinned Playwright container can run
on CPU compute nodes without changing the Rockfish system environment.

## Read-only campaign status

```bash
python -m tools.genome_browser status CAMPAIGN_DIR --job JOB_ID
```

This reports committed packs, source occurrences, bytes, phase and Slurm billing
units without changing jobs or creating a preparation database. A pack percentage
is not a total-runtime percentage: native tracks, reference and final verification
follow conversion. Readiness markers must identify the configured snapshot.
