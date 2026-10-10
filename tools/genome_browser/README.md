# Human genome browser data

This pipeline publishes the existing masked GRCh38 OpenSpliceAI SNV results.
It performs no model inference and never changes scoring jobs or source VCFs.
The browser source is in `docs/genome-browser`; Sphinx publishes its build
under `/OpenSpliceAI/genome/`.

## Frozen configuration

Start with the configuration returned by `tools.genome_browser.review.make_config`
and use a **new immutable ID** and output directory for a genome-wide snapshot.
Remove `chunks`, `regions`, and `reference_review_regions`, use
`scope: "source-collection"`, and set `shard_size: 1000`. Supply:

| Field | Meaning |
| --- | --- |
| `r10_manifest` | Accepted full-content VCF audit TSV, including source paths, SHA-256 and record counts |
| `r13_manifest` | Frozen accepted audit TSV; unaccepted rows are preview gaps |
| `reference`, `reference_sha256` | Pinned indexed FASTA and authenticated reference identity |
| `annotation`, `annotation_sha256` | Scoring annotation TSV and its identity |
| `models` | Labels, source precision, seed and actual single-model hashes |
| `r13_evidence` | Honest description of content, receipt-bound and legacy evidence |
| `default_view` | 0-based half-open initial region |

Keep private absolute paths in local configuration, never in a public catalog.
Copy audit manifests to an immutable preparation directory before a full run.
R13 preview snapshots are explicitly separate from final completion. Scored
record counts are not distinct allele counts. Unsupported and baseline-only
contigs remain inspectable. See [FORMAT.md](FORMAT.md) for exact semantics.

## Prepare, resume and verify

```bash
python -m tools.genome_browser prepare CONFIG.json SNAPSHOT_DIR
python -m tools.genome_browser finalize CONFIG.json SNAPSHOT_DIR
python -m tools.genome_browser verify SNAPSHOT_DIR
```

Preparation hashes each source VCF while parsing it and checks source keys and
row counts between models. Changed source content aborts the shard. Shards are
committed only after their pack is flushed; interrupted `.partial` files are
replaced on resume. The SQLite preparation index uses a few files rather than
per-tile receipts. `--first-shard` and `--last-shard` allow bounded CPU batches.
Use `--workers 2` within a two-CPU allocation for disjoint shard workers;
their SQLite commits are serialized. Do not start a second independent
preparation coordinator for the same snapshot.
Finalization requires every configured shard unless `--allow-subset` explicitly
marks a preparation subset. Complete file verification precedes publication.

The review subset can be reproduced on the data-owning machine with
`python -m tools.genome_browser.review`. Its original VCF hashes, gene annotation
and FASTA are checked; it is not a simulated or genome-wide result.

## Reference and whole-genome motif search

Install `requirements.txt` in a separate environment; scoring environments
remain untouched. Reference blocks and a native-built paged FM index allow
sequence searches entirely through static byte ranges:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m tools.genome_browser reference CONFIG.json SNAPSHOT_DIR --search
python -m tools.genome_browser attach-reference SNAPSHOT_DIR
python -m tools.genome_browser verify SNAPSHOT_DIR
```

Construction processes one reference contig at a time. Native suffix sorting
requires several GB for chr1; request 16 GiB and cap native threads to the
allocated CPUs. `--contigs chrM,chr1` permits an independently resumable batch.
Search needs every contig's index for a whole-reference query. Missing indexes
raise a visible error instead of reporting zero hits. IUPAC N matches canonical
A/C/G/T, not unknown reference bases. Results are bounded to 200 displayed
strand hits; the total count is computed without locating every occurrence.

## Resources and publication

Use packed files to stay within the shared filesystem's inode budget. A full
100,000-chunk snapshot uses about 100 score packs, approximately 200 regional
index files and five reference/search artifacts per contig. Leave room
for the still-running r13 output VCFs and receipts. Compression ratios from
small samples are estimates; the preparation logs report actual bytes.
Use a CPU partition and a separate small allocation for preparation. No GPUs
are needed. Avoid sharing or modifying the sealed scoring controller.

For the existing local campaign, `python -m tools.genome_browser.jobs freeze
REPOSITORY NEW_RUN_DIRECTORY` copies the audit inputs, preparation Python code
and auditor into a new run. The `slurm/prepare.sbatch` launcher executes that
frozen copy. Its score role freshly verifies worker passes before freezing the
preview; score and reference roles then run independently. Checkpoint score
continuations use `afterany` and skip completed work. A scientific failure
records a failure marker and requires review before any continuation. The
finalize role runs only after both paths succeed and verifies every public
file. The quota guard preserves remaining scoring outputs and a file margin,
including files already created when a preparation run resumes.

The October 10 representative 12-chunk benchmark converted 396,841 source
occurrences into 5,945,419 packed bytes in 47.82 wall seconds and 22.22 CPU
seconds, using about 714 MiB peak RAM. Linear extrapolation gives 49.55 GB of
score packs and 51.44 CPU hours. The estimated two-worker score run is 30–65
wall hours; setup, audit, compression mix and shared I/O make this approximate.
Reference/search indexes are additional. This is a preparation estimate, not
the time required to finish model scoring. Actual Slurm allocations and files
should replace estimates as full shards complete.

The small application and review files fit in the documentation build. Full
data belong on the existing institutional host. Before upload, configure its
HTTPS route using the examples in `hosting/` and run local verification. Upload
data files first, then manifest, then catalog. Catalog example:

```json
{"datasets":[{"id":"grch38-immutable-snapshot","label":"GRCh38 r10 complete / r13 preview","manifest":"https://HOST/DATA/SNAPSHOT/manifest.json"}]}
```

Set `docs/genome-browser/public/settings.json` to `{"catalogUrl":"https://HOST/DATA/catalog.json"}`.
Require exact HTTP 206 ranges and CORS exposing `Content-Range`, and disable
HTTP content compression of packed files. Internally gzip-compressed pages
must retain their stored byte offsets. Validate the actual endpoint with:

```bash
python -m tools.genome_browser probe https://HOST/DATA/SNAPSHOT/manifest.json
```

Generate a concrete upload inventory and checksums after verification:

```bash
python -m tools.genome_browser publication-plan SNAPSHOT_DIR \
  https://HOST/DATA/SNAPSHOT/ PUBLICATION_PLAN_DIR
```

This creates `data-files.txt`, `checksums.sha256`, `catalog.json` and a publication
receipt; it does not upload files. Only the manifest-listed data files belong
in the public directory. Private configurations, audit TSVs and preparation
SQLite files remain local. Use the verified inventory for the institutional
host's upload method, then probe its HTTPS endpoint before switching settings.

For local review, `python -m tools.genome_browser serve SNAPSHOT_DIR` supplies
the same range/CORS contract at `http://127.0.0.1:8765/manifest.json`.

## Validation

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s tools/genome_browser/tests -v
cd docs/genome-browser
npm ci
npm test
npm run build
npx playwright install chromium firefox webkit
npm run audit
```

Python tests cover lossless numbers, bounds, missing data, REF flags, hashes,
row alignment, corrupted checkpoints and native sequence-index search. Browser
tests cover real allele round trips, conflicts, saved state, themes, mobile
layouts, sequence search, failed ranges and PNG/SVG/CSV exports. Ubuntu CI runs
all three browser engines; local WebKit availability depends on host libraries.
