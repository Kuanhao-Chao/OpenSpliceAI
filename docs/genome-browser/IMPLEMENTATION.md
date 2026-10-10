# Human genome browser implementation

The browser is a standalone TypeScript application built for
`/OpenSpliceAI/genome/` by the existing Sphinx documentation build. It follows the
yeast browser's navigation, tracks, themes, sequence search, figure export and
mobile interactions. Its measurements are the existing variant delta scores;
no additional inference is performed.

## Data and publication contract

* GRCh38.p14 reference, the scoring annotation's exons and strand, single-model
  rs10 and rs13 results, and the original masked SpliceAI annotation.
* Complete rs10 is the default. rs13 is a frozen, audited preview until a final
  full-content audit permits promotion. A filename alone never establishes
  acceptance. Preview gaps, missing annotations, measured zero and positions
  outside the source collection are separate states.
* Lossless integer DS (scale 100000) and signed DP, per-annotation gene symbols,
  original row identities, duplicate occurrences and reference-match flags.
  Genomic affected-site coordinates are POS + DP on either strand.
* Independent gzip blocks in indexed packed files; summaries serve wide views,
  while byte ranges fetch exact scores near a variant. No per-base files.
* Metadata and file hashes identify immutable snapshots. The manifest is
  published last. Hosting must provide HTTPS, byte ranges and CORS for khchao.com.
  The institutional URL is configurable; paid services are not required.

## Acceptance checklist

- [x] Streaming, resumable data preparation and source-content checks
- [x] Real-data review subset with honest coverage and dataset labels
- [x] Lossless Python/TypeScript decoding and edge-case tests
- [x] Main/alternative contigs, gene/region/variant navigation and MANE features
- [x] Four DS tracks, alternate-allele heatmap, comparison and variant inspector
- [x] Coverage overview, reference, ROI, history and complete shared state
- [x] IUPAC sequence search on both strands, with cancellation and a static index
- [x] Six themes, mobile controls, keyboard and accessible alternatives
- [x] Stable PNG/SVG, exact CSV and separately labelled summary CSV exports
- [x] Bounded cache, workers, range validation, explicit failures and retry
- [x] Sphinx integration, CI, documentation and publication checks
- [x] Chromium/Firefox audits and captured validation evidence
- [x] WebKit acceptance on the Ubuntu CI host
- [ ] Completed genome-wide score/reference packaging and final content verification
- [ ] Institutional upload, endpoint probe and genome-wide catalog activation

These checked items describe implemented and tested features. They do not
assert that all genome-wide preparation or live publication has finished.
See [validation/results.json](validation/results.json) for measured evidence.

## Preparation started on October 10

The frozen run is `grch38-r10-r13preview-20261010T184032Z`. Its preview accepts
92,294 chunks, including 390 freshly content-verified receipt-bound worker
passes. Later scoring progress belongs to a subsequent immutable snapshot.

| Job | Role | CPUs | RAM | Time limit | Dependency |
| --- | --- | ---: | ---: | --- | --- |
| 31847915 | Score conversion | 2 | 8 GiB | 48 h | None |
| 31847916 | Reference and FM indexes | 2 | 16 GiB | 24 h | None |
| 31847939 | Score checkpoint continuation | 2 | 8 GiB | 48 h | After 31847915 ends |
| 31847942 | Score checkpoint continuation | 2 | 8 GiB | 48 h | After 31847939 ends |
| 31847952 | Attach reference and verify public files | 1 | 4 GiB | 2 h | Both paths succeed |

All use `ssalzbe1_bigmem`/`bigmem` and zero GPUs. The concurrent request is
4 CPUs and 24 GiB RAM. Slurm reports billing=2 for each two-CPU job. The
configured maximum is 338 CPU billing hours across all five jobs; completed
score continuations exit immediately, so this ceiling is not a cost forecast.
Reference/search preparation completed successfully for all 705 contigs in
19m27s, producing 2,671,966,163 public data bytes. It used 0.6483 CPU billing
hours and peaked at 3.38 GiB RAM. Score preparation continues with 2 CPUs and
8 GiB; checkpoint continuations and final verification remain dependent jobs.
The measured small-sample estimate is 30–65 score-conversion wall hours with
two workers, approximately 50 GB of score packs, plus reference/search data
and verification. Queue time and source I/O remain uncertain.

The initial failed preparation attempt wrote no score packs. Its quota-table
parser was corrected and tested; this new run leaves the failed evidence
intact. The running code and audit inputs are copied into the run directory,
so subsequent repository work cannot change the executing jobs. Source-control
review uses an isolated worktree and leaves the scoring checkout's HEAD alone.

Full-data preparation reads the existing VCFs without changing them and uses
CPUs only. It does not cancel, resubmit or change any r13 scoring job. The review
subset is not described as a genome-wide publication. Live deployment depends
on the institutional host details and completed data verification.
