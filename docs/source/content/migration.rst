.. _migration:

Migration to 0.1.0.dev0
=======================

This development version corrects scientific behavior and is not a drop-in
reproduction of earlier training runs. Existing raw weights remain loadable.

* Focal alpha/gamma are honored (0.25/2), MultiStepLR steps per epoch, and cosine
  restarts use fractional progress across the entire epoch.
* Seeds cover Python/NumPy/PyTorch before initialization. Partial batches and
  single-window shards are retained unless training explicitly uses ``--drop-last``.
* Validation filenames can be explicit. Full selected-split losses replace the
  former capped evaluation sample; absent-class metrics keep their correct names.
* Transfer trains the output head plus N residual units and freezes BatchNorm
  buffers in frozen units. Student loading is strict by default; partial student
  initialization is explicit. Teachers always load strictly.
* Calibration options control bounded disk-backed NLL fitting. Portable calibrated
  artifacts work directly in prediction/variant. Plot samples and exact metric
  counts are labeled separately.
* Split FASTA output owns each source position once, maps minus-strand/non-chr
  coordinates correctly, excludes padding and uses BED6 for unannotated entries.
* Variant requires a model, uses scoped FP32 defaults, keeps stdout valid, writes
  compressed destinations correctly and validates numeric annotations before
  atomic publication. Library callers receive exceptions instead of process exits.
* Data creation supports advertised biotypes, independent isoform labels and
  validation grouping by gene ID to prevent repeated isoforms crossing splits.
* Annotation caches track source content, GTF transcript/gene-type fields are supported,
  and paralogy filtering uses query-span coverage with explicit failures.
* Exact variant contig names take precedence over optional chr aliases, including
  assemblies mixing chromosome and accession names.
* Zero-label padding is excluded from loss, ranking/classification metrics and
  calibration. Full-split counts now describe observed positions.

Historical reproduction
-----------------------

Use the previous release/source and its recorded environment, input checksums,
checkpoint hashes and command line. ``d23e4413`` identifies the preserved
pre-correction hardening branch; production main is ``074d3e27`` plus its existing
local scorer patches. Those are separate reproducibility targets. Do not combine
new training defaults with an older report and call it a reproduction.

Production campaigns
--------------------

Signed r13 scoring remains on frozen source. Validate resulting shards and their
provenance before switching production, or independently validate and freeze a new
runtime transition. No current campaign scores or published models are regenerated
by this software audit. See the repository audit and resource ledger for evidence.
