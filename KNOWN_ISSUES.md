# Current limits and verification boundaries

This file describes development version 0.1.0.dev0. Historical results remain tied
to their source/environment; see [migration](docs/source/content/migration.rst)
and the [audit evidence](docs/development/comprehensive-audit.md).

## Corrected behavior

Focal alpha/gamma, scheduler timing, initialization seeds, complete observed-label
losses, partial batches and absent-class metrics are corrected. Transfer trains the
output head plus selected residual units, keeps frozen BatchNorm buffers fixed,
and loads students/teachers strictly unless partial student initialization is
explicit. Calibration uses bounded disk-backed logits, honors optimization options
and publishes portable calibrated state. Zero-label padding contributes no loss,
gradient or evaluation/calibration count.

Split FASTA prediction owns each position once, excludes padding and maps both
strands and arbitrary contig names to BED6. Overlapping *distinct annotated genes*
can still describe the same genomic position; ownership deduplicates segments of
one input entry, not unrelated biological annotations. Genomic FASTA header spans
must match sequence lengths. Unannotated FASTA is scored in the supplied orientation.

Annotation caches are tied to GFF/GTF content; gene isoforms stay in one split and
keep independent labels. Requested paralogy filtering fails explicitly if minimap2
cannot initialize and measures coverage over the query span.

## Limits

- Dataset creation still loads assembly sequences and sequence/label tables in
  memory. Calibration is bounded apart from its model/inference batch and temporary
  disk cache. Large input files require adequate RAM/storage.
- Original SpliceAI parity applies to shared supported SNVs/indels using the original
  10,000-base Keras weights. MNV/delins extensions and rejected REF realignment spans
  are tested separately; original placeholders are not numeric parity evidence.
- GPU/CPU tolerance, nondeterministic kernels and biological generalization require
  their own evidence. Seeded fixtures and coverage do not prove universal correctness.
  TF32 and cuDNN benchmarking in variant scoring are explicit opt-ins.
- Finite VCF DS values are validated before atomic filesystem publication. Stdout
  streams cannot be rolled back after bytes have been written.
- Ranking metrics return NaN when a selected split contains no positive sites;
  fixed-order classification metrics report zero for an absent class. The historical
  `accuracy` metric is macro class recall rather than overall base accuracy.
- The historical `test` command is disabled. Standalone experiment scripts and dead
  alternates are excluded from maintained-package coverage and inventoried separately.
- macOS and newer Python versions are configured CI checks until successful job
  evidence is recorded. Missing optional backend/device checks remain explicit gaps.

Signed r13 scoring continues on frozen production source. Repository changes do not
restart workers, retrain released models or regenerate scores. Future controller
recovery helpers are staged and tested with simulated external services.
