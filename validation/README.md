# validation/ — predict & variant correctness audit

Step-by-step validation that the `predict` and `variant` subcommands are correct, done before large-scale
variant scoring. Read **`VALIDATION_REPORT.md`** for the findings and **`ALGORITHM.md`** for the
step-by-step algorithm. Bottom line: `variant` is numerically identical to the original Illumina SpliceAI at
flanking 10000; `predict` maps coordinates correctly on both strands; one real bug was fixed and one guard
added; all flagged concerns resolved.

## Scripts (reproducible; interpreter = pytorch_cuda env)
- `build_equiv_vcf.py` — build the equivalence-test VCF from real genes (both strands + edge cases).
- `compare_equiv.py` — field-level compare of original-SpliceAI vs OpenSpliceAI-keras VCFs (exit 0 iff exact).
- `build_predict_region.py` — build a region FASTA + GFF + ground-truth `sites.tsv` for predict recovery.
- `compare_predict_recovery.py` — check predict BED calls land on annotated splice sites (both strands).
- `dedup_predictions.py` — collapse duplicate BED rows from split-overlap zones (keep max score). **Run this
  on `predict` BED output for any whole-chromosome run.**

## Tests added (in `tests/`)
`unit/test_validation_invariants.py`, `unit/test_predict_variant_hardening.py`,
`equivalence/test_keras_equivalence.py` (keras vs original SpliceAI), `integration/test_variant_output_dir.py`.

Scratch run artifacts (`equiv/`, `predict_test/`, `split_test/`) are git-ignored.
