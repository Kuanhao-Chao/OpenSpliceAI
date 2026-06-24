# Known Issues & Audit Notes

This file records issues found during the correctness audit that were **deliberately not
changed**, plus two audit findings that turned out to be **false alarms** (correct code that is
now protected by regression tests). For the issues that *were* fixed, see the git history and the
tests under `tests/`.

## Deferred — behavior-changing (not fixed to preserve reproducibility)

These are genuine concerns, but fixing them would change training dynamics or outputs relative to
already-published checkpoints. They are pinned by characterization tests and left for a maintainer
decision rather than changed silently.

1. **`focal_loss` ignores `alpha` and hardcodes `gamma`** — `openspliceai/train_base/utils.py`.
   `focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0)` reassigns `gamma = 2` internally and never
   uses `alpha`, so both hyperparameters are inert. The effective loss is a valid focal loss at
   `gamma=2` with no class weighting, just not tunable. Pinned by
   `tests/unit/test_train_base_utils.py::test_focal_loss_known_issue_gamma_ignored`. Only affects
   runs that select `--loss focal_loss` (cross-entropy is the default).

2. **Per-batch `scheduler.step(epoch_fraction)`** — `openspliceai/train_base/utils.py` (train loop).
   The LR scheduler is stepped once per batch with a *fractional* epoch value
   (`global_batch_idx / total_batches_in_epoch`). Because `global_batch_idx` accumulates across
   epochs it roughly reaches integer epoch values at epoch boundaries, so `MultiStepLR` milestones
   mostly fire — but the cadence is fragile with uneven shard sizes and is not the intended schedule
   for `CosineAnnealingWarmRestarts`. Recommended future fix: step once per epoch with an integer
   epoch. This changes the LR trajectory, hence deferred.

3. **`Y` stored with an extra leading dimension** — `openspliceai/create_data/create_dataset.py`
   writes `Y{i}` as shape `(1, n, SL, 3)` (a list-wrapped array). Every consumer already compensates
   with `h5f['Y{i}'][0, ...]` (`train_base/utils.py`, `calibrate/temperature_scaling.py`), so the
   pipeline is correct end-to-end; the layout is just a vestige of SpliceAI's multi-output design.
   Changing the on-disk format would require regenerating all `dataset_*.h5` files and updating every
   loader in lockstep, so it is left as-is. The invariant is exercised by the calibrate integration
   test (which reads the nested schema) and the synthetic fixtures.

## Fixed during the v0.0.7 test-hardening pass

Two latent bugs were uncovered while expanding the test suite for v0.0.7 and fixed. Both live in
code paths the packaged CLI does not currently reach, so real-genome `predict`/`variant` outputs are
**unchanged** (verified: the variant delta-score and keras-equivalence tests are bit-identical
before/after).

- **`variant` `one_hot_encode` miscoded non-ACGTN bases** — `openspliceai/variant/utils.py`.
  The encoder only folded a literal `N` to the all-zero row; any other non-ACGT byte (IUPAC ambiguity
  codes `R`/`Y`/`S`/…, gaps, etc.) aliased onto a concrete base via the `byte % 5` lookup, contradicting
  the map's own "N or any invalid character" contract. Fixed by sanitising `[^ACGT] -> N` before the
  lookup. ACGTN input is bit-identical, so reference-genome scoring is unaffected. Locked by
  `tests/regression/test_encode_decode_roundtrip.py::test_variant_encoder_maps_non_acgt_to_zero_row`.

- **`predict` `get_sequences(neg_strands=...)` crashed** — `openspliceai/predict/predict.py`.
  The minus-strand branch called `.reverse.complement` on a `str` (the sequence had already been
  materialised via `.seq`), raising `AttributeError`. The feature was entirely broken but unreachable
  from `predict_cli` (which never passes `neg_strands`). Fixed by reverse-complementing through the
  pyfaidx `Sequence` object. Locked by
  `tests/unit/test_predict_turbo_and_debug.py::test_get_sequences_neg_strand_reverse_complements`.

## False positives — correct code, now locked by regression tests

The automated audit flagged these as high-severity bugs. Hand-tracing (and the regression tests
below) show the code is **correct**; do not "fix" them.

- **Minus-strand donor/acceptor labeling** — `openspliceai/create_data/create_datafile.py`.
  For the `-` strand the donor index is derived from `exons[i+1].start` and the acceptor from
  `exons[i].end` via the reverse-complement transform `len(labels) - pos - 1`. This is biologically
  correct (verified with a concrete two-exon example whose canonical GT/AG motifs land exactly at the
  computed positions). Locked by `tests/regression/test_minus_strand_labeling.py`.

- **`clip_datapoints` crops X but not Y** — `openspliceai/train_base/utils.py`.
  `Y` is stored at length `SL`; `X` at length `SL + CL_max`. `clip_datapoints` trims `X` to
  `SL + CL`, and the model's internal `Cropping1D` removes the remaining `CL`, yielding `SL` — which
  matches `Y`. Cropping `Y` here would actually break training. Locked by
  `tests/regression/test_clip_datapoints_invariant.py`.

## `predict` / `variant` validation audit (see `validation/`)

A dedicated audit validated the two scoring subcommands step by step (`validation/ALGORITHM.md`,
`validation/VALIDATION_REPORT.md`).

- **Proven correct:** `variant --model-type keras --flanking-size 10000` reproduces the original Illumina
  `spliceai` 1.3.1 **exactly** (100% of DS and DP fields, across `--mask {0,1}` × `--distance {50,500,1000}`
  on 35 variants incl. both strands, indels, multiallelic, MNV). Locked by
  `tests/equivalence/test_keras_equivalence.py`. Equivalence requires **flanking 10000** (original hardcodes
  `wid = 10000 + cov`).

- **Fixed:** `variant/variant.py` crashed (`FileNotFoundError: ''`) when `--output-vcf` was a bare filename;
  now guards the empty-dirname case. Locked by `tests/integration/test_variant_output_dir.py`.

- **Hardened:** `predict.py` `load_pytorch_models` now raises on an unsupported `--flanking-size` (was a
  silent 80nt default; unreachable via the CLI's argparse `choices`). Locked by
  `tests/unit/test_predict_variant_hardening.py`.

- **Caveat (not a bug):** whole-genome `predict` writes **duplicate BED rows** in the `CL_max//2` overlap
  zones of sequences split beyond `--split-threshold` (1.5 Mb). Coordinates are correct, rows are redundant;
  dedup with `validation/dedup_predictions.py`.
