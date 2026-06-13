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
