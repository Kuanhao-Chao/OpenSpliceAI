.. _train_subcommand:

train
=====

Train a residual convolutional model from scratch using three distinct HDF5
splits. ``--validation-dataset`` is explicit; the legacy fallback replaces
``train`` with ``validation`` in the training filename. Other filenames require
the explicit flag. Validation selects the best checkpoint; test results do not.

AdamW starts at 1e-3. ``MultiStepLR`` steps once per completed epoch, with positive
late-epoch milestones. ``CosineAnnealingWarmRestarts`` steps after each optimizer
update using progress across all shards. ``--early-stopping --patience N`` stops
after N validation epochs without improvement. Epoch and best checkpoints are
written atomically as raw state dictionaries.

Cross-entropy is the default. Focal loss honors ``--focal-alpha`` (0.25) and
``--focal-gamma`` (2); scalar alpha scales every class equally. All samples are
retained, including partial batches, unless ``--drop-last`` is explicitly set
for training. Evaluation always retains them and reports the complete selected
split. Absent-class precision/recall/F1 are zero with explicit support; ranking
metrics are undefined (``nan``) when no positive splice sites exist.

``--random-seed`` seeds Python, NumPy and PyTorch before initialization and gives
reproducible advancing shuffle generators. Hardware/kernel differences can still
change numerical results; record source, environment, inputs and seeds.

Example
-------

.. code-block:: bash

   openspliceai train --train-dataset data/dataset_train.h5 --validation-dataset data/dataset_validation.h5 --test-dataset data/dataset_test.h5 -f 80 --epochs 10 --project-name example --output-dir training

Complete options
----------------

.. include:: ../_generated/cli-train.inc

See :doc:`output_explanation` for schemas and :doc:`migration` for changed behavior.
