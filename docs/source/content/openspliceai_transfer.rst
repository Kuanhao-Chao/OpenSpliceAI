.. _transfer_subcommand:

transfer
========

Adapt a complete raw checkpoint to a new dataset using the same explicit
split, loss, seed and scheduler controls as training. AdamW starts at 1e-4.
The output head and last ``--unfreeze N`` residual units train; N=0 trains only
the head. Frozen BatchNorm buffers stay fixed. ``--unfreeze-all`` trains all
layers. A mismatched checkpoint fails unless student-only partial initialization
is requested with ``--allow-partial-checkpoint``; missing keys are reported.
Teachers, calibration and inference always require complete checkpoints.

Optional forgetting controls are off by default:

* ``--genomic-eval-dataset`` measures held-out genomic performance each epoch;
   it does not select checkpoints or contribute gradients.
* ``--rehearsal-dataset --rehearsal-shards N`` mixes labeled genomic shards into
   training. The default count -1 uses all shards.
* ``--distill-weight W --distill-shards anchors.h5`` adds soft-target
   cross-entropy from a frozen teacher. The teacher defaults to the initial
   checkpoint; ``--distill-teacher`` selects another compatible complete model.
* ``--l2sp W`` penalizes trainable-weight drift toward the teacher and requires
   positive distillation weight. ``--weight-decay`` controls decay toward zero.

Anchors use the same X/Y schema as training even when their labels are not used
by the teacher. Partial anchor batches are retained. Keep measurement data
separate from rehearsal/distillation inputs to avoid evaluating trained-on data.

Example
-------

.. code-block:: bash

   openspliceai transfer --pretrained-model model_80nt.pt --train-dataset data/dataset_train.h5 --validation-dataset data/dataset_validation.h5 --test-dataset data/dataset_test.h5 -f 80 --unfreeze 2 --project-name adapted --output-dir transfer

Complete options
----------------

.. include:: ../_generated/cli-transfer.inc

See :doc:`output_explanation` for schemas and :doc:`migration` for changed behavior.
