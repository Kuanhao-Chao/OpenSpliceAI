.. _calibrate_subcommand:

calibrate
=========

Fit three class temperatures on validation logits by minimizing negative log
likelihood. Test data is evaluated after fitting and never updates temperatures.
Supply ``--validation-dataset`` directly; ``--train-dataset`` is a legacy filename
shorthand and its training contents are not used. Calibration accepts NLL only.

``--epochs``, ``--early-stopping`` and ``--patience`` control the optimization.
The base model stays in evaluation mode. Gradients accumulate over disk-backed
chunks; validation logits are computed once. Temperatures stay in [0.05,5] and
the best post-update NLL is restored, with identity T=1 as the initial candidate.
``--temperature-file`` restores an existing three-element tensor instead of fitting.

``calibrated_model.pt`` contains base weights, class order, context and temperatures
in a versioned portable dictionary. Use it directly with ``predict`` or PyTorch
``variant`` at the matching context. Raw base checkpoints remain supported.
Never use a pickled full model object as an inference artifact.

NLL, ECE, class Brier scores and reliability-bin counts use the entire selected
split. Score histograms use an aligned deterministic sample of at most 100000
observations; JSON summaries label the two scopes. Temporary logits require disk
space proportional to observations (approximately 20 bytes per position before
HDF5 overhead) and are removed after success or failure.

Example
-------

.. code-block:: bash

   openspliceai calibrate --pretrained-model model_80nt.pt --validation-dataset data/dataset_validation.h5 --test-dataset data/dataset_test.h5 -f 80 --epochs 10 --project-name calibrated --output-dir calibration

Complete options
----------------

.. include:: ../_generated/cli-calibrate.inc

See :doc:`output_explanation` for schemas and :doc:`migration` for changed behavior.
