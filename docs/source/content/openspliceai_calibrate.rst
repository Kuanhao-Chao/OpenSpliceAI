.. _calibrate_subcommand:

calibrate
=========

The ``calibrate`` command fits three class-specific temperatures to a trained
model's logits: one each for non-splice, acceptor and donor predictions.
For class :math:`c`, the scaled logit is :math:`z'_c = z_c / T_c`; softmax
then converts the scaled logits to probabilities. The model weights and
batch-normalization statistics are kept fixed during evaluation.

Calibration fits negative log-likelihood (NLL) on the validation split and
reports NLL, expected calibration error (ECE), per-class Brier scores and
reliability curves on validation and test splits. Lower validation NLL does
not guarantee improved calibration on independent data. Class-specific
scaling can change the predicted class, unlike a single shared temperature.

Inputs and splits
-----------------

Supply a complete PyTorch state dictionary whose architecture matches
``--flanking-size``. Missing or incompatible weights are rejected.
``--train-dataset`` resolves the split directory: replacing ``train`` with
``validation`` in its filename locates the validation HDF5 file. Temperature
fitting uses that validation file; the training file is opened for the
shared dataset-loading interface. The independently supplied
``--test-dataset`` is evaluated after fitting and is not used to fit temperatures.
HDF5 inputs retain the standard ``X{i}`` and nested ``Y{i}`` layout.

.. code-block:: bash

   openspliceai calibrate \
      --pretrained-model model_best.pt \
      --train-dataset dataset_train.h5 \
      --test-dataset dataset_test.h5 \
      --flanking-size 10000 \
      --project-name human_MANE_calibration \
      --output-dir ./calibration_results/ \
      --random-seed 42

Use ``openspliceai calibrate --help`` for the current command-line arguments.
The shared training options ``--epochs``, ``--patience``, ``--early-stopping``
and ``--loss`` do not configure the temperature optimizer. It currently uses
Adam at 0.01, ReduceLROnPlateau with factor 0.1 and patience 2, a maximum of
2000 updates, and early stopping after 2 updates without an NLL improvement
exceeding 1e-6. Temperatures start at 1.0 and are clamped to [0.05, 5.0].

Outputs and reuse
-----------------

``temperature.pt`` contains a three-element tensor, and ``temperature.txt``
records its values. The tensor file is written after fitting; when an existing
file is loaded with ``--temperature-file``, the text and evaluation outputs
are written without copying the input tensor file.

``calibrated_model.pt`` is a serialized full model with its temperature vector.
It is not a state-dictionary checkpoint accepted by ``predict`` or ``variant``.
Load it in Python only from a trusted source (recent PyTorch versions require
an explicit ``weights_only=False`` for full-object loading), or reuse
``temperature.pt`` with ``calibrate --temperature-file`` and the original checkpoint.

Results are written under ``calibration/results/{validation,test}/``. Each
split includes original and scaled metric files, probability distributions,
Brier scores, and per-class calibration arrays and plots. Reliability curves
omit empty bins; their counts align with the occupied bins and include
probabilities of exactly 0 and 1. Scaled curves use their own counts because
temperature scaling can move observations between bins. Shaded confidence
intervals are normal approximations for bin frequencies; they do not measure
uncertainty in the model or the temperature estimate.
