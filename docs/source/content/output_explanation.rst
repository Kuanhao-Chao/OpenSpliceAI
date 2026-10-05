.. _output_files:

Output files
============

Datasets
--------

``datafile_<split>.h5`` stores parallel NAME/CHROM/STRAND/TX_START/TX_END/SEQ/LABEL
fields. Labels are 0=non-splice, 1=acceptor, 2=donor in transcript orientation.
``dataset_<split>.h5`` stores paired Xn/Yn shards; see :doc:`openspliceai_create-data`.
Metadata is ignored when enumerating shard pairs. Unpaired shards or incompatible
shapes fail explicitly. A zero-vector nucleotide means unknown/padding; a
zero-vector label is padding in the legacy dataset representation.
Losses, classification/ranking metrics and calibration counts exclude these
unobserved label positions; adding padding does not change their denominators.

Training and transfer
---------------------

Experiment paths encode project, context, experiment and seed. ``model_<epoch>.pt``
and ``model_best.pt`` are raw state dictionaries. TRAIN/VALIDATION/TEST logs include
complete-split loss, class precision/recall/F1, AUPRC, top-k and learning rates.
The historical ``accuracy`` field is macro class recall (including zero recall for
absent classes); ``*_accuracy`` is that class's recall. Absent-positive ranking
metrics are ``nan`` and support counts are logged. GENOMIC logs are optional
transfer measurement results and do not control checkpoint selection.

Calibration
-----------

``temperature.pt`` stores a three-value tensor; ``calibrated_model.pt`` is a
versioned primitive/tensor dictionary. ``optimization.json`` records NLL history,
restored best temperatures and options. ``calibration/results/<split>/`` holds
metrics, reliability curves, plots and scope-aware ``summary.json``.

Prediction
----------

``acceptor_predictions.bed`` and ``donor_predictions.bed`` use BED6, including
plain FASTA entries. Scores are probabilities, not conventional 0–1000 BED scores.
Stored ``predict.h5``/``predict.pt`` probability arrays include padded window tails;
BED output excludes those tails and unowned split halos. Intermediate NAME/LEN
metadata is needed to interpret stored arrays. Output can remain partial if
prediction is interrupted; only variant VCF and model checkpoints are atomic.

Variant VCF
-----------

``OpenSpliceAI`` INFO strings have ten fields:

.. code-block:: text

   ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL

Each overlapping gene/alternate allele can contribute one value. DS is a signed
probability difference and DP an integer offset from the VCF position. Unscored
records remain in the file and must not be counted as scored negatives in analysis.
