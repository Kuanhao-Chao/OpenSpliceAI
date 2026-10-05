.. _train_your_own_model_mouse:

Train a Mouse model
===================

Download a genome and GFF3 annotation for the same assembly/release. Record their
checksums and annotation version; use UCSC contig names for the human split,
otherwise choose random chromosome splitting. The following commands show
an 80-base model workflow; a 10000-base model requires more resources.

.. code-block:: bash

   openspliceai create-data --genome-fasta genome.fa --annotation-gff annotation.gff3 --output-dir data --verify-h5

.. code-block:: bash

   openspliceai train --train-dataset data/dataset_train.h5 --validation-dataset data/dataset_validation.h5 --test-dataset data/dataset_test.h5 -f 80 --epochs 10 --project-name example --output-dir training

.. code-block:: bash

   openspliceai calibrate --pretrained-model model_80nt.pt --validation-dataset data/dataset_validation.h5 --test-dataset data/dataset_test.h5 -f 80 --epochs 10 --project-name calibrated --output-dir calibration

Use the actual ``model_best.pt`` path from the training experiment in calibration.
Run :doc:`../openspliceai_predict` or :doc:`../openspliceai_variant` with the same
context. Published species models were trained with their original source/defaults;
these updated commands do not reproduce their training by themselves.
