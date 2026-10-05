.. _quick-start_train:

Quick start: train
==================

Use :doc:`../openspliceai_train` for input requirements and complete options.

.. code-block:: bash

   openspliceai train --train-dataset data/dataset_train.h5 --validation-dataset data/dataset_validation.h5 --test-dataset data/dataset_test.h5 -f 80 --epochs 10 --project-name example --output-dir training

Replace every input path with a matching file. Model context and assembly must agree.
For a fully runnable small example, use ``python examples/tutorial/run.py``
from a repository checkout. See :doc:`../output_explanation` to inspect outputs.
