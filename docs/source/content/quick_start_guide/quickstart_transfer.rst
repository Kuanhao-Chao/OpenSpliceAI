.. _quick-start_transfer:

Quick start: transfer
=====================

Use :doc:`../openspliceai_transfer` for input requirements and complete options.

.. code-block:: bash

   openspliceai transfer --pretrained-model model_80nt.pt --train-dataset data/dataset_train.h5 --validation-dataset data/dataset_validation.h5 --test-dataset data/dataset_test.h5 -f 80 --unfreeze 2 --project-name adapted --output-dir transfer

Replace every input path with a matching file. Model context and assembly must agree.
For a fully runnable small example, use ``python examples/tutorial/run.py``
from a repository checkout. See :doc:`../output_explanation` to inspect outputs.
