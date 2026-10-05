.. _quick-start_calibrate:

Quick start: calibrate
======================

Use :doc:`../openspliceai_calibrate` for input requirements and complete options.

.. code-block:: bash

   openspliceai calibrate --pretrained-model model_80nt.pt --validation-dataset data/dataset_validation.h5 --test-dataset data/dataset_test.h5 -f 80 --epochs 10 --project-name calibrated --output-dir calibration

Replace every input path with a matching file. Model context and assembly must agree.
For a fully runnable small example, use ``python examples/tutorial/run.py``
from a repository checkout. See :doc:`../output_explanation` to inspect outputs.
